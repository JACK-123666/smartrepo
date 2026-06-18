"""工具协议——标准化工具系统的基础定义。

============================================================
工具注册 → 参数校验 → 审批 → 执行的完整流程
============================================================

SmartRepo 的工具系统围绕一条严格的执行管道设计，确保 agent 无法绕过安全控制：

1. 工具注册 (Registration)
   - 各模块通过 register_*() 函数创建 Tool 实例并注册到 ToolRegistry。
   - 注册时同时定义：名称、描述、JSON Schema 参数、风险等级、是否需要审批。
   - 示例：shell.py 的 register_shell_tools()、git_tools.py 的 register_git_tools()。
   - ToolRegistry 集中管理所有可用工具，负责索引和去重。

2. 参数校验 (Validation)
   - 当 LLM 通过 function calling 调用工具时，框架根据 Tool.parameters 中定义的
     JSON Schema 对 LLM 提供的参数进行验证（通过 ParameterValidator）。
   - 验证内容：必填参数是否存在、类型是否匹配（string/integer/boolean/array/object）。
   - 校验失败 → 返回错误信息给 LLM（不会进入审批流程），要求 LLM 修正参数后重试。

3. 审批门禁 (Approval)
   - 根据 Tool.risk_level 和 Tool.requires_approval 决定是否需要人工审批：
     - low 风险 + requires_approval=False → 自动执行（如 git_status、git_diff）
     - medium 风险 + requires_approval=True  → 需要用户审批（如文件写入）
     - high 风险 + requires_approval=True    → 强制审批，不可绕过（如 shell、commit）
   - 审批门禁在参数校验通过之后、handler 执行之前介入。

4. 工具执行 (Execution)
   - 通过审批后，调用 Tool.execute(**validated_params)。
   - execute() 内部调用绑定在 Tool.handler 上的异步函数。
   - handler 执行实际逻辑（如读取文件、执行 Shell、操作 Git）。
   - 执行结果封装为 ToolResult，统一包含：成功/失败状态、输出内容、错误信息。
   - ToolResult.truncate() 防止超长输出撑爆 LLM 上下文窗口（默认 8000 字符）。

管道图示:
  LLM调用 → ToolRegistry.lookup(name) → ParameterValidator.validate(params, schema)
         → Approval.check(risk_level, requires_approval) → Tool.execute(**params)
         → ToolResult.truncate() → 返回给 LLM

============================================================
核心数据类
============================================================

本模块定义了 Tool 和 ToolResult 两个核心数据类，是 SmartRepo 工具系统的基石。
设计理念：
  - 所有工具都通过 Tool 数据类统一描述（名称、描述、参数 schema、处理器、风险等级）。
  - ToolResult 统一封装执行结果，将成功/失败/错误信息结构化，便于上游统一处理。
  - 这样的设计使得工具注册、Schema 生成、OpenAI/Claude 格式转换都在同一套抽象上运作。

Tool protocol — base definitions for the standardized tool system.
This module provides the two core dataclasses (Tool and ToolResult) that form the
foundation of SmartRepo's tool system. Every agent capability — file I/O, shell
execution, git operations — is represented as a Tool instance with a uniform shape.

=============================
Tool pipeline: Register → Validate → Approve → Execute
=============================

1. Registration — modules register Tool instances with ToolRegistry.
2. Validation — ParameterValidator checks params against JSON Schema.
3. Approval — risk_level + requires_approval determine if human consent is needed.
4. Execution — handler runs the actual logic, result wrapped as ToolResult.

Pipeline diagram:
  LLM call → ToolRegistry.lookup(name) → ParameterValidator.validate(params, schema)
           → Approval.check(risk_level, requires_approval) → Tool.execute(**params)
           → ToolResult.truncate() → return to LLM
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

# Tool handler signature: async callable receiving kwargs → str
# 工具处理器签名：异步可调用对象，接收关键字参数，返回字符串
ToolHandler = Callable[..., Awaitable[str]]


@dataclass
class Tool:
    """标准化的工具定义。

    每一个 Tool 实例代表代理（agent）可以调用的一个操作，例如读取文件、执行 Shell 命令等。
    职责：
      - 描述工具的元信息（名称、用途说明、参数 Schema）
      - 绑定异步处理器（handler），在调用 execute() 时实际执行
      - 标注风险等级（risk_level）和是否需要人工审批（requires_approval）

    使用方式：
        tool = Tool(
            name="read_file",
            description="读取工作区内的文件内容",
            parameters={...},       # JSON Schema 格式的参数定义
            handler=my_async_func,
            risk_level="low",
        )
        result = await tool.execute(path="some/file.txt")

    Standardized tool definition.

    Attributes:
        name: Unique tool identifier (e.g. "read_file", "shell_exec").
              工具的唯一标识符（如 "read_file"、"shell_exec"）。
        description: Natural language description for the model.
                     给模型阅读的自然语言描述。
        parameters: JSON Schema for the tool's input parameters.
                    工具输入参数的 JSON Schema 定义。
        handler: Async callable that executes the tool.
                 执行该工具逻辑的异步可调用对象。
        risk_level: "low" | "medium" | "high" — for security approval.
                    风险等级：低(low)/中(medium)/高(high)，用于安全审批决策。
        requires_approval: Whether this tool requires human approval.
                           该工具是否需要人工审批后才能执行。
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    risk_level: str = "low"
    requires_approval: bool = False

    def to_openai_schema(self) -> dict[str, Any]:
        """转换为 OpenAI 函数调用格式的 Schema。

        返回值可直接放入 Chat Completion 请求的 tools 字段中。

        Convert to OpenAI function-calling schema.
        Returns a dict compatible with OpenAI's tools/functions API.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """转换为简化的字典表示（不含 handler 和 type 包装）。

        用于内部传递、序列化或简化的工具展示场景。
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    async def execute(self, **kwargs: Any) -> str:
        """使用已验证的参数执行该工具。

        调用绑定的 handler 并返回其字符串结果。
        注意：参数应在调用此方法之前经过验证（通过 ParameterValidator）。

        Execute the tool with validated parameters.
        Delegates to the bound async handler and returns its string output.
        """
        return await self.handler(**kwargs)


@dataclass
class ToolResult:
    """工具执行结果的数据类。

    统一封装每次工具调用的结果，无论成功还是失败。
    职责：
      - 记录工具名称、成功/失败状态、输出内容
      - 可选的错误信息（失败时填充）
      - 可扩展的元数据字典（如截断信息、调用ID等）
      - 提供截断方法，防止超长输出撑爆上下文窗口

    Result of a tool execution.
    Wraps both successful and failed executions in a consistent structure.
    """

    tool_name: str
    success: bool
    output: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_message(self, tool_call_id: str = "") -> dict[str, Any]:
        """转换为对话消息字典。

        用于将工具结果注入回对话历史（role="tool"），
        如果存在错误信息会拼接到 content 最前面。

        Convert to a message dict for the conversation.
        Prepends error info to content when present.
        """
        content = self.output
        if self.error:
            content = f"Error: {self.error}\n\n{content}"
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": self.tool_name,
            "content": content,
        }

    def truncate(self, max_chars: int = 8000) -> ToolResult:
        """截断输出内容到指定字符数。

        超过 max_chars 时截断并添加截断提示信息，
        同时在 metadata 中记录原始长度和截断标记。
        默认限制 8000 字符，防止上下文溢出。

        Truncate output to max_chars, adding a note.
        Records original_length and truncated flag in metadata.
        """
        if len(self.output) <= max_chars:
            return self
        truncated = self.output[:max_chars]
        note = f"\n\n[... output truncated: {len(self.output)} → {max_chars} chars ...]"
        return ToolResult(
            tool_name=self.tool_name,
            success=self.success,
            output=truncated + note,
            error=self.error,
            metadata={**self.metadata, "truncated": True, "original_length": len(self.output)},
        )
