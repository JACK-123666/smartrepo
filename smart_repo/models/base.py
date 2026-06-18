"""
抽象基类提供者 —— 所有 LLM 后端的统一接口。

为什么要这样设计：
1. **统一消息格式（Message）**：不同 LLM 提供商（OpenAI、Anthropic 等）的消息格式各异，
   通过统一的 Message dataclass 屏蔽差异，上层代码只需操作一种消息格式。
2. **统一响应格式（ProviderResponse）**：无论底层调用哪个 API，返回的响应结构一致，
   包含文本内容、工具调用、完成原因和用量信息。
3. **抽象基类（BaseProvider）**：定义所有提供商必须实现的核心方法（chat、count_tokens），
   新增提供商只需继承并实现这些方法即可接入系统。
4. **工具 Schema 转换**：提供统一的工具 schema 转换入口，子类可按需覆盖。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from smart_repo.tools.base import Tool


@dataclass
class Message:
    """跨所有提供商的统一消息格式（Unified message format across all providers）。

    职责：
    - 表示对话中的一条消息，角色可以是 system、user、assistant 或 tool。
    - 封装文本内容、工具调用（tool_calls）以及工具返回信息。
    - 提供工厂类方法（system/user/assistant/tool）方便快速创建。

    使用示例：
        msg = Message.user("你好")
        msg = Message.assistant(content="回复", tool_calls=[...])
        msg = Message.tool(content="结果", tool_call_id="abc", name="search")
    """

    role: str  # 消息角色： "system" | "user" | "assistant" | "tool"
    content: str | list[dict[str, Any]]  # 消息内容：纯文本字符串，或多模态内容块列表
    tool_calls: list[dict[str, Any]] | None = None  # 工具调用列表（仅 assistant 角色可能有）
    tool_call_id: str | None = None  # 工具调用 ID（仅 tool 角色使用）
    name: str | None = None  # 工具名称（仅 tool 角色使用，OpenAI 特有字段）

    def to_dict(self) -> dict[str, Any]:
        """将消息序列化为字典格式，供 API 调用时使用。

        Returns:
            dict: 包含 role 和 content 的字典，可选字段仅在非 None 时包含。
        """
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        return d

    @classmethod
    def system(cls, content: str) -> Message:
        """创建一条 system 角色消息，用于设定系统提示词。

        Args:
            content: 系统提示文本。

        Returns:
            Message: role="system" 的消息实例。
        """
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        """创建一条 user 角色消息，代表用户输入。

        Args:
            content: 用户输入的文本。

        Returns:
            Message: role="user" 的消息实例。
        """
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str | None = None,
                  tool_calls: list[dict[str, Any]] | None = None) -> Message:
        """创建一条 assistant 角色消息，代表模型回复。

        Args:
            content: 模型回复的文本内容，可以为 None（仅使用工具调用时）。
            tool_calls: 模型请求的工具调用列表。

        Returns:
            Message: role="assistant" 的消息实例。
        """
        return cls(role="assistant", content=content or "", tool_calls=tool_calls)

    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: str = "") -> Message:
        """创建一条 tool 角色消息，代表工具执行后的返回结果。

        Args:
            content: 工具执行返回的内容。
            tool_call_id: 对应的工具调用 ID，用于匹配请求。
            name: 工具名称（可选）。

        Returns:
            Message: role="tool" 的消息实例。
        """
        return cls(role="tool", content=content, tool_call_id=tool_call_id, name=name)


@dataclass
class ProviderResponse:
    """来自任意提供商的统一响应（Unified response from any provider）。

    职责：
    - 封装 LLM 调用返回的所有信息，无论底层是 OpenAI 还是 Anthropic。
    - 通过 has_tool_calls / is_finished 属性提供便捷判断。

    字段说明：
        content: 模型生成的文本内容。
        tool_calls: 模型请求的工具调用列表（兼容 OpenAI function calling 格式）。
        finish_reason: 停止原因，"stop"（正常结束）、"length"（达到长度限制）、"error"（异常）。
        usage: token 用量统计 {"input_tokens": N, "output_tokens": M, ...}。
    """

    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        """是否包含工具调用请求。

        Returns:
            bool: True 表示模型请求执行工具。
        """
        return len(self.tool_calls) > 0

    @property
    def is_finished(self) -> bool:
        """模型是否正常结束（即 finish_reason == "stop"）。

        Returns:
            bool: True 表示对话回合正常完成。
        """
        return self.finish_reason == "stop"


class BaseProvider(ABC):
    """LLM 提供商的抽象基类（Abstract base class for LLM providers）。

    职责：
    - 定义所有提供商必须遵循的统一接口。
    - 子类必须实现 chat() 和 count_tokens() 方法。
    - 提供工具链的通用辅助方法（_tools_to_schema、_format_tool_result）。

    所有提供商必须实现（All providers MUST implement）：
      - chat(): 发送消息 + 工具定义，返回统一的 ProviderResponse
      - count_tokens(): 返回给定消息列表的 token 数量
      - model_name: 属性，返回当前使用的模型标识符
    """

    def __init__(self, model: str, api_key: str, **kwargs: Any) -> None:
        """初始化提供商。

        Args:
            model: 模型标识符（如 "gpt-4o"、"claude-sonnet-4-6"）。
            api_key: API 密钥。
            **kwargs: 额外的提供商特定配置参数。
        """
        self.model = model
        self.api_key = api_key
        self.extra_config = kwargs

    @property
    @abstractmethod
    def model_name(self) -> str:
        """返回当前使用的模型标识符（Return the active model identifier）。

        Returns:
            str: 模型 ID 字符串。
        """
        ...

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> ProviderResponse:
        """发送聊天请求并返回统一响应（Send a chat request and return a unified response）。

        Args:
            messages: 统一格式的消息列表。
            tools: 可用的工具定义列表，None 表示不使用工具。
            temperature: 采样温度，控制输出随机性（0-2，越高越随机）。
            max_tokens: 模型输出的最大 token 数。

        Returns:
            ProviderResponse: 统一的响应对象，包含文本、工具调用、用量等信息。
        """
        ...

    @abstractmethod
    def count_tokens(self, messages: list[Message]) -> int:
        """计算消息列表的 token 数量（Count tokens for a list of messages）。

        Args:
            messages: 统一格式的消息列表。

        Returns:
            int: 预估的 token 总数。
        """
        ...

    def _tools_to_schema(self, tools: list[Tool]) -> list[dict[str, Any]]:
        """【关键转换】将 Tool 对象列表转换为 OpenAI 兼容的 function schema 格式。

        这是工具定义从内部表示到 API 格式的核心转换点。
        OpenAI 和大多数兼容 API 都使用这种 function-calling schema 格式。

        Args:
            tools: Tool 对象列表。

        Returns:
            list[dict]: OpenAI function-calling 格式的工具 schema 列表。
        """
        return [t.to_openai_schema() for t in tools]

    def _format_tool_result(self, tool_call_id: str, name: str,
                            result: str) -> Message:
        """创建一条工具执行结果的 Message 消息（Create a tool result message）。

        将工具执行的结果封装为统一的 tool 角色消息，用于追加到对话历史中。

        Args:
            tool_call_id: 工具调用 ID，用于关联请求和响应。
            name: 工具名称。
            result: 工具执行的输出内容。

        Returns:
            Message: role="tool" 的消息实例。
        """
        return Message.tool(content=result, tool_call_id=tool_call_id, name=name)
