"""工具注册中心——负责工具的注册、查询和 Schema 生成。

本模块是 SmartRepo 工具系统的中枢，所有工具（文件操作、Shell、Git 等）都必须
通过 ToolRegistry 注册后才能被代理调用。

设计理念：
  - 集中式注册表：用 dict 维护 name → Tool 的映射，O(1) 查询。
  - 按风险分级索引：_by_risk 字典按 low/medium/high 分组维护工具名列表，
    便于安全审批模块快速查找需要审批的高风险工具。
  - 多格式 Schema 输出：支持 OpenAI 和 Claude 两种格式，适配不同 LLM 后端。
  - 安全的执行包装：execute() 方法捕获所有异常，永远不会向外抛出，确保代理循环
    不会因为一次工具调用异常而崩溃。

Tool registry — registration, lookup, and schema generation.
Acts as the central hub where all tool capabilities are catalogued and
discoverable by the agent loop.
"""

from __future__ import annotations

from typing import Any

from smart_repo.tools.base import Tool, ToolResult


class ToolRegistry:
    """所有代理工具的中央注册表。

    职责：
      - 维护工具名称到 Tool 实例的映射
      - 按风险等级索引工具（便于安全门控）
      - 生成 OpenAI / Claude 兼容的工具 Schema
      - 安全地执行工具（捕获异常，返回 ToolResult）

    使用方式：
        registry = ToolRegistry()
        registry.register(read_file_tool)
        registry.register_many([write_file_tool, shell_tool])

        # 获取给 OpenAI 用的 schemas
        schemas = registry.get_schemas()

        # 执行工具
        result = await registry.execute("read_file", path="foo.py")

        # 查询高风险工具列表
        high_risk = registry.get_high_risk_tools()

    Central registry for all agent tools.

    Usage:
        registry = ToolRegistry()
        registry.register(read_file_tool)
        registry.register(write_file_tool)
        schemas = registry.get_schemas()
    """

    def __init__(self) -> None:
        # 工具名 → Tool 实例的主映射表 / Primary name→Tool mapping
        self._tools: dict[str, Tool] = {}
        # 按风险等级分组的工具名列表，便于安全模块快速查找
        # Tool names grouped by risk level for quick security lookup
        self._by_risk: dict[str, list[str]] = {"low": [], "medium": [], "high": []}

    def register(self, tool: Tool) -> None:
        """注册一个工具。如果同名工具已存在，则覆盖旧定义。

        同时将工具名加入对应风险等级的索引中。

        Register a tool. Overwrites if name already exists.
        Also indexes the tool by its risk level.
        """
        self._tools[tool.name] = tool
        # setdefault 确保风险等级 key 存在 / Ensure the risk-level bucket exists
        self._by_risk.setdefault(tool.risk_level, []).append(tool.name)

    def register_many(self, tools: list[Tool]) -> None:
        """批量注册多个工具。

        内部逐个调用 register()，每个工具的注册都是独立的。

        Register multiple tools at once.
        """
        for t in tools:
            self.register(t)

    def unregister(self, name: str) -> None:
        """按名称移除一个工具。

        会同时从主映射表和风险等级索引中清除该工具。

        Remove a tool by name.
        Cleans up both the main map and the risk-level index.
        """
        if name in self._tools:
            tool = self._tools.pop(name)
            risk_list = self._by_risk.get(tool.risk_level, [])
            if name in risk_list:
                risk_list.remove(name)

    def get(self, name: str) -> Tool | None:
        """按名称获取工具，不存在时返回 None。

        Get a tool by name. Returns None if not found.
        """
        return self._tools.get(name)

    def list_names(self) -> list[str]:
        """返回所有已注册工具的名称列表（按字母排序）。

        Return all registered tool names in sorted order.
        """
        return sorted(self._tools.keys())

    def get_schemas(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        """获取 OpenAI 兼容的函数调用 Schema 列表。

        这是代理循环中最常用的方法——获取工具 Schema 后注入 LLM 请求，
        让模型知道它可以调用哪些工具。

        Args:
            names: 可选，指定要获取的工具名列表。如果为 None，返回所有已注册工具。

        Returns:
            OpenAI function-calling 格式的 Schema 列表。

        Get OpenAI-compatible function schemas for the requested tools.

        Args:
            names: If provided, only return schemas for these tool names.
                   If None, return all registered tools.
        """
        if names is None:
            names = list(self._tools.keys())
        return [self._tools[n].to_openai_schema()
                for n in names if n in self._tools]

    def get_claude_tools(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        """获取 Claude 兼容的工具定义列表。

        与 OpenAI 格式不同，Claude 使用 "input_schema" 而非 "parameters" 字段。
        此方法处理了两种格式之间的差异。

        Get Claude-compatible tool definitions.
        Uses "input_schema" key instead of "parameters" per Claude's API spec.
        """
        if names is None:
            names = list(self._tools.keys())
        return [{
            "name": self._tools[n].name,
            "description": self._tools[n].description,
            "input_schema": self._tools[n].parameters,
        } for n in names if n in self._tools]

    def get_high_risk_tools(self) -> list[str]:
        """返回所有高风险工具的名称列表。

        高风险工具（如 shell、git_commit）始终需要人工审批。

        Return names of high-risk tools that always need approval.
        """
        return list(self._by_risk.get("high", []))

    def get_medium_risk_tools(self) -> list[str]:
        """返回所有中风险工具的名称列表。

        中风险工具在交互模式下需要用户确认。

        Return names of medium-risk tools that need confirmation.
        """
        return list(self._by_risk.get("medium", []))

    async def execute(self, name: str, tool_call_id: str = "",
                      **kwargs: Any) -> ToolResult:
        """按名称执行工具，传入指定参数。

        这是一个"安全执行"方法：
          1. 工具不存在时，返回失败 ToolResult（不会抛出 KeyError）
          2. 工具执行过程中抛出任何异常时，捕获并封装到 ToolResult 中
          3. 无论如何都会返回一个 ToolResult，保证代理循环的稳定性

        Args:
            name: 要执行的工具名称。
            tool_call_id: 关联的工具调用 ID（用于对话消息追踪）。
            **kwargs: 传递给工具 handler 的参数。

        Returns:
            ToolResult：包含成功/失败状态和输出内容。

        Execute a tool by name with the given parameters.

        Returns a ToolResult, never raises — errors are captured in the result.
        This guarantees the agent loop won't crash on a single tool failure.
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                tool_name=name,
                success=False,
                output="",
                error=f"Unknown tool: '{name}'. Available: {', '.join(self.list_names())}",
            )

        try:
            output = await tool.execute(**kwargs)
            return ToolResult(
                tool_name=name,
                success=True,
                output=str(output),
                metadata={"tool_call_id": tool_call_id},
            )
        except Exception as e:
            # 捕获所有异常，包装为失败结果 / Catch all, wrap as failure result
            return ToolResult(
                tool_name=name,
                success=False,
                output="",
                error=f"{type(e).__name__}: {e}",
                metadata={"tool_call_id": tool_call_id},
            )

    def __len__(self) -> int:
        """返回已注册工具的数量。"""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """支持 'name' in registry 的成员检查语法。"""
        return name in self._tools
