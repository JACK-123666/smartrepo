"""
上下文管控器 (Context Governor) — 管理上下文窗口的预算分配与裁剪。

============================================================
为什么需要 ContextGovernor？
============================================================

在 agent 循环中，每次调用 LLM 时需要发送以下内容：
  1. system prompt（系统级指令，如角色定义、行为准则）
  2. tool definitions（可用工具的定义/JSON Schema）
  3. conversation history（对话历史，包含 user/assistant/tool 消息）
  4. 可能的 file contents（注入的文件内容）

这些内容的总 token 数不能超过模型的上下文窗口限制。
ContextGovernor 的角色是"调度中心"：
  - 从配置中读取各层的预算比例（如 system 占 5%，history 占 70%）
  - 计算 system prompt + tools 的固定开销
  - 用剩余预算来"治理"对话历史（通过 ContextPruner 裁剪）
  - 最终组装出可以直接发送给 LLM 的消息列表

设计思路：预算分配在初始化时算好，govern() 调用时只做
        实际计数和裁剪，保证每次调用的开销可控。
============================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

from smart_repo.config import Config
from smart_repo.context.token_counter import TokenCounter
from smart_repo.context.pruner import ContextPruner, PruningResult
from smart_repo.models.base import Message


@dataclass
class ContextBudget:
    """上下文 Token 预算 — 按层分配的总 token 上限。

    一个完整的上下文窗口被分为 4 层，每层有各自的预算上限：
      - system: 系统提示（角色定义、行为准则等）
      - tools:   工具定义（JSON Schema）
      - history: 对话历史（user/assistant/tool 消息）
      - files:   注入的文件内容
    """

    total: int       # 总预算（模型上下文窗口大小）
    system: int      # system prompt 预算
    tools: int       # 工具定义预算
    history: int     # 对话历史预算
    files: int       # 文件内容预算

    @classmethod
    def from_config(cls, config: Config) -> ContextBudget:
        """从配置对象构建预算分配。

        从 Config 中读取 max_context_tokens 和各层比例，
        计算每层的绝对值（int）。
        """
        total = config.max_context_tokens
        return cls(
            total=total,
            system=int(total * config.system_budget_ratio),
            tools=int(total * config.tools_budget_ratio),
            history=int(total * config.history_budget_ratio),
            files=int(total * config.files_budget_ratio),
        )


@dataclass
class GovernedContext:
    """管控后的上下文 — 可以直接发送给 LLM 的消息集合。

    Attributes:
        system_prompt: 系统级指令文本。
        messages: 组装后的消息列表（system prompt + 裁剪后的历史）。
        tool_schemas: 工具定义列表。
        total_tokens: 所有内容的总 token 数。
        budget: 当前使用的预算分配方案。
        pruning_result: 裁剪操作的详细结果（无裁剪时为 None）。
    """

    system_prompt: str
    messages: list[Message]
    tool_schemas: list[dict]
    total_tokens: int
    budget: ContextBudget
    pruning_result: PruningResult | None = None

    @property
    def is_within_budget(self) -> bool:
        """检查当前上下文是否在总预算以内。"""
        return self.total_tokens <= self.budget.total


class ContextGovernor:
    """上下文管控器 — 预算分配、裁剪、组装的总调度中心。

    职责：
      1. 根据配置初始化预算分配方案（ContextBudget）
      2. 在每次 govern() 调用中：
         a. 计算 system prompt + tools 的固定 token 开销
         b. 计算出对话历史可用的剩余预算
         c. 委托 ContextPruner 裁剪对话历史
         d. 组装最终的 GovernedContext（system prompt + 裁剪后的历史）
      3. 对外提供 should_summarize / get_compression_stats 等查询接口

    使用方法:
        gov = ContextGovernor(config)
        ctx = gov.govern(system_prompt, messages, tool_schemas=schemas, model="claude-sonnet-4-6")
        # ctx.messages 可直接发送给 LLM
    """

    def __init__(self, config: Config) -> None:
        """初始化上下文管控器。

        Args:
            config: 全局配置对象，包含模型名、上下文限制、预算比例等。
        """
        self.config = config
        self.counter = TokenCounter()
        # 裁剪器使用配置中的目标压缩率
        self.pruner = ContextPruner(
            token_counter=self.counter,
            target_compression=config.target_compression_ratio,
        )
        self.budget = ContextBudget.from_config(config)

    def govern(
        self,
        system_prompt: str,
        messages: list[Message],
        tool_names: list[str] | None = None,
        tool_schemas: list[dict] | None = None,
        model: str = "",
        context_limit: int | None = None,
    ) -> GovernedContext:
        """组装并管控完整的上下文窗口。

        这是 ContextGovernor 的核心方法，每次向 LLM 发送请求前调用。
        执行流程：
          1. 确定模型和上下文限制（可用配置默认值或参数覆盖）
          2. 计算 system prompt 和 tools 的固定 token 开销
          3. 计算对话历史可用的剩余预算（扣除固定开销 + 500 token 安全缓冲）
          4. 调用 ContextPruner 裁剪对话历史
          5. 组装最终消息列表（system prompt 放在最前面）

        Args:
            system_prompt: 系统级指令（角色定义、行为准则等）。
            messages: 完整的对话历史。
            tool_names: 活跃工具名列表（用于 token 估算），可省略。
            tool_schemas: 工具定义（JSON Schema 列表），发送给 LLM 的工具描述。
            model: 模型标识符，为空时使用配置中的默认值。
            context_limit: 覆盖默认的上下文限制，为 None 时使用预算中的 total。

        Returns:
            GovernedContext: 包含裁剪后消息列表和统计指标的管控结果。
        """
        model = model or self.config.default_model
        context_limit = context_limit or self.budget.total

        tool_schemas = tool_schemas or []

        # 1. 计算固定开销：system prompt 的 token 数
        system_tokens = self.counter.count_text(system_prompt, model)
        tools_tokens = self._estimate_tools_tokens(tool_schemas, model)

        # 2. 计算对话历史可用的剩余预算
        #    500 token 的缓冲用于防止边界误差（token 估算不精确时留有余地）
        available_for_content = context_limit - system_tokens - tools_tokens - 500

        # 防御：如果固定开销已经超过总预算（不应发生，但做兜底处理）
        if available_for_content <= 0:
            available_for_content = max(1000, context_limit // 2)

        # 3. 裁剪对话历史以适配剩余预算
        pruning_result = self.pruner.prune(
            messages=messages,
            max_tokens=available_for_content,
            model=model,
        )

        # 4. 组装最终消息列表：system prompt 在最前面
        final_messages = [Message.system(system_prompt)] + pruning_result.messages

        # 5. 计算总 token 数
        total_tokens = (
            system_tokens
            + pruning_result.pruned_tokens
            + tools_tokens
        )

        return GovernedContext(
            system_prompt=system_prompt,
            messages=final_messages,
            tool_schemas=tool_schemas,
            total_tokens=total_tokens,
            budget=self.budget,
            pruning_result=pruning_result,
        )

    def _estimate_tools_tokens(self, schemas: list[dict], model: str) -> int:
        """估算工具定义（JSON Schema）的 token 数。

        将 schemas 列表序列化为 JSON 字符串，然后用 TokenCounter 计数。
        这是一种近似估算，因为 LLM 实际编码工具定义的方式可能不同，
        但通常 JSON 序列化后的长度与 token 数成线性关系。
        """
        if not schemas:
            return 0
        import json
        text = json.dumps(schemas)
        return self.counter.count_text(text, model)

    def should_summarize(self, token_count: int) -> bool:
        """判断是否需要触发摘要/裁剪。

        当对话历史的 token 数超过配置中的 summarization_threshold 时，
        建议进行裁剪操作。调用方可据此决定是否提前触发裁剪，
        而非等到预算完全不够时再被动裁剪。
        """
        return token_count >= self.config.summarization_threshold

    def get_compression_stats(self) -> dict:
        """返回当前预算分配和裁剪策略的统计信息。

        用于调试/监控，可查看各层预算分配比例和裁剪阈值。
        """
        return {
            "budget_total": self.budget.total,
            "budget_system": self.budget.system,
            "budget_tools": self.budget.tools,
            "budget_history": self.budget.history,
            "budget_files": self.budget.files,
            "summarization_threshold": self.config.summarization_threshold,
            "target_compression": self.config.target_compression_ratio,
        }
