"""
分层上下文裁剪模块 — 智能压缩对话上下文，确保不超过模型的上下文窗口限制。

============================================================
设计背景：为什么需要分层裁剪？
============================================================

LLM 的上下文窗口是有限的（如 128K、200K tokens）。在 agent 循环中，
对话历史不断增长（用户消息 + 助手回复 + 工具调用/结果），很容易超出窗口。
简单地丢弃旧消息会丢失关键信息（如系统提示、工具输出中的关键数据）。

因此本模块采用**分层裁剪策略**，从"损耗最小"到"损耗最大"逐层尝试，
目标是：在保留任务关键信息的前提下，把 token 数压到预算以内。

============================================================
4 层裁剪策略（按优先级从高到低，逐一尝试）
============================================================

第 1 层：工具结果截断 (Tool Result Truncation)
  操作：将旧的工具返回内容截断到 tool_result_max_chars（默认 2000 字符），
        保留开头部分，末尾附加截断说明。
  为什么优先：工具结果通常很长（如文件内容、shell 输出），但很多时候
              只需要开头部分就够了，中间/末尾是冗余数据。
  损耗评估：低 — 信息仍在，只是被截断了，开头通常包含了最重要的输出。

第 2 层：历史摘要压缩 (History Summarization)
  操作：将较老的消息（前半部分或前 2/3）压缩为一条摘要消息（system 角色），
        只保留最近的消息保持原样。
  为什么排第二：摘要比直接丢弃好——至少保留了"曾经讨论过什么"的语义，
                  但具体细节会丢失。
  损耗评估：中 — 丢失细节，但保留了讨论的脉络和关键点。

第 3 层：丢弃旧消息 (Message Dropping)
  操作：从最旧的非 system 消息开始逐条丢弃，直到 token 数达标。
        不会丢弃 system 消息和最近 2-3 条消息（保持上下文连贯性）。
  为什么排第三：这是"有损压缩"，信息彻底丢失了。
                  但有时对话太长，不丢就无法继续。
  损耗评估：高 — 旧信息永久丢失。

第 4 层：激进截断 (Aggressive Truncation)
  操作：将所有消息内容截断到 500 字符以内。
  为什么是最后手段：这几乎破坏了所有语义信息，只应在前 3 层都不够用时使用。
                    通常意味着 token 预算设置得太小，或对话实在太长。
  损耗评估：极高 — 大部分内容被丢弃，仅保留开头片段。

============================================================
目标：35%+ token 压缩率，同时保留任务关键信息。
============================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from smart_repo.context.token_counter import TokenCounter
from smart_repo.models.base import Message


@dataclass
class PruningResult:
    """裁剪操作的结果。

    Attributes:
        messages: 裁剪后的消息列表。
        original_tokens: 裁剪前的 token 总数。
        pruned_tokens: 裁剪后的 token 总数。
        compression_ratio: 压缩率（0.0-1.0），0 表示无压缩，1 表示完全清空。
        actions_taken: 实际执行的裁剪动作列表（如 "Truncated 3 tool results"）。
    """

    messages: list[Message]
    original_tokens: int
    pruned_tokens: int
    compression_ratio: float
    actions_taken: list[str] = field(default_factory=list)

    @property
    def tokens_saved(self) -> int:
        """节约的 token 数。"""
        return self.original_tokens - self.pruned_tokens


class ContextPruner:
    """分层上下文裁剪引擎。

    按优先级从低损耗到高损耗，逐层尝试裁剪，直到消息 token 数达到预算。

    4 层策略（按顺序执行，每层执行后检查是否达标，达标即停止）：
      1. Tool result truncation  — 截断旧的工具返回结果
      2. History summarization   — 摘要压缩旧对话
      3. Message dropping        — 丢弃最旧的非关键消息
      4. File content truncation — 激进截断所有消息内容
    """

    def __init__(
        self,
        token_counter: TokenCounter | None = None,
        target_compression: float = 0.35,
        tool_result_max_chars: int = 2000,
        summarization_model: str = "gpt-4o-mini",
    ) -> None:
        """初始化裁剪器。

        Args:
            token_counter: Token 计数器，不传则自动创建。
            target_compression: 目标压缩率（默认 0.35 = 压缩掉 35%）。
            tool_result_max_chars: 第 1 层：工具结果最大保留字符数。
            summarization_model: 第 2 层：用于生成摘要的模型（轻量模型即可）。
        """
        self.counter = token_counter or TokenCounter()
        self.target_compression = target_compression
        self.tool_result_max_chars = tool_result_max_chars
        self.summarization_model = summarization_model

    def prune(
        self,
        messages: list[Message],
        max_tokens: int,
        model: str = "gpt-4o",
    ) -> PruningResult:
        """执行分层裁剪，将消息列表压缩到 max_tokens 预算以内。

        Args:
            messages: 完整的对话历史。
            max_tokens: 目标 token 预算上限。
            model: 模型标识符，用于精确 token 计数。

        Returns:
            PruningResult: 裁剪后的消息列表及裁剪统计指标。
        """
        original_tokens = self.counter.count(messages, model)
        actions: list[str] = []

        # 如果当前已在预算内，不做任何裁剪
        if original_tokens <= max_tokens:
            return PruningResult(
                messages=messages,
                original_tokens=original_tokens,
                pruned_tokens=original_tokens,
                compression_ratio=0.0,
                actions_taken=["no pruning needed"],
            )

        pruned = list(messages)

        # ============================================================
        # 第 1 层：截断旧的工具返回结果
        # 策略：工具结果通常包含大量输出（如文件内容、命令输出），
        #       但前 2000 字符通常已包含最重要的信息。
        #       截断后附加 "... truncated ..." 标记，让模型知道内容不完整。
        # ============================================================
        pruned, action1 = self._truncate_tool_results(pruned, model)
        if action1:
            actions.append(action1)
        if self._fits(pruned, max_tokens, model):
            return self._result(pruned, original_tokens, max_tokens, model, actions)

        # ============================================================
        # 第 2 层：摘要压缩旧对话
        # 策略：取消息列表的前 2/3 作为"旧消息"，生成一条摘要，
        #       摘要 + 最近 1/3 消息 = 压缩后的消息列表。
        #       这样既保留了近期上下文连贯性，又避免了旧消息占满窗口。
        # ============================================================
        pruned, action2 = self._summarize_old_turns(pruned, model, max_tokens)
        if action2:
            actions.append(action2)
        if self._fits(pruned, max_tokens, model):
            return self._result(pruned, original_tokens, max_tokens, model, actions)

        # ============================================================
        # 第 3 层：丢弃最旧的非关键消息
        # 策略：从最旧的消息开始，逐条丢弃 user/assistant 消息，
        #       保留 system 消息和最近 2 条消息（保持基本上下文）。
        #       这是"有损"操作，旧信息永久丢失。
        # ============================================================
        pruned, action3 = self._drop_oldest(pruned, max_tokens, model)
        if action3:
            actions.append(action3)
        if self._fits(pruned, max_tokens, model):
            return self._result(pruned, original_tokens, max_tokens, model, actions)

        # ============================================================
        # 第 4 层：激进截断 — 所有消息内容截断到 500 字符
        # 策略：前 3 层都失败时的最后手段。将所有超过 500 字符的
        #       消息全部截断。这几乎破坏了所有长内容的语义。
        #       触发此层通常意味着预算设置太小或对话异常长。
        # ============================================================
        pruned, action4 = self._aggressive_truncate(pruned, max_tokens, model)
        if action4:
            actions.append(action4)

        return self._result(pruned, original_tokens, max_tokens, model, actions)

    def _fits(self, messages: list[Message], max_tokens: int,
              model: str) -> bool:
        """检查消息列表的 token 数是否已在预算以内。"""
        return self.counter.count(messages, model) <= max_tokens

    def _ensure_tool_pairs(
        self, messages: list[Message],
    ) -> tuple[list[Message], str]:
        """修复 tool_use/tool_result 配对一致性（裁剪后的兜底）。

        Claude 和 OpenAI 都要求每个 tool_use 恰好有一个 tool_result，反之亦然，
        否则 API 返回 400。裁剪（尤其 _summarize_old_turns / _drop_oldest）可能
        破坏配对。此方法做最后兜底：

        - 孤立的 tool_result（tool_call_id 不在任何 assistant.tool_calls 中）→ 移除
        - 孤立的 tool_use（assistant.tool_calls 中某 id 无对应 tool_result）→ 从该
          assistant 的 tool_calls 中移除该 id；若全部无配对则清空 tool_calls（变为
          纯文本 assistant）。有损兜底，但优于 API 400 崩溃。

        Returns:
            (修复后的消息列表, 动作描述字符串)；无修复时动作为空。
        """
        tool_ids = {
            msg.tool_call_id for msg in messages
            if msg.role == "tool" and msg.tool_call_id
        }
        use_ids: set[str] = set()
        for msg in messages:
            if msg.role == "assistant" and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.get("id"):
                        use_ids.add(tc["id"])

        removed_results = 0
        cleaned_assistants = 0
        result: list[Message] = []
        for msg in messages:
            # 孤立 tool_result：无对应 tool_use → 移除
            if msg.role == "tool" and msg.tool_call_id \
                    and msg.tool_call_id not in use_ids:
                removed_results += 1
                continue
            # assistant 含无配对的 tool_call → 清理 tool_calls
            if msg.role == "assistant" and msg.tool_calls:
                paired = [tc for tc in msg.tool_calls if tc.get("id") in tool_ids]
                if len(paired) != len(msg.tool_calls):
                    cleaned_assistants += 1
                    msg = Message(
                        role="assistant",
                        content=msg.content,
                        tool_calls=paired if paired else None,
                    )
            result.append(msg)

        action = ""
        if removed_results or cleaned_assistants:
            action = (
                f"Repaired tool pairing: removed {removed_results} orphan "
                f"tool_result(s), cleaned {cleaned_assistants} assistant(s)"
            )
        return result, action

    def _result(
        self, pruned: list[Message], original: int, budget: int,
        model: str, actions: list[str],
    ) -> PruningResult:
        """构建裁剪结果对象，计算压缩率和 token 统计。

        Args:
            pruned: 裁剪后的消息列表。
            original: 裁剪前的 token 总数。
            budget: 目标预算（仅用于对比记录，不参与计算）。
            model: 模型标识符。
            actions: 执行过的裁剪动作列表。

        Returns:
            PruningResult: 包含完整裁剪指标的结果对象。
        """
        # 配对一致性兜底：裁剪可能破坏 tool_use/tool_result 配对，修复后再计数
        pruned, pair_action = self._ensure_tool_pairs(pruned)
        if pair_action:
            actions = [*actions, pair_action]
        pruned_tokens = self.counter.count(pruned, model)
        ratio = 1.0 - (pruned_tokens / original) if original > 0 else 0.0
        return PruningResult(
            messages=pruned,
            original_tokens=original,
            pruned_tokens=pruned_tokens,
            compression_ratio=ratio,
            actions_taken=actions,
        )

    def _truncate_tool_results(
        self, messages: list[Message], model: str,
    ) -> tuple[list[Message], str]:
        """第 1 层：截断旧的工具返回消息到 tool_result_max_chars 字符。

        对于 role="tool" 且内容超过阈值的消息：
          - 保留前 tool_result_max_chars 个字符
          - 末尾添加截断说明（原始长度 → 截断后长度）
          - 其他类型的消息原样保留

        Args:
            messages: 待处理的消息列表。
            model: 模型标识符（保留参数，供未来扩展，如区分不同模型的截断策略）。

        Returns:
            (处理后的消息列表, 动作描述字符串)，如无截断则动作描述为空字符串。
        """
        result = []
        truncated = 0
        for msg in messages:
            if msg.role == "tool" and isinstance(msg.content, str) \
                    and len(msg.content) > self.tool_result_max_chars:
                truncated += 1
                shortened = (
                    msg.content[:self.tool_result_max_chars]
                    + f"\n\n[... tool output truncated: "
                    f"{len(msg.content)} → {self.tool_result_max_chars} chars ...]"
                )
                result.append(Message.tool(
                    content=shortened,
                    tool_call_id=msg.tool_call_id or "",
                    name=msg.name or "",
                ))
            else:
                result.append(msg)
        action = f"Truncated {truncated} tool results" if truncated else ""
        return result, action

    def _summarize_old_turns(
        self, messages: list[Message], model: str, budget: int,
    ) -> tuple[list[Message], str]:
        """第 2 层：将较老的对话轮次压缩为一条摘要消息。

        策略：
          - 保留最近 1/3 的消息（最少 4 条）作为"近期上下文"
          - 前 2/3 的消息提取 user/assistant 角色的内容预览（各 200 字符）
          - 将这些预览拼接为一条 system 角色的摘要消息
          - 最多保留 20 条预览，超出部分截断

        Args:
            messages: 待处理的消息列表。
            model: 模型标识符（保留参数）。
            budget: 目标 token 预算（保留参数，供未来智能摘要长度控制）。

        Returns:
            (处理后的消息列表, 动作描述字符串)。
        """
        # 保留最近 1/3 消息（至少 4 条），其余视为"旧消息"
        # Keep last 1/3 of messages, minimum 4
        keep_recent = max(4, len(messages) // 3)

        # 切点对齐：避免在工具回合中间切断。若切点落在 tool 消息上（其对应的
        # assistant tool_use 在 old 段），会把 tool_result 孤立在 recent 段，破坏
        # tool_use/tool_result 配对导致 API 400。向前回退到非 tool 消息，保证
        # recent 段不以孤立 tool_result 开头、use 与 result 不被分到两段。
        split = len(messages) - keep_recent
        while 0 < split < len(messages) and messages[split].role == "tool":
            split -= 1
        if split <= 0:
            # 整段都在一个工具回合内，无法安全切分，放弃摘要
            return messages, ""

        old_portion = messages[:split]
        recent_portion = messages[split:]
        if not old_portion:
            return messages, ""

        # 为每条旧消息生成简短预览（user/assistant 各取前 200 字符）
        summary_parts = []
        for msg in old_portion:
            if msg.role in ("user", "assistant"):
                content_preview = (
                    msg.content[:200]
                    if isinstance(msg.content, str)
                    else str(msg.content)[:200]
                )
                summary_parts.append(f"[{msg.role}]: {content_preview}")

        # 构建摘要消息（显式 if/else，避免三元表达式优先级陷阱：
        # `A + B if cond else ""` 在 cond 为 False 时整体为 ""，导致空摘要）
        header = (
            f"[Context Summary — earlier conversation compressed]\n"
            f"{len(old_portion)} messages summarized. Key points:\n"
        )
        if len(summary_parts) > 20:
            preview = "\n".join(summary_parts[:20])
            more = max(0, len(summary_parts) - 20)
            summary = f"{header}{preview}\n[... {more} more items truncated ...]"
        else:
            summary = header + "\n".join(summary_parts)

        # 摘要消息以 system 角色插入到列表开头
        summary_msg = Message.system(summary)
        result = [summary_msg] + recent_portion
        return result, f"Summarized {len(old_portion)} old messages"

    def _drop_oldest(
        self, messages: list[Message], budget: int, model: str,
    ) -> tuple[list[Message], str]:
        """第 3 层：逐条丢弃最旧的非关键消息，直到 token 数达标。

        策略：
          - 从最旧的非 system、非 tool 消息开始丢弃
          - 优先丢弃 user/assistant 消息
          - 保留最后 2 条消息以维持上下文连贯性
          - 逐条丢弃直到预算满足或无法继续丢弃

        Args:
            messages: 待处理的消息列表。
            budget: 目标 token 预算。
            model: 模型标识符。

        Returns:
            (处理后的消息列表, 动作描述字符串)。
        """
        result = list(messages)
        dropped = 0
        # 持续丢弃直到达标或只剩 3 条消息（保留最小上下文）
        while not self._fits(result, budget, model) and len(result) > 3:
            # 查找第一条可安全丢弃的消息（非 system，且不是最后 2 条）
            for i, msg in enumerate(result):
                # 不丢弃带 tool_calls 的 assistant：单独丢它会留下孤立的 tool_result，
                # 破坏 tool_use/tool_result 配对。带 tool_calls 的回合由后处理兜底。
                if msg.role in ("user", "assistant") and not msg.tool_calls \
                        and i < len(result) - 2:
                    result.pop(i)
                    dropped += 1
                    break  # 每次只丢弃一条，然后重新检查是否达标
            else:
                break  # 没有更多可丢弃的消息
        return result, f"Dropped {dropped} old messages" if dropped else ""

    def _aggressive_truncate(
        self, messages: list[Message], budget: int, model: str,
    ) -> tuple[list[Message], str]:
        """第 4 层（最后手段）：将所有消息内容激进截断到 500 字符。

        策略：
          - 遍历所有消息，超过 500 字符的截断到 500 字符
          - 末尾附加截断说明
          - 不影响 tool_calls 等元数据字段
          - 这是最后的兜底手段，确保无论如何都能返回结果

        Args:
            messages: 待处理的消息列表。
            budget: 目标 token 预算（本方法不实际检查预算，仅做最大压缩）。
            model: 模型标识符（保留参数）。

        Returns:
            (处理后的消息列表, 动作描述字符串)。
        """
        result = []
        for msg in messages:
            if isinstance(msg.content, str) and len(msg.content) > 500:
                shortened = msg.content[:500] + (
                    f"\n[... content aggressively truncated: "
                    f"{len(msg.content)} → 500 chars ...]"
                )
                new_msg = Message(
                    role=msg.role,
                    content=shortened,
                    tool_calls=msg.tool_calls,
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                )
                result.append(new_msg)
            else:
                result.append(msg)
        return result, "Aggressively truncated all message content to 500 chars"
