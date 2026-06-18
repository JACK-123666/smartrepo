"""
Token 计数模块 — 统一的 token 计数接口，支持 tiktoken 精确计数 + 启发式回退。

为什么这样设计：
  1. 在 agent 循环中，上下文窗口管理需要实时知道"当前占用多少 token"，
     不同的 LLM 提供商（OpenAI / Anthropic / 其他）使用不同的 tokenizer，
     因此需要一个统一接口，内部根据模型名自动选择计数策略。
  2. 对于 OpenAI 模型，优先使用 tiktoken 精确计数（精确到 token 级）；
     对于 Claude 等无公开 tokenizer 的模型，使用字符数 / 每token字符数 估算，
     用稍微保守的估计值（宁可高估不低估），保证不超出上下文窗口。
  3. 同时提供 budget_allocations 静态工具方法，按比例分配 token 预算，
     供 ContextGovernor 等上层模块使用。
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from smart_repo.models.base import Message


class TokenCounter:
    """Token 计数器，统一的 token 计数接口。

    职责：
      对 OpenAI 模型使用 tiktoken 精确计数，
      对其他模型（如 Claude）使用字符启发式估算。

    使用方法:
        counter = TokenCounter()
        tokens = counter.count(messages, model="gpt-4o")
        # 也支持直接计算纯文本:
        tokens = counter.count_text("Hello, world!", model="claude-sonnet-4-6")
    """

    # 各模型系列的每 token 字符数估算值（启发式回退时使用）
    # Characters-per-token estimates by provider family
    CHAR_PER_TOKEN: dict[str, float] = {
        "claude": 3.5,
        "gpt": 4.0,
        "o1": 4.0,
        "o3": 4.0,
    }

    def __init__(self) -> None:
        """初始化计数器。内部维护 encoding 缓存，避免重复创建 tokenizer。"""
        self._encoding_cache: dict[str, Any] = {}

    def count(self, messages: list[Message], model: str = "gpt-4o") -> int:
        """计算消息列表的总 token 数。

        Args:
            messages: Message 对象列表（对话历史）。
            model: 模型标识符，用于选择计数策略（如 "gpt-4o", "claude-sonnet-4-6"）。

        Returns:
            int: 总 token 数。

        策略：OpenAI 模型走 tiktoken 精确路径，其余模型走字符启发式估算。
        """
        # Uses tiktoken for OpenAI models, heuristic for others.
        if model.startswith(("gpt-", "o1", "o3")):
            return self._count_openai(messages, model)
        return self._count_heuristic(messages, model)

    def count_text(self, text: str, model: str = "gpt-4o") -> int:
        """计算纯文本字符串的 token 数。

        Args:
            text: 纯文本内容。
            model: 模型标识符。

        Returns:
            int: token 数。
        """
        msg = Message.user(text)
        return self.count([msg], model)

    def _count_openai(self, messages: list[Message], model: str) -> int:
        """使用 tiktoken 精确计算 OpenAI 模型的 token 数。

        如果 tiktoken 未安装，则自动回退到启发式估算。
        如果模型名在 tiktoken 中找不到对应编码，则使用 cl100k_base 作为兜底。
        """
        try:
            import tiktoken
        except ImportError:
            # tiktoken 未安装 → 回退到启发式估算
            return self._count_heuristic(messages, model)

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # 模型名未知 → 使用通用 cl100k_base 编码
            encoding = tiktoken.get_encoding("cl100k_base")

        total = 0
        for msg in messages:
            # 每条消息约 4 token 的协议开销（role 标记等）
            # Each message has ~4 tokens of overhead
            total += 4
            content = msg.content
            if isinstance(content, str):
                total += len(encoding.encode(content))
            elif isinstance(content, list):
                # 多模态内容（如 [{"type": "text", "text": "..."}, {"type": "image_url", ...}]）
                for block in content:
                    if isinstance(block, dict):
                        total += len(encoding.encode(json.dumps(block)))
                    else:
                        total += len(encoding.encode(str(block)))
            # Tool calls 定义也会消耗 token
            if msg.tool_calls:
                total += len(encoding.encode(json.dumps(msg.tool_calls)))
        return total

    def _count_heuristic(self, messages: list[Message], model: str) -> int:
        """使用字符数 / 每 token 字符数 来估算 token 数（启发式回退）。

        策略说明：
          - Claude 模型：约 3.5 字符/token（保守估计，真实值约 3.5-4.0）
          - OpenAI 模型：约 4.0 字符/token
          - 未知模型：使用 3.5（最保守，确保不高估上下文容量）

        注意：这是估计值，可能偏高或偏低。宁高不低以防止超出上下文窗口。
        """
        # 根据模型名前缀判断所属系列
        # Determine provider family
        if model.startswith("claude"):
            cpt = self.CHAR_PER_TOKEN["claude"]
        elif model.startswith(("gpt", "o1", "o3")):
            cpt = self.CHAR_PER_TOKEN["gpt"]
        else:
            cpt = 3.5  # 默认保守估计 / Default conservative estimate

        total = 0
        for msg in messages:
            total += 4  # 每条消息的协议开销 / Message overhead
            content = msg.content
            if isinstance(content, list):
                text = json.dumps(content)
            elif isinstance(content, str):
                text = content
            else:
                text = str(content)
            total += len(text) / cpt
            if msg.tool_calls:
                total += len(json.dumps(msg.tool_calls)) / cpt
        return int(total)

    @staticmethod
    def budget_allocations(total_budget: int, ratios: dict[str, float]) -> dict[str, int]:
        """按比例分配 token 预算到各上下文层。

        这是一个纯工具方法，供 ContextGovernor 使用，
        将总预算按 system/history/tools/files 等层按比例切分。

        Args:
            total_budget: 总可用 token 数。
            ratios: 层名 → 比例 的字典（所有比例之和应约等于 1.0）。
                    例如：{"system": 0.05, "history": 0.70, "tools": 0.10, "files": 0.15}

        Returns:
            层名 → 该层的最大 token 数的字典。
        """
        return {layer: int(total_budget * ratio)
                for layer, ratio in ratios.items()}
