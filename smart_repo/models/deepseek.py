"""
DeepSeek 提供者 —— 基于 OpenAI 兼容协议接入 DeepSeek（deepseek-chat / deepseek-reasoner）。

DeepSeek 的 API 与 OpenAI Chat Completions 完全兼容，仅 base_url 与模型名不同，
因此直接继承 OpenAIProvider，复用其消息转换（_convert_message）与响应解析
（_parse_response），只覆盖：
  - client：指向 DeepSeek 端点（https://api.deepseek.com）
  - context_limit：DeepSeek 模型的上下文窗口
  - chat()：deepseek-reasoner 不支持 temperature 等采样参数，需剔除

模型：
  - deepseek-chat：DeepSeek-V3，支持 function calling 与 temperature，适合 agent 主力
  - deepseek-reasoner：DeepSeek-R1 推理模型，不支持 temperature / top_p 等采样参数
"""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI, APIError, APITimeoutError

from smart_repo.models.base import Message, ProviderResponse
from smart_repo.models.openai import OpenAIProvider
from smart_repo.tools.base import Tool


# DeepSeek API 端点（OpenAI 兼容）
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# DeepSeek 模型的上下文窗口（token 数）：deepseek-chat / deepseek-reasoner 均为 64K
DEEPSEEK_CONTEXT_LIMITS: dict[str, int] = {
    "deepseek-chat": 64_000,
    "deepseek-reasoner": 64_000,
}


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek 提供者，复用 OpenAI 兼容协议，仅切到 DeepSeek 端点。

    使用方式：
        provider = DeepSeekProvider(model="deepseek-chat", api_key="sk-...")
        response = await provider.chat(messages=[...], tools=[...])
    """

    @property
    def context_limit(self) -> int:
        """返回当前 DeepSeek 模型的上下文窗口大小。"""
        return DEEPSEEK_CONTEXT_LIMITS.get(self.model, 64_000)

    @property
    def client(self) -> AsyncOpenAI:
        """惰性创建指向 DeepSeek 端点的 AsyncOpenAI 客户端。"""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.api_key, base_url=DEEPSEEK_BASE_URL,
            )
        return self._client

    async def chat(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> ProviderResponse:
        """发送聊天请求到 DeepSeek API 并返回统一响应。

        与 OpenAIProvider.chat 的差异：
        - 前置校验 DEEPSEEK_API_KEY（错误消息指明 DeepSeek 而非 OpenAI）
        - deepseek-reasoner 不支持 temperature 等采样参数，需剔除（否则 API 400）
        """
        if not self.api_key or not self.api_key.strip():
            return ProviderResponse(
                content="DeepSeek API error: DEEPSEEK_API_KEY is not set. "
                        "Configure it via the DEEPSEEK_API_KEY env var or Config.",
                finish_reason="error",
            )

        openai_messages = [self._convert_message(m) for m in messages]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "max_tokens": max_tokens,
        }
        # deepseek-reasoner（R1）不支持 temperature / top_p / presence_penalty 等
        # 采样参数，传入会返回 400；deepseek-chat 正常支持。
        if self.model.lower() != "deepseek-reasoner":
            kwargs["temperature"] = temperature
        if tools:
            kwargs["tools"] = self._tools_to_schema(tools)
            kwargs["tool_choice"] = "auto"

        try:
            response = await self.client.chat.completions.create(**kwargs)
        except (APIError, APITimeoutError) as e:
            return ProviderResponse(
                content=f"DeepSeek API error: {e}",
                finish_reason="error",
            )

        return self._parse_response(response)
