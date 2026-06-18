"""
OpenAI GPT 提供者 —— 基于 Chat Completions API 并支持 function calling。

为什么要这样设计：
1. **原生兼容**：OpenAI 的 Chat Completions API 是业界事实标准，统一消息格式（Message）
   与 OpenAI API 格式几乎一一对应，转换逻辑最为简洁。
2. **Function calling 支持**：通过 tool_choice="auto" 启用自动工具选择，OpenAI 的
   function calling 格式也是许多第三方 API 兼容的基础。
3. **上下文管理**：维护各 GPT/O 系列模型的上下文窗口大小，供调用方做截断判断。
4. **错误容错**：API 异常时返回 finish_reason="error" 的 ProviderResponse，
   而非抛出异常，保证上层调用链的稳定性。
"""

from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI, APIError, APITimeoutError

from smart_repo.models.base import (
    BaseProvider, Message, ProviderResponse,
)
from smart_repo.tools.base import Tool


# OpenAI 模型的上下文窗口限制（token 数）
OPENAI_CONTEXT_LIMITS: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-2024-08-06": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4-0125-preview": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo-0125": 16_385,
    "gpt-3.5-turbo": 4_096,
    "o1": 200_000,
    "o1-mini": 128_000,
    "o3-mini": 200_000,
}


class OpenAIProvider(BaseProvider):
    """OpenAI GPT 提供者，通过 Chat Completions API 访问（OpenAI GPT provider via Chat Completions API）。

    职责：
    - 将内部统一消息格式转换为 OpenAI Chat Completions API 格式。
    - 管理工具定义（function calling）的转换。
    - 解析 OpenAI API 响应，产出统一的 ProviderResponse。

    使用方式：
        provider = OpenAIProvider(model="gpt-4o", api_key="sk-...")
        response = await provider.chat(messages=[...], tools=[...])
    """

    def __init__(self, model: str, api_key: str, **kwargs: Any) -> None:
        """初始化 OpenAI 提供者。

        Args:
            model: 模型标识符（如 "gpt-4o"、"o3-mini"）。
            api_key: OpenAI API 密钥。
            **kwargs: 额外的配置参数。
        """
        super().__init__(model, api_key, **kwargs)
        self._client: AsyncOpenAI | None = None  # 延迟初始化的异步客户端

    @property
    def model_name(self) -> str:
        """返回当前使用的模型标识符。

        Returns:
            str: 模型 ID。
        """
        return self.model

    @property
    def context_limit(self) -> int:
        """返回当前模型的上下文窗口大小（token 数）。

        从 OPENAI_CONTEXT_LIMITS 字典中查找，未找到时默认返回 128,000。

        Returns:
            int: 上下文窗口 token 数。
        """
        return OPENAI_CONTEXT_LIMITS.get(self.model, 128_000)

    @property
    def client(self) -> AsyncOpenAI:
        """获取或创建 OpenAI 异步客户端（惰性初始化）。

        仅在首次访问时创建客户端实例，后续调用复用同一个实例。

        Returns:
            AsyncOpenAI: 已初始化的异步客户端。
        """
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def chat(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> ProviderResponse:
        """【核心方法】发送聊天请求到 OpenAI Chat Completions API 并返回统一响应。

        关键转换逻辑：
        1. 所有消息通过 _convert_message 转为 OpenAI 格式（相比 Claude 简单很多）。
        2. 工具定义通过基类的 _tools_to_schema 转为 OpenAI function calling 格式。
        3. tool_choice="auto" 让模型自动决定是否调用工具。
        4. API 异常时返回 finish_reason="error" 的 ProviderResponse。

        Args:
            messages: 统一格式的消息列表。
            tools: 可选，工具定义列表。
            temperature: 采样温度（0-2），控制输出随机性。
            max_tokens: 最大输出 token 数。

        Returns:
            ProviderResponse: 统一响应对象。
        """
        # 前置校验 API key：AsyncOpenAI(api_key='') 构造时即抛 OpenAIError，
        # 而 OpenAIError 是 APIError 的父类，不会被 except (APIError, ...) 捕获，
        # 故在此前置校验，返回友好错误响应而非抛异常。
        if not self.api_key or not self.api_key.strip():
            return ProviderResponse(
                content="OpenAI API error: OPENAI_API_KEY is not set. "
                        "Configure it via the OPENAI_API_KEY env var or Config.",
                finish_reason="error",
            )

        # OpenAI 的消息格式与统一 Message 高度兼容，转换较为简单
        openai_messages = [self._convert_message(m) for m in messages]

        # o1/o3 推理模型不支持 temperature（仅支持 1），且 max_tokens 已废弃，
        # 须改用 max_completion_tokens，否则 API 必返回 400。
        is_reasoning = self.model.lower().startswith(("o1", "o3"))
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
        }
        if is_reasoning:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["temperature"] = temperature
            kwargs["max_tokens"] = max_tokens
        if tools:
            # 将工具定义转为 OpenAI function calling 格式
            kwargs["tools"] = self._tools_to_schema(tools)
            # tool_choice="auto" 让模型自行判断是否需要调用工具
            kwargs["tool_choice"] = "auto"

        try:
            response = await self.client.chat.completions.create(**kwargs)
        except (APIError, APITimeoutError) as e:
            # API 调用失败时返回错误响应，不抛出异常，保证上层调用稳定性
            return ProviderResponse(
                content=f"OpenAI API error: {e}",
                finish_reason="error",
            )

        return self._parse_response(response)

    def count_tokens(self, messages: list[Message]) -> int:
        """【启发式估算】为 OpenAI 消息估算 token 数量（Estimate token count for OpenAI messages）。

        实现说明：
        - 这是一个**粗略估算**，并非精确计数。
        - 使用约 4 个字符/1 token 的比例（GPT 系列分词器，粗略启发式）。
        - 对于精确计数，应使用 tiktoken 库（见 context/counter.py）。

        Args:
            messages: 统一格式的消息列表。

        Returns:
            int: 预估的 token 总数。
        """
        total = 0
        for msg in messages:
            content = msg.content
            if isinstance(content, list):
                text = json.dumps(content)
            elif isinstance(content, str):
                text = content
            else:
                text = str(content)
            total += len(text) // 4  # GPT 分词器约 4 字符/1 token
            total += 4  # message overhead / 每条消息的结构开销（约 4 token）
        return int(total)

    def _convert_message(self, msg: Message) -> dict[str, Any]:
        """【关键转换】将统一 Message 转换为 OpenAI Chat Completions API 格式。

        相比 Claude 的转换，OpenAI 的转换非常简单：
        - role 和 content 直接映射。
        - assistant 的 tool_calls 保留在顶层（与内部格式一致）。
        - tool 角色保留，只需补充 tool_call_id 和 name 字段。

        OpenAI 的消息格式是内部 Message 设计的主要参考来源，
        因此转换几乎不需要结构性变化。

        Args:
            msg: 内部统一格式的消息。

        Returns:
            dict: OpenAI Chat Completions API 兼容的消息字典。
        """
        d: dict[str, Any] = {"role": msg.role, "content": msg.content}

        if msg.role == "assistant" and msg.tool_calls:
            # assistant 有工具调用时，保留 tool_calls 字段
            d["tool_calls"] = msg.tool_calls
        if msg.role == "tool":
            # tool 角色补充 tool_call_id 和 name 字段
            d["tool_call_id"] = msg.tool_call_id
            if msg.name:
                d["name"] = msg.name
        return d

    def _parse_response(self, response: Any) -> ProviderResponse:
        """【响应解析】将 OpenAI API 原始响应解析为统一的 ProviderResponse。

        关键转换：
        1. 从 response.choices[0] 获取第一个选项的消息。
        2. 提取文本内容（可能为 None，当仅返回 tool_calls 时）。
        3. 将 message.tool_calls 映射为统一的 {id, type:"function", function:{name, arguments}} 格式。
        4. 提取 token 用量（prompt_tokens / completion_tokens / total_tokens）。
        5. 提取 finish_reason。

        Args:
            response: OpenAI Chat Completions API 的原始响应对象。

        Returns:
            ProviderResponse: 统一格式的响应。
        """
        choice = response.choices[0]  # 始终取第一个选项
        message = choice.message

        content_text = message.content or ""  # 文本内容可能为 None
        tool_calls: list[dict[str, Any]] = []

        if message.tool_calls:
            # 【关键转换】OpenAI tool_calls → 统一格式
            # OpenAI 的 tool_calls 已经接近统一格式，只需提取字段重组
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })

        # 提取 token 用量信息
        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens or 0,
                "output_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }

        # 提取停止原因，默认 "stop"
        finish_reason = choice.finish_reason or "stop"

        return ProviderResponse(
            content=content_text,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )
