"""
Anthropic Claude 提供者 —— 基于 Messages API 并支持工具调用（tool-use）。

为什么要这样设计：
1. **Messages API 适配**：Claude API 的消息格式与 OpenAI 有显著差异，需要专门的转换逻辑。
   - system 提示从消息列表中提取为独立的 `system` 参数。
   - assistant 的 tool_calls 需转换为 content 中的 `tool_use` 块。
   - tool 角色消息需转换为 user 角色的 `tool_result` 内容块。
2. **上下文限制管理**：通过字典维护各 Claude 模型的最大上下文窗口（统一为 200K）。
3. **响应解析**：Claude 的响应结构（content block list）需解析为统一的 ProviderResponse 格式，
   特别是 tool_use 块需转换为 OpenAI 兼容的 function calling 格式。
"""

from __future__ import annotations

import json
from typing import Any

from anthropic import AsyncAnthropic, APIError, APITimeoutError

from smart_repo.models.base import (
    BaseProvider, Message, ProviderResponse,
)
from smart_repo.tools.base import Tool


# Anthropic 模型的 token 限制（上下文窗口大小）
# 当前所有 Claude 模型均支持 200,000 token 的上下文
CLAUDE_CONTEXT_LIMITS: dict[str, int] = {
    "claude-fable-5": 200_000,
    "claude-opus-4-8": 200_000,
    "claude-opus-4-7": 200_000,
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
    "claude-3-opus-20240229": 200_000,
}


class ClaudeProvider(BaseProvider):
    """Anthropic Claude 提供者，通过 Messages API 访问（Anthropic Claude provider via Messages API）。

    职责：
    - 将内部统一消息格式转换为 Claude API 所需的格式（含 system 提示提取）。
    - 将工具定义从内部 Tool 对象转换为 Claude 的 tool 格式。
    - 解析 Claude API 响应，产出统一的 ProviderResponse。

    使用方式：
        provider = ClaudeProvider(model="claude-sonnet-4-6", api_key="sk-ant-...")
        response = await provider.chat(messages=[...], tools=[...])
    """

    def __init__(self, model: str, api_key: str, **kwargs: Any) -> None:
        """初始化 Claude 提供者。

        Args:
            model: Claude 模型标识符（如 "claude-sonnet-4-6"）。
            api_key: Anthropic API 密钥。
            **kwargs: 额外的配置参数。
        """
        super().__init__(model, api_key, **kwargs)
        self._client: AsyncAnthropic | None = None  # 延迟初始化的异步客户端

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

        从 CLAUDE_CONTEXT_LIMITS 字典中查找，未找到时默认返回 200,000。

        Returns:
            int: 上下文窗口 token 数。
        """
        return CLAUDE_CONTEXT_LIMITS.get(self.model, 200_000)

    @property
    def client(self) -> AsyncAnthropic:
        """获取或创建 Anthropic 异步客户端（惰性初始化）。

        仅在首次访问时创建客户端实例，后续调用复用同一个实例。

        Returns:
            AsyncAnthropic: 已初始化的异步客户端。
        """
        if self._client is None:
            self._client = AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def chat(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> ProviderResponse:
        """【核心方法】发送聊天请求到 Claude API 并返回统一响应。

        关键转换逻辑：
        1. 从消息列表中提取 system 角色的消息，合并为独立的 system 参数。
        2. 其余角色消息通过 _convert_message 转为 Claude 格式。
        3. 工具定义通过 _tools_to_claude 转为 Claude 的 tool 格式。
        4. API 异常时返回 finish_reason="error" 的 ProviderResponse。

        Args:
            messages: 统一格式的消息列表。
            tools: 可选，工具定义列表。
            temperature: 采样温度（0-2），控制输出随机性。
            max_tokens: 最大输出 token 数。

        Returns:
            ProviderResponse: 统一响应对象。
        """
        # 前置校验 API key：AsyncAnthropic(api_key='') 构造不抛但调用必败，
        # 在此统一返回友好错误响应，避免 SDK 原始异常透传。
        if not self.api_key or not self.api_key.strip():
            return ProviderResponse(
                content="Claude API error: ANTHROPIC_API_KEY is not set. "
                        "Configure it via the ANTHROPIC_API_KEY env var or Config.",
                finish_reason="error",
            )

        system_prompt = ""
        chat_messages: list[dict[str, Any]] = []

        # 遍历消息列表，分离 system 提示和对话消息
        for msg in messages:
            if msg.role == "system":
                # Claude API 的 system 参数是独立的字符串，不是消息列表中的一条
                system_prompt += (msg.content if isinstance(msg.content, str)
                                  else str(msg.content)) + "\n"
            else:
                # 其他角色（user/assistant/tool）转换为 Claude 格式后加入消息列表
                chat_messages.append(self._convert_message(msg))

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": chat_messages,
            "temperature": temperature,
        }
        # 仅在 system_prompt 非空时添加，避免发送空字符串
        if system_prompt.strip():
            kwargs["system"] = system_prompt.strip()
        # 有工具定义时，转换为 Claude 格式并加入请求
        if tools:
            kwargs["tools"] = self._tools_to_claude(tools)

        try:
            response = await self.client.messages.create(**kwargs)
        except (APIError, APITimeoutError) as e:
            # API 调用失败时返回错误响应，不抛出异常，保证上层调用稳定性
            return ProviderResponse(
                content=f"Claude API error: {e}",
                finish_reason="error",
            )

        return self._parse_response(response)

    def count_tokens(self, messages: list[Message]) -> int:
        """【启发式估算】为 Claude 消息估算 token 数量（Estimate token count for Claude messages）。

        实现说明：
        - 这是一个**粗略估算**，并非精确计数。
        - 使用约 3.5 个字符/1 token 的比例（适用于英文文本），另加每条消息 4 token 的
          结构开销。
        - 对于精确计数，应使用 Anthropic 官方 tokenizer 或 API 返回的 usage 数据。

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
            # ~3.5 chars per token for Claude tokenizer (rough)
            # 约 3.5 个字符 ≈ 1 个 token（Claude 分词器的粗略估算）
            # 用 int(len/3.5) 而非 len//3.5：int//float 会返回 float 污染累加器类型
            total += int(len(text) / 3.5)
            total += 4  # message overhead / 每条消息的结构开销（约 4 token）
        return total

    def _convert_message(self, msg: Message) -> dict[str, Any]:
        """【关键转换】将统一 Message 转换为 Claude API 所需的消息格式。

        这是系统中最核心的格式转换逻辑之一。Claude API 与 OpenAI API 的消息格式
        有本质差异，需要逐角色处理：

        - **assistant + tool_calls**：Claude 使用 content 中的 tool_use 块，
          而非独立的 tool_calls 字段。这里将 tool_calls 列表转换为 Claude 的
          tool_use 内容块，与文本内容块合并为 content 数组。
        - **tool**：Claude 中工具结果作为 user 角色的 tool_result 内容块返回，
          而非独立的 tool 角色。这要求将 tool 角色的消息转换为 user 角色。
        - **其他角色**：直接映射 role 和 content 字段。

        Args:
            msg: 内部统一格式的消息。

        Returns:
            dict: Claude API 兼容的消息字典。
        """
        if msg.role == "assistant" and msg.tool_calls:
            # 【工具调用转换】assistant 请求工具调用时：
            # OpenAI 格式在顶层有独立的 tool_calls 字段，
            # Claude 格式则将 tool_use 块嵌入 content 数组中
            claude_tool_use: list[dict[str, Any]] = []
            claude_content: list[dict[str, Any]] = []
            for tc in msg.tool_calls:
                func = tc.get("function", tc)
                tool_input = func.get("arguments", {})
                if isinstance(tool_input, str):
                    # arguments 可能是 JSON 字符串，Claude 需要解析后的 dict
                    # 处理空字符串或非法 JSON（防御模型偶尔返回错误格式）
                    if not tool_input.strip():
                        tool_input = {}
                    else:
                        try:
                            tool_input = json.loads(tool_input)
                        except json.JSONDecodeError:
                            tool_input = {}
                claude_tool_use.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": func.get("name", ""),
                    "input": tool_input,
                })
            # 如果有文本内容，也加入 content 数组
            if msg.content:
                claude_content.append({
                    "type": "text",
                    "text": msg.content if isinstance(msg.content, str) else str(msg.content),
                })
            # content 是文本块 + tool_use 块的混合数组
            return {"role": "assistant", "content": claude_content + claude_tool_use}

        if msg.role == "tool":
            # 【工具结果转换】tool 角色 → user 角色 + tool_result 内容块
            # Claude 不支持独立的 tool 角色，工具结果必须以 user 身份提交
            return {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id or "",
                    "content": msg.content if isinstance(msg.content, str) else str(msg.content),
                }],
            }

        # 简单角色（user、无 tool_calls 的 assistant）：直接映射
        content = msg.content
        if isinstance(content, str):
            return {"role": msg.role, "content": content}
        return {"role": msg.role, "content": str(content)}

    def _tools_to_claude(self, tools: list[Tool]) -> list[dict[str, Any]]:
        """【工具 Schema 转换】将 Tool 对象列表转换为 Claude API 的 tool 格式。

        Claude 的 tool 格式与 OpenAI function calling 不同：
        - Claude 使用 "name" / "description" / "input_schema" 三字段。
        - OpenAI 使用嵌套的 function 对象和 parameters 字段。
        - 这里将 Tool 对象的 name、description、parameters 直接映射到 Claude 格式。

        Args:
            tools: Tool 对象列表。

        Returns:
            list[dict]: Claude API 兼容的工具定义列表。
        """
        return [{
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        } for t in tools]

    def _parse_response(self, response: Any) -> ProviderResponse:
        """【响应解析】将 Claude API 的原始响应解析为统一的 ProviderResponse。

        关键转换：
        1. **内容提取**：遍历 response.content 块列表，text 块拼接到 content_text，
           tool_use 块转换为 OpenAI 兼容的 function calling 格式。
        2. **Tool use → Function call**：将 Claude 的 tool_use 块映射为
           {id, type:"function", function:{name, arguments}} 的标准格式。
        3. **用量提取**：从 response.usage 提取 input_tokens 和 output_tokens。
        4. **停止原因**：从 response.stop_reason 映射到 finish_reason。

        Args:
            response: Claude API 的原始响应对象。

        Returns:
            ProviderResponse: 统一格式的响应。
        """
        content_text = ""
        tool_calls: list[dict[str, Any]] = []

        # 遍历 Claude 返回的 content 块列表
        for block in response.content:
            if block.type == "text":
                # 文本块：直接拼接
                content_text += block.text
            elif block.type == "tool_use":
                # 【关键转换】Claude tool_use → OpenAI function calling 格式
                # Claude 的 input 是 dict，需序列化为 JSON 字符串以兼容 OpenAI 格式
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                })

        # 提取 token 用量信息（Anthropic 不返回 total_tokens，需手动求和）
        usage = {}
        if hasattr(response, "usage"):
            _inp = getattr(response.usage, "input_tokens", 0)
            _out = getattr(response.usage, "output_tokens", 0)
            usage = {
                "input_tokens": _inp,
                "output_tokens": _out,
                "total_tokens": _inp + _out,
            }

        # 提取停止原因，归一化到统一格式
        # Anthropic stop_reason: "end_turn" | "max_tokens" | "tool_use" | "stop_sequence"
        # 将 "end_turn" 映射为 "stop"（表示模型正常结束），其余保留原值
        finish_reason = "stop"
        if hasattr(response, "stop_reason"):
            raw = response.stop_reason or "stop"
            finish_reason = "stop" if raw == "end_turn" else raw

        return ProviderResponse(
            content=content_text,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )
