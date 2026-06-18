"""
模型注册中心 —— 动态提供商加载与管理。

为什么要这样设计：
1. **模型到提供商的映射**：通过注册机制建立模型 ID（如 "gpt-4o"）到提供商类（如 OpenAIProvider）
   的映射关系，实现"按名创建"的便捷接口。
2. **自动注册内置模型**：初始化时自动注册 Claude 和 OpenAI 的已知模型，
   用户无需手动配置即可使用主流模型。
3. **启发式匹配**：对于未注册的模型 ID，通过前缀推断（"gpt" → OpenAI，"claude" → Claude），
   支持新模型的无缝接入。
4. **可扩展性**：通过 register() 方法，第三方可以注册自定义模型和提供商，
   无需修改核心代码。
"""

from __future__ import annotations

from typing import Any

from smart_repo.models.base import BaseProvider
from smart_repo.models.claude import ClaudeProvider
from smart_repo.models.openai import OpenAIProvider
from smart_repo.models.deepseek import DeepSeekProvider


class ModelRegistry:
    """模型注册中心，将模型标识符映射到对应的提供商类（Registry that maps model identifiers to their provider classes）。

    职责：
    - 维护 model_id → ProviderClass 的映射表。
    - 根据模型 ID 创建对应的提供商实例。
    - 提供查询接口（list_models、list_by_provider、is_registered）。

    使用方式（Usage）：
        registry = ModelRegistry()
        registry.register("claude-sonnet-4-6", ClaudeProvider)
        provider = registry.create("claude-sonnet-4-6", api_key="...")

        # 也可以直接使用自动注册的内置模型
        provider = registry.create("gpt-4o", api_key="sk-...")
    """

    def __init__(self) -> None:
        """初始化注册中心，自动注册所有内置模型。

        内置模型包括：
        - Claude 系列：fable-5, opus-4-x, sonnet-4-6, haiku-4-5, claude-3.5 系列
        - OpenAI 系列：gpt-4o, gpt-4-turbo, gpt-3.5-turbo, o1/o3 系列
        """
        self._providers: dict[str, type[BaseProvider]] = {}  # model_id → ProviderClass 映射表
        self._default_model: str | None = None  # 默认模型（预留字段）

        # 自动注册内置提供商的模型映射
        self._register_builtins()

    def _register_builtins(self) -> None:
        """注册内置的模型→提供商映射（Register built-in model→provider mappings）。

        将已知的 Claude 和 OpenAI 模型自动注册到映射表中。
        模型 ID 在存储时会转为小写，确保大小写不敏感的查询。
        """
        # Claude 系列模型 → ClaudeProvider
        claude_models = [
            "claude-fable-5",
            "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
            "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ]
        for m in claude_models:
            self.register(m, ClaudeProvider)

        # OpenAI 系列模型 → OpenAIProvider
        openai_models = [
            "gpt-4o", "gpt-4o-2024-08-06", "gpt-4o-mini",
            "gpt-4-turbo", "gpt-4-0125-preview", "gpt-4",
            "gpt-3.5-turbo-0125", "gpt-3.5-turbo",
            "o1", "o1-mini", "o3-mini",
        ]
        for m in openai_models:
            self.register(m, OpenAIProvider)

        # DeepSeek 系列模型 → DeepSeekProvider（OpenAI 兼容协议，独立端点）
        deepseek_models = [
            "deepseek-chat",
            "deepseek-reasoner",
        ]
        for m in deepseek_models:
            self.register(m, DeepSeekProvider)

    def register(self, model_id: str, provider_cls: type[BaseProvider]) -> None:
        """注册一个模型 ID → 提供商类的映射。

        模型 ID 存储时会统一转为小写，保证查询时大小写不敏感。

        Args:
            model_id: 模型标识符（如 "gpt-4o"、"claude-sonnet-4-6"）。
            provider_cls: 提供商类（必须是 BaseProvider 的子类）。

        Raises:
            TypeError: 如果 provider_cls 不是 BaseProvider 的子类。
        """
        self._providers[model_id.lower()] = provider_cls

    def get_provider_class(self, model_id: str) -> type[BaseProvider]:
        """根据模型 ID 查找对应的提供商类（Look up the provider class for a model ID）。

        查找逻辑：
        1. 首先在注册表中精确匹配（大小写不敏感）。
        2. 若未找到，使用启发式规则通过前缀推断：
           - "claude" 开头 → ClaudeProvider
           - "gpt"、"o1"、"o3" 开头 → OpenAIProvider
        3. 若仍未匹配，抛出 KeyError。

        Args:
            model_id: 模型标识符。

        Returns:
            type[BaseProvider]: 对应的提供商类。

        Raises:
            KeyError: 当模型 ID 未注册且无法推断时抛出。
        """
        key = model_id.lower()
        if key not in self._providers:
            # Heuristic: model names starting with "claude" → Claude, "gpt"|"o1"|"o3" → OpenAI
            # 启发式匹配：通过模型名前缀判断提供商类型
            if key.startswith("claude"):
                return ClaudeProvider
            if any(key.startswith(p) for p in ("gpt", "o1", "o3")):
                return OpenAIProvider
            if key.startswith("deepseek"):
                return DeepSeekProvider
            raise KeyError(
                f"Unknown model '{model_id}'. Register it first with "
                f"registry.register('{model_id}', YourProvider)."
            )
        return self._providers[key]

    def create(
        self,
        model_id: str,
        api_key: str = "",
        **kwargs: Any,
    ) -> BaseProvider:
        """根据模型 ID 创建对应的提供商实例。

        这是注册中心最常用的方法。它结合了查询和实例化两步操作。

        Args:
            model_id: 模型标识符。
            api_key: API 密钥（可以为空字符串，由提供商自行处理）。
            **kwargs: 传递给提供商构造函数的额外参数。

        Returns:
            BaseProvider: 已初始化的提供商实例。

        Raises:
            KeyError: 当模型 ID 未注册且无法推断时抛出。
        """
        cls = self.get_provider_class(model_id)
        return cls(model=model_id, api_key=api_key, **kwargs)

    def list_models(self) -> list[str]:
        """列出所有已注册的模型 ID。

        Returns:
            list[str]: 按字母排序的模型 ID 列表。
        """
        return sorted(self._providers.keys())

    def list_by_provider(self) -> dict[str, list[str]]:
        """按提供商类型分组列出所有模型。

        Returns:
            dict[str, list[str]]: 格式为 {提供商类名: [模型ID列表]}，
            例如 {"ClaudeProvider": ["claude-opus-4-6", ...], "OpenAIProvider": ["gpt-4o", ...]}。
        """
        groups: dict[str, list[str]] = {}
        for model_id, cls in self._providers.items():
            name = cls.__name__
            groups.setdefault(name, []).append(model_id)
        return groups

    def is_registered(self, model_id: str) -> bool:
        """检查某个模型 ID 是否已注册。

        Args:
            model_id: 模型标识符（大小写不敏感）。

        Returns:
            bool: True 表示该模型已注册。
        """
        return model_id.lower() in self._providers
