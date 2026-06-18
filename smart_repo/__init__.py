"""
SmartRepo — 面向代码仓库的本地多模型智能体工具。

SmartRepo 是一个可本地运行的多模型智能体框架，专为代码仓库场景设计。
提供统一的 API 来接入 Claude、OpenAI 等多种 LLM，内置安全检查点恢复、
分层上下文治理、安全沙箱、结构化记忆等核心能力。

核心特性:
- 检查点恢复（Checkpoint Recovery）: 每步自动保存，支持从中断处恢复
- 分层上下文治理（Layered Context Governance）: 精细的 token 预算分配
- 安全沙箱（Security Sandbox）: 文件访问控制和命令拦截
- 结构化记忆（Structured Memory）: 文件缓存和知识图谱
- 统一多模型访问（Unified Multi-Model Access）: 一套 API 切换不同 LLM

英文: A local multi-model intelligent agent harness for code repositories.
Features: checkpoint recovery, layered context governance, security sandbox,
structured memory, and unified multi-model access.

使用方式:
    from smart_repo import SmartRepo, Config

    config = Config(workspace_dir=Path("/my/project"))
    sr = SmartRepo(workspace_dir=Path("/my/project"), config=config)
    session = await sr.run(task="查找所有 TODO 注释")
"""

__version__ = "1.0.0"

# 公开 API 清单 — 这些名称可通过 from smart_repo import X 访问
# 实际导入通过下方的 __getattr__ 实现延迟加载（lazy import）
__all__ = [
    "SmartRepo",         # 智能体主类
    "Session",            # 会话管理
    "Config",             # 全局配置
    "ToolRegistry",       # 工具注册表
    "ContextGovernor",    # 上下文治理器
    "SecuritySandbox",    # 安全沙箱
]


def __getattr__(name: str):
    """
    模块级延迟导入（Lazy Import）实现。

    当用户执行 `from smart_repo import SmartRepo` 时，
    Python 在模块属性中找不到 'SmartRepo'，会调用此函数。
    我们在此按需导入对应子模块，避免启动时加载所有依赖。

    设计原因:
        SmartRepo 的子模块（core、tools、security、context 等）
        依赖较重。使用延迟导入可确保:
        1. CLI 启动速度更快（不用加载未使用的模块）
        2. 仅导入必要的子模块，减少内存占用
        3. 避免循环导入问题

    参数:
        name: 要导入的属性名称（必须在 __all__ 中定义）

    返回:
        对应的类对象

    异常:
        AttributeError: 请求的名称不在 __all__ 清单中
    """
    if name == "SmartRepo":
        from smart_repo.core.runtime import SmartRepo
        return SmartRepo
    if name == "Session":
        from smart_repo.core.session import Session
        return Session
    if name == "Config":
        from smart_repo.config import Config
        return Config
    if name == "ToolRegistry":
        from smart_repo.tools.registry import ToolRegistry
        return ToolRegistry
    if name == "ContextGovernor":
        from smart_repo.context.governor import ContextGovernor
        return ContextGovernor
    if name == "SecuritySandbox":
        from smart_repo.security.sandbox import SecuritySandbox
        return SecuritySandbox
    raise AttributeError(f"模块 {__name__!r} 没有属性 {name!r}")
