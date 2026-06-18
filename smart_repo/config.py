"""
全局配置模块 — 基于 Pydantic Settings 的统一配置管理中心。

本模块是 SmartRepo 所有配置项的单一入口，负责从环境变量和 .env 文件
加载配置，并提供类型安全的字段访问。

设计原因：
1. 使用 Pydantic Settings 而非裸 os.environ 读取，保证每个配置项都有
   明确的类型、默认值和校验规则，避免运行时因缺少配置导致的隐性错误。
2. Config 类采用单例模式（由调用方控制实例数），全项目共享一份配置，
   确保 workspace_dir、API key 等关键参数在整个会话中保持一致。
3. 通过 SettingsConfigDict 的 env_prefix="SMARTREPO_" 和 env_file=".env"，
   支持多种部署场景：开发环境用 .env 文件，生产环境用系统环境变量，
   容器化部署用 K8s ConfigMap 注入。

使用方式:
    from smart_repo.config import Config
    config = Config(workspace_dir=Path("/my/project"))
    config.resolve_api_key("claude")  # -> 返回 API key
    config.ensure_dirs()              # -> 创建所需的目录
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 把项目根的 .env 加载进 os.environ，使 resolve_api_key 能读到无前缀的 key
# （DEEPSEEK_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY）。
# pydantic-settings 的 env_file 只把带 SMARTREPO_ 前缀的变量加载到字段，不设 os.environ，
# 故这里显式 load_dotenv 补齐。
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class Config(BaseSettings):
    """
    SmartRepo 全局配置类 — 从环境变量和 .env 文件加载所有配置。

    职责：
    - 管理路径配置（工作区、检查点、记忆、日志目录）
    - 管理模型配置（默认模型、提供商、API key）
    - 管理上下文治理参数（token 预算、摘要阈值、压缩比）
    - 管理安全策略（允许目录、禁用命令、审批规则）
    - 管理检查点和文件缓存参数

    使用方式：
        config = Config(workspace_dir=Path("/path/to/project"))
        key = config.resolve_api_key("claude")
        config.ensure_dirs()

    所有字段均可通过环境变量覆盖，格式: SMARTREPO_<字段名大写>
    例如: SMARTREPO_DEFAULT_MODEL=gpt-4o
    """

    model_config = SettingsConfigDict(
        env_prefix="SMARTREPO_",          # 环境变量前缀，如 SMARTREPO_WORKSPACE_DIR
        env_file=".env",                  # 从项目根目录的 .env 文件加载
        env_file_encoding="utf-8",        # .env 文件编码
        extra="ignore",                   # 忽略未定义的环境变量，防止意外注入
    )

    # =========================================================================
    # 路径配置 — Paths
    # 影响范围: 决定智能体在哪个目录操作、检查点和日志存放在哪
    # =========================================================================
    workspace_dir: Path = Field(
        default_factory=lambda: Path.cwd(),
        description=(
            "智能体操作的工作目录（Working directory for agent operations）。"
            "默认值为当前工作目录。所有文件读写、代码搜索都在此目录下进行。"
        ),
    )
    checkpoint_dir: Path = Field(
        default_factory=lambda: Path.home() / ".smartrepo" / "checkpoints",
        description=(
            "检查点存储目录（Directory for checkpoint storage）。"
            "用于保存/恢复会话快照，默认在用户 HOME 下的 .smartrepo/checkpoints。"
            "每个检查点包含会话状态、对话历史和上下文摘要。"
        ),
    )
    memory_dir: Path = Field(
        default_factory=lambda: Path.home() / ".smartrepo" / "memory",
        description=(
            "结构化记忆存储目录（Directory for structured memory storage）。"
            "存放学习记录、用户偏好、项目知识图谱等持久化记忆数据。"
            "默认在用户 HOME 下的 .smartrepo/memory。"
        ),
    )
    log_dir: Path = Field(
        default_factory=lambda: Path.home() / ".smartrepo" / "logs",
        description=(
            "会话日志目录（Directory for session logs）。"
            "存放每次会话的详细日志，包括 token 消耗、工具调用、审批决策等。"
            "默认在用户 HOME 下的 .smartrepo/logs。"
        ),
    )

    # =========================================================================
    # 模型配置 — Model
    # 影响范围: 决定使用哪个 LLM 模型、提供商和 API key
    # =========================================================================
    default_model: str = Field(
        default="claude-sonnet-4-6",
        description=(
            "默认模型标识符（Default model identifier）。"
            "当 CLI 未指定 --model 参数时使用此值。支持的模型取决于 provider。"
        ),
    )
    default_provider: Literal["claude", "openai", "deepseek"] = Field(
        default="claude",
        description=(
            "默认 LLM 提供商（Default LLM provider）。"
            "可选值: claude（Anthropic）、openai 或 deepseek。决定使用哪个 SDK 和后端 API。"
        ),
    )
    anthropic_api_key: str = Field(
        default="",
        description=(
            "Anthropic API key。也可通过环境变量 ANTHROPIC_API_KEY 设置。"
            "若两者都未设置，resolve_api_key() 将返回空字符串，导致调用 Anthropic API 时失败。"
        ),
    )
    openai_api_key: str = Field(
        default="",
        description=(
            "OpenAI API key。也可通过环境变量 OPENAI_API_KEY 设置。"
            "若两者都未设置，resolve_api_key() 将返回空字符串，导致调用 OpenAI API 时失败。"
        ),
    )
    deepseek_api_key: str = Field(
        default="",
        description=(
            "DeepSeek API key。也可通过环境变量 DEEPSEEK_API_KEY 设置。"
            "若两者都未设置，resolve_api_key() 将返回空字符串，导致调用 DeepSeek API 时失败。"
        ),
    )

    # =========================================================================
    # 上下文治理 — Context
    # 影响范围: 控制每次 LLM 请求的 token 分配和对话历史的管理策略
    # 这些参数直接影响 API 成本（token 消耗）和回复质量
    # =========================================================================
    max_context_tokens: int = Field(
        default=128_000,
        description=(
            "上下文窗口最大 token 数（Maximum context window size in tokens）。"
            "设定总 token 预算上限，超出时将触发上下文修剪/摘要。"
            "128K 适用于 Claude 和 GPT-4 系列模型。"
        ),
    )
    system_budget_ratio: float = Field(
        default=0.20,
        description=(
            "系统提示词的 token 预算占比（Token budget ratio for system prompt）。"
            "例如 max_context_tokens=128K 时，system prompt 最多占 25.6K tokens。"
            "系统提示词包含工具定义、安全规则和行为指引。"
        ),
    )
    tools_budget_ratio: float = Field(
        default=0.15,
        description=(
            "工具定义的 token 预算占比（Token budget ratio for tool definitions）。"
            "例如 128K * 0.15 = 19.2K tokens 用于工具 schema 定义。"
            "工具数量多时可能需要调高此比例。"
        ),
    )
    history_budget_ratio: float = Field(
        default=0.50,
        description=(
            "对话历史的 token 预算占比（Token budget ratio for conversation history）。"
            "例如 128K * 0.50 = 64K tokens 用于保存最近的对话轮次。"
            "这是最大的预算项，因为多轮对话会快速累积。"
        ),
    )
    files_budget_ratio: float = Field(
        default=0.15,
        description=(
            "文件内容的 token 预算占比（Token budget ratio for file contents）。"
            "例如 128K * 0.15 = 19.2K tokens 用于注入相关文件内容到上下文。"
            "在大型代码仓库中此比例可能需要调整。"
        ),
    )
    summarization_threshold: int = Field(
        default=64_000,
        description=(
            "触发历史摘要的 token 阈值（Token count at which to trigger history summarization）。"
            "当对话历史 token 数超过此值时，自动对较早的对话进行摘要压缩。"
            "64K 是一个平衡点：既保留足够的近期上下文，又避免 token 溢出。"
        ),
    )
    target_compression_ratio: float = Field(
        default=0.35,
        description=(
            "目标压缩比（Target compression ratio for context pruning）。"
            "上下文修剪或摘要时将 token 数压缩到原来的 35% 以下。"
            "较低的值节省更多 token 但可能丢失细节；较高的值保留更多信息但节省较少。"
        ),
    )

    # =========================================================================
    # 安全配置 — Security
    # 影响范围: 决定智能体的文件访问范围、命令执行权限和人工审批策略
    # =========================================================================
    allowed_directories: list[Path] = Field(
        default_factory=lambda: [Path.cwd()],
        description=(
            "允许智能体读写的目录列表（Directories the agent is allowed to read/write）。"
            "默认仅允许当前工作目录，防止智能体访问系统敏感路径。"
            "通过 CLI --workspace 参数或环境变量 SMARTREPO_ALLOWED_DIRECTORIES 扩展。"
        ),
    )
    blocked_commands: list[str] = Field(
        default_factory=lambda: [
            "rm -rf /", "dd if=", "mkfs.", ":(){ :|:& };:",
            "chmod 777 /", "> /dev/sda", "shutdown", "reboot",
        ],
        description=(
            "始终禁止执行的 Shell 命令模式（Shell command patterns that are always blocked）。"
            "这些模式会被安全沙箱在任何情况下拦截，防止灾难性操作。"
            "包含 fork bomb、磁盘破坏、权限滥用等高风险命令。"
        ),
    )
    require_approval_for: list[str] = Field(
        default_factory=lambda: [
            "shell", "write", "delete", "git_push", "git_force_push",
        ],
        description=(
            "需要人工审批的工具名称列表（Tool names that require human approval）。"
            "shell — 执行任意 Shell 命令；write — 写入/修改文件；"
            "delete — 删除文件；git_push/git_force_push — 推送代码。"
            "这些操作具有不可逆性或安全风险，默认为需要人工确认。"
        ),
    )
    auto_approve_in_workspace: bool = Field(
        default=True,
        description=(
            "是否自动批准工作区内的安全文件操作（Auto-approve safe file operations within workspace）。"
            "开启后，allowed_directories 内的读写操作无需人工审批，提升使用流畅度。"
            "关闭后每次文件操作都需要确认，适合严格的安全审计场景。"
        ),
    )

    # =========================================================================
    # 检查点配置 — Checkpoint
    # 影响范围: 决定会话恢复的频率和存储开销
    # =========================================================================
    checkpoint_interval: int = Field(
        default=1,
        description=(
            "检查点保存间隔（Save checkpoint after every N tool calls）。"
            "1 表示每次工具调用后都保存检查点，提供最细粒度的恢复能力。"
            "增大此值可减少 I/O 开销，但会降低恢复精度。"
        ),
    )
    max_checkpoints_per_session: int = Field(
        default=50,
        description=(
            "每个会话最多保留的检查点数量（Maximum checkpoints to retain per session）。"
            "超出后旧检查点将被轮转删除，防止磁盘空间无限增长。"
        ),
    )

    # =========================================================================
    # 记忆/缓存配置 — Memory
    # 影响范围: 影响文件读取性能和内存占用
    # =========================================================================
    enable_file_cache: bool = Field(
        default=True,
        description=(
            "是否缓存文件内容到内存（Cache file contents in memory to avoid re-reading）。"
            "开启后，已读取的文件在 TTL 内不会重复从磁盘读取，显著加速代码分析。"
            "内存紧张时可关闭此选项。"
        ),
    )
    file_cache_ttl_seconds: int = Field(
        default=300,
        description=(
            "文件缓存条目的生存时间（TTL for file cache entries in seconds）。"
            "300 秒（5 分钟）后缓存条目过期，下次访问时重新从磁盘读取。"
            "较短的 TTL 确保文件内容变更后能及时更新。"
        ),
    )

    def resolve_api_key(self, provider: str) -> str:
        """
        解析指定提供商的 API key，按优先级顺序查找。

        查找顺序（优先级从高到低）：
        1. Config 对象上直接设置的字段值（anthropic_api_key / openai_api_key）
        2. 系统环境变量（ANTHROPIC_API_KEY / OPENAI_API_KEY / CLAUDE_API_KEY）

        参数:
            provider: 提供商名称，"claude" 或 "openai"

        返回:
            找到的 API key 字符串；若未找到则返回空字符串
        """
        env_map = {
            "claude": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
            "openai": ["OPENAI_API_KEY"],
            "deepseek": ["DEEPSEEK_API_KEY"],
        }
        # 优先使用 Config 字段中直接配置的 key
        if provider == "claude" and self.anthropic_api_key:
            return self.anthropic_api_key
        if provider == "openai" and self.openai_api_key:
            return self.openai_api_key
        if provider == "deepseek" and self.deepseek_api_key:
            return self.deepseek_api_key
        # 回退到系统环境变量
        for env_var in env_map.get(provider, []):
            val = os.environ.get(env_var)
            if val:
                return val
        return ""

    def ensure_dirs(self) -> None:
        """
        创建必要的目录（如果不存在）。

        确保 checkpoint_dir、memory_dir、log_dir 三个目录存在，
        若不存在则递归创建父目录。通常在程序启动时调用一次。

        注意:
            不会创建 workspace_dir —— 工作目录应在启动前已存在，
            否则可能表示用户指定了无效路径。
        """
        for d in [self.checkpoint_dir, self.memory_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)
