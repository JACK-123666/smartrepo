"""会话状态机 — 管理 Agent 会话的完整生命周期和状态转换。

Session state machine — manages agent lifecycle and state transitions.

本模块定义了 SmartRepo 中"会话"的全部数据结构。一次"会话"代表从用户提交任务
到 Agent 完成（或出错、中断）的完整过程。核心设计包括：

  - SessionState 枚举：7 种生命周期状态（IDLE → RUNNING → COMPLETED/ERROR/INTERRUPTED）
  - SessionConfig 数据类：一次会话的运行参数（模型、温度、轮次上限等）
  - Session 数据类：会话的完整状态快照（消息列表、token 统计、状态机）
    - 支持序列化（to_dict）和反序列化（from_dict），用于检查点持久化
    - 提供便捷属性（is_active、duration_seconds）和统计方法

设计原则：
  - Session 是"纯数据对象 + 简单状态机"，不包含 IO 或异步逻辑
  - 所有字段都有默认值，方便从检查点"部分恢复"
  - to_dict/from_dict 保证检查点的完整可恢复性
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from smart_repo.models.base import Message


class SessionState(enum.Enum):
    """会话生命周期状态枚举（Session lifecycle states）。

    状态流转图:
        IDLE ──start()──> RUNNING ──正常结束──> COMPLETED
          │                   │
          │                   ├──出错──> ERROR
          │                   ├──中断──> INTERRUPTED
          │                   ├──需审批──> WAITING_APPROVAL
          │                   └──等用户输入──> WAITING_USER
          │
          └── 从检查点恢复: INTERRUPTED ──resume()──> RUNNING
    """

    IDLE = "idle"                  # 已创建但未启动（Created, not started）
    RUNNING = "running"            # Agent 正在执行中（Agent is executing）
    WAITING_APPROVAL = "waiting_approval"  # 等待人工审批（Waiting for human approval）
    WAITING_USER = "waiting_user"  # 等待用户交互输入（Waiting for user input, interactive）
    COMPLETED = "completed"        # 任务成功完成（Task finished successfully）
    ERROR = "error"                # 因错误终止（Terminated with error）
    INTERRUPTED = "interrupted"    # 执行中途被打断，可恢复（Interrupted mid-execution）


@dataclass
class SessionConfig:
    """会话运行配置（Configuration for a session）。

    与全局 Config 不同：SessionConfig 只包含本次任务执行所需的参数，
    而全局 Config 包含路径、安全策略等长期配置。
    """

    task: str                                         # 用户的任务描述 / prompt
    model: str = "claude-sonnet-4-6"                  # 使用的模型标识符
    provider: str = "claude"                          # 模型提供商（"claude" 或 "openai"）
    max_turns: int = 100                              # 最大对话轮次（防止无限循环）
    temperature: float = 0.7                          # 模型温度参数（0=确定性, 1=创造性）
    max_tokens_per_response: int = 4096               # 单次响应的最大 token 数
    system_prompt: str = ""                           # 自定义系统提示（为空则用默认值）
    metadata: dict[str, Any] = field(default_factory=dict)  # 附加元数据，自由扩展


@dataclass
class Session:
    """代表一次完整的 Agent 会话（Represents a single agent session with full state）。

    会话对象持有整个对话的历史、配置和执行状态。它可以被序列化（to_dict）
    以便通过检查点（checkpoint）进行持久化和恢复。

    A session holds the entire conversation, configuration, and execution state.
    It can be serialized for checkpoint recovery.

    属性说明:
        id: 会话唯一标识（12位十六进制随机字符串）
        config: 会话配置（任务、模型、温度等）
        state: 当前生命周期状态
        messages: 完整消息历史（包含 system、user、assistant、tool 四种角色）
        turn_count: 已执行的对话轮次数（每次 assistant 回复算一轮）
        total_tokens_used: 累计消耗的 token 总数
        created_at / updated_at / completed_at: 时间戳
        error_message: 错误详情（仅 state==ERROR 时有意义）
        checkpoints: 关联的检查点 ID 列表
        metadata: 附加元数据
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    config: SessionConfig | None = None
    state: SessionState = SessionState.IDLE
    messages: list[Message] = field(default_factory=list)
    turn_count: int = 0
    total_tokens_used: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    error_message: str = ""
    checkpoints: list[str] = field(default_factory=list)  # 关联的检查点 ID 列表
    metadata: dict[str, Any] = field(default_factory=dict)

    # =========================================================================
    # 生命周期方法（Lifecycle Methods）
    # =========================================================================

    def start(self, config: SessionConfig) -> None:
        """初始化会话并注入任务（Initialize the session with a task）。

        此方法做以下事情:
          1. 绑定配置，状态从 IDLE 切换到 RUNNING
          2. 清空消息列表和时间戳
          3. 构建并添加系统提示消息（system prompt）
          4. 将用户任务作为第一条 user 消息加入对话

        参数（Args）:
            config: 本次会话的运行配置。
        """
        self.config = config
        self.state = SessionState.RUNNING
        self.messages = []
        self.turn_count = 0
        self.created_at = time.time()
        self.updated_at = time.time()

        # 构建系统提示：优先用自定义的，否则使用内置默认提示
        # Build system prompt
        sys_prompt = config.system_prompt or self._default_system_prompt()
        self.messages.append(Message.system(sys_prompt))

        # 将用户的任务作为对话的第一条消息
        # Add the user's task
        self.messages.append(Message.user(config.task))

    # =========================================================================
    # 消息记录方法（Message Recording Methods）
    # =========================================================================

    def add_assistant_message(self, content: str,
                               tool_calls: list[dict[str, Any]] | None = None,
                               usage: dict[str, int] | None = None) -> None:
        """记录一条助手（模型）回复消息（Record an assistant response）。

        参数（Args）:
            content: 模型返回的文本内容。
            tool_calls: 可选的工具调用列表（每个元素含 id、function、arguments）。
            usage: 可选的 token 用量统计（含 total_tokens 等字段）。
        """
        msg = Message.assistant(content=content, tool_calls=tool_calls)
        self.messages.append(msg)
        self.turn_count += 1                    # 每一条 assistant 消息算一轮
        if usage:
            # 兜底：部分 provider 可能不返回 total_tokens，用 input+output 求和
            total = usage.get("total_tokens")
            if total is None:
                total = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            self.total_tokens_used += total
        self.updated_at = time.time()

    def add_tool_result(self, tool_call_id: str, tool_name: str,
                        result: str) -> None:
        """记录一条工具执行结果消息（Record a tool execution result）。

        参数（Args）:
            tool_call_id: 工具调用的唯一 ID，用于与对应的 tool_call 关联。
            tool_name: 工具名称（如 "read_file"、"execute_shell"）。
            result: 工具执行的输出文本。
        """
        msg = Message.tool(content=result, tool_call_id=tool_call_id, name=tool_name)
        self.messages.append(msg)
        self.updated_at = time.time()

    def add_user_message(self, content: str) -> None:
        """添加一条用户消息——用于交互式会话（Add a user message, for interactive sessions）。

        参数（Args）:
            content: 用户的输入文本。
        """
        self.messages.append(Message.user(content))
        self.updated_at = time.time()

    # =========================================================================
    # 状态管理方法（State Management Methods）
    # =========================================================================

    def set_state(self, state: SessionState) -> None:
        """切换到新状态（Transition to a new state）。

        副作用（Side effect）:
            - 更新时间戳
            - 如果切换到 COMPLETED，记录完成时间
        """
        self.state = state
        self.updated_at = time.time()
        if state == SessionState.COMPLETED:
            self.completed_at = time.time()

    def set_error(self, error: str) -> None:
        """设置错误状态并记录错误信息（Set error state with message）。

        参数（Args）:
            error: 描述错误原因的字符串。
        """
        self.state = SessionState.ERROR
        self.error_message = error
        self.updated_at = time.time()

    # =========================================================================
    # 序列化方法（Serialization Methods）
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """将会话序列化为字典——用于检查点存储（Serialize session to a dictionary for checkpoint storage）。

        返回（Returns）:
            包含所有会话字段的字典，可直接 JSON 序列化。
            消息列表中的每条 Message 也会被递归序列化。
        """
        return {
            "id": self.id,
            "config": {
                "task": self.config.task,
                "model": self.config.model,
                "provider": self.config.provider,
                "max_turns": self.config.max_turns,
                "temperature": self.config.temperature,
                "max_tokens_per_response": self.config.max_tokens_per_response,
                "system_prompt": self.config.system_prompt,
                "metadata": self.config.metadata,
            } if self.config else None,
            "state": self.state.value,
            "messages": [m.to_dict() for m in self.messages],
            "turn_count": self.turn_count,
            "total_tokens_used": self.total_tokens_used,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "error_message": self.error_message,
            "checkpoints": self.checkpoints,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Session:
        """从检查点字典反序列化恢复会话（Deserialize a session from a checkpoint）。

        此方法是一个"安全恢复"过程：对于字典中缺失的字段，全部使用默认值填充，
        确保即使旧版检查点数据也能被正常恢复。

        参数（Args）:
            d: to_dict() 产出的字典或等效结构。

        返回（Returns）:
            恢复后的 Session 对象。
        """
        # 第1步：构建 Session 壳体（使用默认值兜底）
        session = cls(
            id=d["id"],
            state=SessionState(d.get("state", "idle")),
            turn_count=d.get("turn_count", 0),
            total_tokens_used=d.get("total_tokens_used", 0),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
            completed_at=d.get("completed_at"),
            error_message=d.get("error_message", ""),
            checkpoints=d.get("checkpoints", []),
            metadata=d.get("metadata", {}),
        )

        # 第2步：恢复配置信息
        if d.get("config"):
            session.config = SessionConfig(
                task=d["config"]["task"],
                model=d["config"].get("model", "claude-sonnet-4-6"),
                provider=d["config"].get("provider", "claude"),
                max_turns=d["config"].get("max_turns", 100),
                temperature=d["config"].get("temperature", 0.7),
                max_tokens_per_response=d["config"].get("max_tokens_per_response", 4096),
                system_prompt=d["config"].get("system_prompt", ""),
                metadata=d["config"].get("metadata", {}),
            )

        # 第3步：逐条恢复消息历史
        for m in d.get("messages", []):
            session.messages.append(Message(
                role=m["role"],
                content=m["content"],
                tool_calls=m.get("tool_calls"),
                tool_call_id=m.get("tool_call_id"),
                name=m.get("name"),
            ))

        return session

    # =========================================================================
    # 便捷查询方法（Convenience Query Methods）
    # =========================================================================

    def get_last_message(self) -> Message | None:
        """获取最后一条消息，如果消息列表为空则返回 None。"""
        return self.messages[-1] if self.messages else None

    def message_count_by_role(self) -> dict[str, int]:
        """按角色统计消息数量（Count messages by role）。

        返回（Returns）:
            字典，key 为角色名（"system"/"user"/"assistant"/"tool"），value 为计数。
        """
        counts: dict[str, int] = {}
        for m in self.messages:
            counts[m.role] = counts.get(m.role, 0) + 1
        return counts

    # =========================================================================
    # 便捷属性（Convenience Properties）
    # =========================================================================

    @property
    def is_active(self) -> bool:
        """会话是否处于活跃状态（即可继续执行）。

        活跃状态包括：RUNNING（执行中）和 WAITING_APPROVAL（等待审批）。
        处于 COMPLETED、ERROR 或等待用户输入时视为非活跃。
        """
        return self.state in (SessionState.RUNNING, SessionState.WAITING_APPROVAL)

    @property
    def duration_seconds(self) -> float:
        """会话已持续的秒数（从创建到当前或完成）。"""
        end = self.completed_at or time.time()
        return end - self.created_at

    # =========================================================================
    # 内部工具方法（Internal Helper）
    # =========================================================================

    @staticmethod
    def _default_system_prompt() -> str:
        """内置默认系统提示：定义 SmartRepo 的身份、能力与行为准则。"""
        return (
            "你是 SmartRepo——一个面向代码仓库的本地 AI 编程助手。\n"
            "能力：读取 / 写入 / 编辑 / 搜索文件、执行 shell 命令、操作 git。\n"
            "身份：你是 SmartRepo，被问及你是谁时，回答你是 SmartRepo。\n"
            "行为：做修改前先说明思路；完成任务后说明你做了什么以及为什么。"
        )
