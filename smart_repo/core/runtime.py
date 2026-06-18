"""SmartRepo 顶层执行框架 — 将所有组件组装为可用的 Agent 应用。

SmartRepo — top-level execution runtime.

本模块是 SmartRepo 的程序化入口。它的职责是"组装"（wiring）:
  第1步：加载全局配置（Config）
  第2步：初始化模型注册表和提供商（ModelRegistry + Provider）
  第3步：初始化工具注册表，注册三类标准工具（文件操作、Shell 命令、Git 操作）
  第4步：初始化记忆子系统（MemoryStore → TaskMemory / FileMemory / ProcessNotes）
  第5步：初始化检查点管理器（CheckpointManager），使用 SQLite 持久化
  第6步：初始化安全审批管理器（ApprovalManager）
  第7步：将所有组件注入 Agent，创建核心循环实例

此外，它还提供:
  - run()：同步阻塞式执行一个任务
  - resume()：从检查点恢复中断的会话
  - list_sessions() / list_checkpoints()：查询历史会话
  - stop()：优雅中断
  - get_stats()：获取运行统计信息

设计原则:
  - SmartRepo 是"胶水层"，本身不做复杂逻辑，只负责创建和连接各个子系统
  - 采用依赖注入模式：所有组件在 __init__ 中创建完毕，Agent 只管使用
  - 支持信号处理（SIGINT/SIGTERM）实现优雅中断并自动保存检查点

This is the main entry point for programmatic usage.
It wires together all components: config, models, tools, memory,
security, checkpoints, and the agent loop.
"""

from __future__ import annotations

import asyncio
import signal
from pathlib import Path
from typing import Any

from smart_repo.config import Config
from smart_repo.core.agent import Agent
from smart_repo.core.checkpoint import CheckpointManager
from smart_repo.core.session import Session, SessionConfig, SessionState
from smart_repo.memory.file_memory import FileMemory
from smart_repo.memory.process_notes import ProcessNotes
from smart_repo.memory.store import MemoryStore
from smart_repo.memory.task_memory import TaskMemory
from smart_repo.models.registry import ModelRegistry
from smart_repo.security.approval import ApprovalManager
from smart_repo.tools.file_tools import register_file_tools
from smart_repo.tools.git_tools import register_git_tools
from smart_repo.tools.registry import ToolRegistry
from smart_repo.tools.shell import register_shell_tools


class SmartRepo:
    """顶层编排器——协调 SmartRepo 所有组件协同工作（Top-level runtime that orchestrates all SmartRepo components）。

    这是用户直接使用的入口类，封装了从配置加载到 Agent 运行的完整初始化流程。

    用法（Usage）:
        # 基本用法：分析代码仓库
        sr = SmartRepo(workspace_dir="/path/to/repo")
        session = await sr.run("Analyze this codebase and find potential bugs.")
        print(session.messages[-1].content)

        # 从检查点恢复中断的会话
        session = await sr.resume(session_id="abc123")

        # 查看统计信息
        stats = sr.get_stats()
        print(stats["tools_registered"], "tools available")
    """

    def __init__(
        self,
        workspace_dir: str | Path = ".",
        config: Config | None = None,
        model: str = "",
        provider: str = "",
    ) -> None:
        """初始化 SmartRepo 框架——创建并连接所有子系统。

        Initialize the SmartRepo runtime.

        初始化步骤（Initialization steps）:
          第1步：确定工作目录和全局配置
          第2步：初始化模型注册表和提供商
          第3步：初始化工具注册表，注册文件、Shell、Git 三类工具
          第4步：初始化记忆子系统（MemoryStore　→ TaskMemory / FileMemory / ProcessNotes）
          第5步：初始化检查点管理器（SQLite）
          第6步：初始化安全审批管理器
          第7步：将上述所有组件注入 Agent

        参数（Args）:
            workspace_dir: Agent 操作的工作根目录（所有文件路径的相对基准）。
            config: 可选的全局配置覆盖（不传则使用默认配置）。
            model: 模型标识符覆盖（如 "claude-sonnet-4-6"）。
            provider: 提供商覆盖（"claude" 或 "openai"）。
        """
        # --- 第1步：工作目录和全局配置 ---
        self.workspace_dir = Path(workspace_dir).resolve()
        self.config = config or Config(workspace_dir=self.workspace_dir)
        self.model = model or self.config.default_model
        self.provider_name = provider or self.config.default_provider

        # 确保配置中指定的所有目录都存在（Ensure directories exist）
        self.config.ensure_dirs()

        # --- 第2步：模型注册表和提供商 ---
        # Model registry & provider
        self.model_registry = ModelRegistry()
        api_key = self.config.resolve_api_key(self.provider_name)
        self.provider = self.model_registry.create(
            self.model, api_key=api_key,
        )

        # --- 第3步：工具注册表（注册三大类标准工具）---
        # Tool registry (populated with standard tools)
        self.tool_registry = ToolRegistry()
        # 3a. 文件操作工具：read_file, write_file, edit_file, glob, grep 等
        register_file_tools(self.tool_registry, self.workspace_dir)
        # 3b. Shell 命令工具：execute_shell，受 allowed_dirs 和 blocked_patterns 限制
        register_shell_tools(
            self.tool_registry,
            self.workspace_dir,
            allowed_dirs=self.config.allowed_directories,
            blocked_patterns=self.config.blocked_commands,
        )
        # 3c. Git 操作工具：git_status, git_diff, git_commit 等
        register_git_tools(self.tool_registry, self.workspace_dir)

        # --- 第4步：记忆子系统（Memory stores）---
        # 4a. 通用记忆存储（持久化 JSON 文件）
        self.memory_store = MemoryStore(
            self.config.memory_dir / "memory.json",
        )
        # 4b. 任务记忆：记录用户的任务偏好和历史
        self.task_memory = TaskMemory(
            MemoryStore(self.config.memory_dir / "tasks.json"),
        )
        # 4c. 文件记忆：缓存已读取文件内容，减少重复 I/O
        self.file_memory = FileMemory(
            MemoryStore(self.config.memory_dir / "file_cache.json"),
            ttl_seconds=self.config.file_cache_ttl_seconds,
            workspace=self.workspace_dir,
        )
        # 4d. 过程笔记：记录里程碑、洞察、错误和决策
        self.process_notes = ProcessNotes(
            MemoryStore(self.config.memory_dir / "notes.json"),
        )

        # --- 第5步：检查点管理器（SQLite 持久化）---
        # Checkpoint manager
        self.checkpoint_mgr = CheckpointManager(
            self.config.checkpoint_dir / "checkpoints.db",
            max_checkpoints=self.config.max_checkpoints_per_session,
        )

        # --- 第6步：安全审批管理器 ---
        # Approval manager
        self.approval = ApprovalManager(
            auto_approve_low=True,
            auto_approve_in_workspace=self.config.auto_approve_in_workspace,
        )

        # --- 第7步：创建 Agent 核心循环（将全部子系统注入）---
        # The agent
        self.agent = Agent(
            config=self.config,
            provider=self.provider,
            tool_registry=self.tool_registry,
            checkpoint_manager=self.checkpoint_mgr,
            task_memory=self.task_memory,
            file_memory=self.file_memory,
            process_notes=self.process_notes,
            approval_manager=self.approval,
        )

        # 当前活跃会话引用（Active session）
        self._current_session: Session | None = None
        self._interrupted = False

    # =========================================================================
    # 主入口：运行和恢复（Main Entry Points: run & resume）
    # =========================================================================

    async def run(
        self,
        task: str,
        system_prompt: str = "",
        max_turns: int = 100,
        temperature: float = 0.7,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        """运行 Agent 去执行一个任务（Run the agent on a task）。

        这是主入口方法。它会：
          1. 创建 SessionConfig 和 Session 对象
          2. 启动会话（添加系统提示和用户任务消息）
          3. 保存初始检查点
          4. 注册信号处理器（用于 Ctrl+C 优雅中断）
          5. 运行 Agent 主循环
          6. 根据结果保存最终检查点

        参数（Args）:
            task: 任务描述 / 用户 prompt。
            system_prompt: 可选的自定义系统提示。为空则使用内置默认值。
            max_turns: 最大对话轮次数（防止无限循环）。
            temperature: 模型温度（0.0=确定性输出，1.0=最大创造性）。
            metadata: 可选的附加元数据字典。

        返回（Returns）:
            已完成的 Session 对象，包含全部消息和执行结果。
        """
        # 第1步：构建会话配置
        session_config = SessionConfig(
            task=task,
            model=self.model,
            provider=self.provider_name,
            max_turns=max_turns,
            temperature=temperature,
            system_prompt=system_prompt,
            metadata=metadata or {},
        )

        # 第2步：创建并启动会话
        session = Session()
        session.start(session_config)

        self._current_session = session

        # 第3步：保存初始检查点（Save initial checkpoint）
        self.checkpoint_mgr.save(session, summary="Initial checkpoint")

        # 第4步：注册信号处理器，以便用户 Ctrl+C 时可以优雅中断
        # Set up signal handler for graceful interrupt
        self._setup_signal_handler()

        # 第5步：运行 Agent 主循环（Run the agent）
        session = await self.agent.run(session)

        # 第6步：保存最终检查点——区分完成和出错两种情况
        # Save final checkpoint
        if session.state == SessionState.COMPLETED:
            self.checkpoint_mgr.save(session, summary="Final checkpoint — completed")
        elif session.state == SessionState.ERROR:
            self.checkpoint_mgr.save(session, summary="Final checkpoint — error")

        self._current_session = session
        return session

    async def resume(
        self,
        session_id: str,
        checkpoint_id: str | None = None,
    ) -> Session | None:
        """从检查点恢复一个中断的会话（Resume a session from a checkpoint）。

        参数（Args）:
            session_id: 要恢复的会话 ID。
            checkpoint_id: 指定要恢复的具体检查点。None 表示恢复最新一个。

        返回（Returns）:
            恢复并执行完成的 Session 对象；如果找不到则返回 None。
        """
        # 第1步：从检查点恢复会话
        session = self.checkpoint_mgr.restore(session_id, checkpoint_id)
        if session is None:
            return None

        # 第2步：将状态从 INTERRUPTED 切换到 RUNNING，标记为当前会话
        # Re-create agent with current state
        session.set_state(SessionState.RUNNING)
        self._current_session = session

        # 第3步：注册信号处理器
        self._setup_signal_handler()

        # 第4步：续跑 Agent 主循环
        session = await self.agent.run(session)

        # 第5步：如果正常完成，保存最终检查点
        if session.state == SessionState.COMPLETED:
            self.checkpoint_mgr.save(session, summary="Resumed session completed")

        self._current_session = session
        return session

    # =========================================================================
    # 查询接口（Query Interfaces）
    # =========================================================================

    def list_sessions(self) -> list[str]:
        """列出所有有检查点记录的会话 ID（List all session IDs that have checkpoints）。

        返回（Returns）:
            去重后的会话 ID 列表。
        """
        # 扫描检查点数据库，查找所有不重复的 session_id
        # Scan checkpoint DB for unique session IDs
        import sqlite3
        conn = None
        try:
            conn = sqlite3.connect(str(self.config.checkpoint_dir / "checkpoints.db"))
            rows = conn.execute(
                "SELECT DISTINCT session_id FROM checkpoints ORDER BY session_id"
            ).fetchall()
            return [r[0] for r in rows]
        except (sqlite3.OperationalError, sqlite3.DatabaseError, FileNotFoundError):
            return []
        finally:
            if conn:
                conn.close()

    def list_checkpoints(self, session_id: str) -> list[dict[str, Any]]:
        """列出某个会话的所有检查点（List checkpoints for a session）。

        参数（Args）:
            session_id: 要查询的会话 ID。

        返回（Returns）:
            检查点摘要列表（不含完整 data）。
        """
        return self.checkpoint_mgr.list_checkpoints(session_id)

    # =========================================================================
    # 控制接口（Control Interfaces）
    # =========================================================================

    def stop(self) -> None:
        """请求当前 Agent 运行优雅停止（Request the current agent run to stop gracefully）。

        通过设置中断标志位，Agent 会在下一个循环迭代时检查并退出。
        """
        self._interrupted = True
        self.agent.stop()

    def get_stats(self) -> dict[str, Any]:
        """获取当前框架的运行统计信息（Get statistics about the current runtime state）。

        返回（Returns）:
            字典，包含:
              - workspace: 工作目录路径
              - model / provider: 使用的模型和提供商
              - tools_registered: 已注册的工具数量
              - tool_names: 已注册的工具名称列表
              - file_cache: 文件缓存统计
              - approval_stats: 审批统计
              - session: 当前会话的基本信息（id、state、turns、tokens）
        """
        return {
            "workspace": str(self.workspace_dir),
            "model": self.model,
            "provider": self.provider_name,
            "tools_registered": len(self.tool_registry),
            "tool_names": self.tool_registry.list_names(),
            "file_cache": self.file_memory.cache_stats(),
            "approval_stats": self.approval.stats(),
            "session": {
                "id": self._current_session.id if self._current_session else None,
                "state": self._current_session.state.value if self._current_session else "none",
                "turns": self._current_session.turn_count if self._current_session else 0,
                "tokens": self._current_session.total_tokens_used if self._current_session else 0,
            },
        }

    # =========================================================================
    # 资源管理（Resource Management）
    # =========================================================================

    def close(self) -> None:
        """释放底层资源——关闭 SQLite 检查点连接（Release resources）。

        CLI 短命令在进程退出时会由 OS 回收，但长期运行的 Python API 使用方
        应显式调用 close() 或使用 ``with SmartRepo(...) as sr:`` 上下文管理，
        确保 SQLite 连接及时关闭、避免句柄堆积。
        """
        if self.checkpoint_mgr is not None:
            self.checkpoint_mgr.close()

    def __enter__(self) -> "SmartRepo":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # =========================================================================
    # 信号处理（Signal Handling）
    # =========================================================================

    def _setup_signal_handler(self) -> None:
        """注册 SIGINT（Ctrl+C）和 SIGTERM 的处理器，用于优雅关闭。

        Set up SIGINT/SIGTERM handler for graceful shutdown.

        注意：Windows 上 add_signal_handler 支持有限，因此捕获
        NotImplementedError 和 RuntimeError 以兼容跨平台运行。
        """
        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, self._handle_interrupt)
                except (NotImplementedError, RuntimeError):
                    # Windows 对 add_signal_handler 支持较差（Windows doesn't support add_signal_handler well）
                    pass
        except RuntimeError:
            # 可能没有运行中的事件循环
            pass

    def _handle_interrupt(self) -> None:
        """处理中断信号——紧急保存检查点后停止 Agent。

        Handle interrupt signal — save checkpoint and stop.

        这样即使用户按 Ctrl+C，当前的进度也不会丢失，后续可以通过 resume() 恢复。
        """
        self._interrupted = True
        self.agent.stop()
        if self._current_session:
            self.checkpoint_mgr.save(
                self._current_session,
                summary="Emergency checkpoint — interrupted",
            )
