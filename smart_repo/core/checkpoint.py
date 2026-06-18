"""检查点管理器 — 增量保存/恢复会话状态，支持断点续跑。

Checkpoint manager — incremental save/restore for session recovery.

本模块使用 SQLite 实现高效的检查点存储，具备以下能力：
  - 完整会话状态序列化（full session state serialization）：将会话的所有消息、配置、统计信息打包
  - 增量保存（incremental saves）：每次只新增一行记录，不覆盖历史
  - 时间点恢复（point-in-time recovery）：可以恢复到任意历史检查点
  - 自动清理（automatic cleanup）：每个会话最多保留 N 个检查点，超出则删除最旧的

设计原则：
  - 使用 SQLite 而非 JSON 文件：支持并发读写、索引查询、事务安全
  - 检查点是不可变记录：save() 总是插入新行，更新操作是追加式的
  - sequence 字段按会话独立递增，用于排序和"恢复最新"
  - 恢复后的会话状态设为 INTERRUPTED，提醒调用方这是从快照恢复的

Uses SQLite for efficient checkpoint storage with:
  - Full session state serialization
  - Incremental saves (only changed messages)
  - Point-in-time recovery
  - Automatic cleanup of old checkpoints
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from smart_repo.core.session import Session, SessionState


class CheckpointManager:
    """管理 Agent 会话的检查点保存和恢复（Manages checkpoint save/restore for agent sessions）。

    每个检查点捕获完整的会话状态（Each checkpoint captures the full session state），包括:
      - 对话中的所有消息（All messages in the conversation）
      - 会话配置（Session configuration）
      - 轮次计数和 token 用量（Turn count and token usage）
      - 记忆状态引用（Memory state references）

    数据库表结构（SQLite schema）:
        checkpoints (
            id TEXT PRIMARY KEY,       -- 检查点唯一 ID，格式 "ckpt_xxxxxxxx"
            session_id TEXT NOT NULL,  -- 所属会话 ID
            sequence INTEGER NOT NULL, -- 在该会话中的序号（递增，从 0 开始）
            timestamp REAL NOT NULL,   -- 创建时间戳
            state TEXT NOT NULL,       -- 保存时的会话状态
            data TEXT NOT NULL,        -- 完整会话 JSON
            summary TEXT DEFAULT ''    -- 人类可读的摘要
        )
    """

    def __init__(self, db_path: Path, max_checkpoints: int = 50) -> None:
        """初始化检查点管理器。

        参数（Args）:
            db_path: SQLite 数据库文件路径（目录会自动创建）。
            max_checkpoints: 每个会话最多保留的检查点数量（超出则删除最旧的）。
        """
        self.db_path = db_path
        self.max_checkpoints = max_checkpoints
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        """创建检查点数据库——如果不存在则自动建表。

        Create the checkpoint database if it doesn't exist.

        创建的表:
            - checkpoints: 存储所有检查点记录
            - idx_checkpoints_session: 按 (session_id, sequence) 的复合索引，加速查询
        """
        # 确保父目录存在
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._get_conn()
        # 建表：如果表已存在则跳过（IF NOT EXISTS）
        conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                state TEXT NOT NULL,
                data TEXT NOT NULL,
                summary TEXT DEFAULT ''
            )
        """)
        # 建索引：加速"获取某会话所有检查点"和"获取最新检查点"查询
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_checkpoints_session
            ON checkpoints(session_id, sequence)
        """)
        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接（懒初始化，首次调用时建立连接）。

        设置 row_factory = sqlite3.Row 使查询结果可以通过列名访问（如 row["data"]）。
        """
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    # =========================================================================
    # 核心操作：保存、恢复、列表、删除
    # =========================================================================

    def save(self, session: Session, summary: str = "") -> str:
        """保存当前会话状态的检查点（Save a checkpoint of the current session state）。

        执行步骤:
          第1步：生成检查点 ID（格式 "ckpt_xxxxxxxx"）
          第2步：计算该会话的下一个序列号（sequence = 上次最大值 + 1）
          第3步：将会话序列化为 JSON 字符串
          第4步：插入数据库
          第5步：将检查点 ID 追加到 session.checkpoints 列表
          第6步：如果该会话检查点数超过 max_checkpoints，清理最旧的

        参数（Args）:
            session: 要创建检查点的会话对象。
            summary: 人类可读的摘要，说明自上次检查点以来发生了什么。

        返回（Returns）:
            新创建的检查点 ID 字符串。
        """
        # 第1步：生成唯一检查点 ID
        checkpoint_id = f"ckpt_{uuid.uuid4().hex[:8]}"
        conn = self._get_conn()

        # 第2步：获取下一个序列号（Get next sequence number for this session）
        row = conn.execute(
            "SELECT COALESCE(MAX(sequence), -1) + 1 as next_seq FROM checkpoints WHERE session_id = ?",
            (session.id,),
        ).fetchone()
        sequence = row["next_seq"]

        # 第3步：序列化会话为 JSON（Serialize session）
        data = json.dumps(session.to_dict(), ensure_ascii=False)

        # 第4步：插入数据库
        conn.execute(
            """INSERT INTO checkpoints (id, session_id, sequence, timestamp, state, data, summary)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                checkpoint_id,
                session.id,
                sequence,
                time.time(),
                session.state.value,
                data,
                summary,
            ),
        )
        conn.commit()

        # 第5步：在会话对象中追踪此检查点（Track checkpoint in session）
        session.checkpoints.append(checkpoint_id)

        # 第6步：清理超出上限的旧检查点（Cleanup old checkpoints）
        self._cleanup(session.id)

        return checkpoint_id

    def restore(self, session_id: str,
                checkpoint_id: str | None = None) -> Session | None:
        """从检查点恢复会话（Restore a session from a checkpoint）。

        参数（Args）:
            session_id: 要恢复的会话 ID。
            checkpoint_id: 指定要恢复的具体检查点 ID。如果为 None，则恢复最新的一个。

        返回（Returns）:
            恢复后的 Session 对象；如果未找到则返回 None。
            注意：恢复后的会话状态会被设为 INTERRUPTED，表示这是中断后恢复。
        """
        conn = self._get_conn()

        # 按指定检查点 ID 查询，或取最新的（按 sequence DESC）
        if checkpoint_id:
            row = conn.execute(
                "SELECT * FROM checkpoints WHERE id = ? AND session_id = ?",
                (checkpoint_id, session_id),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM checkpoints WHERE session_id = ? ORDER BY sequence DESC LIMIT 1",
                (session_id,),
            ).fetchone()

        if row is None:
            return None

        # 从 JSON 反序列化会话
        data = json.loads(row["data"])
        session = Session.from_dict(data)
        # 标记为 INTERRUPTED —— 因为我们是从快照恢复的，需要让 Agent 知道这是续跑
        session.state = SessionState.INTERRUPTED  # Mark as interrupted since we're resuming
        return session

    def list_checkpoints(self, session_id: str) -> list[dict[str, Any]]:
        """列出某个会话的所有检查点（List all checkpoints for a session）。

        参数（Args）:
            session_id: 要查询的会话 ID。

        返回（Returns）:
            检查点摘要列表，每个元素包含 id、sequence、timestamp、state、summary。
            按 sequence 升序排列（最旧 → 最新）。
            注意：不包含完整的 data 字段（数据量太大）。
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, sequence, timestamp, state, summary FROM checkpoints "
            "WHERE session_id = ? ORDER BY sequence",
            (session_id,),
        ).fetchall()

        return [{
            "id": r["id"],
            "sequence": r["sequence"],
            "timestamp": r["timestamp"],
            "state": r["state"],
            "summary": r["summary"],
        } for r in rows]

    def get_latest_sequence(self, session_id: str) -> int:
        """获取某会话的最新检查点序列号（Get the latest checkpoint sequence number for a session）。

        返回（Returns）:
            最新的 sequence 值；如果该会话还没有任何检查点，返回 -1。
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT MAX(sequence) as seq FROM checkpoints WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row["seq"] if row["seq"] is not None else -1

    def delete_session(self, session_id: str) -> int:
        """删除某个会话的全部检查点（Delete all checkpoints for a session）。

        参数（Args）:
            session_id: 要清理的会话 ID。

        返回（Returns）:
            被删除的检查点记录数量。
        """
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM checkpoints WHERE session_id = ?",
            (session_id,),
        )
        conn.commit()
        return cursor.rowcount

    # =========================================================================
    # 内部维护方法（Internal Maintenance）
    # =========================================================================

    def _cleanup(self, session_id: str) -> None:
        """清理最旧的检查点——当某个会话的检查点数超过 max_checkpoints 时触发。

        策略：删除 sequence 最小的 N 条记录，保留最新的 max_checkpoints 条。
        Remove oldest checkpoints if exceeding max per session.
        """
        conn = self._get_conn()
        # 统计当前会话有多少个检查点
        count = conn.execute(
            "SELECT COUNT(*) as cnt FROM checkpoints WHERE session_id = ?",
            (session_id,),
        ).fetchone()["cnt"]

        if count > self.max_checkpoints:
            # 计算需要删除的多余数量
            excess = count - self.max_checkpoints
            # 删除 sequence 最小的（即最旧的）excess 条记录
            conn.execute(
                """DELETE FROM checkpoints WHERE id IN (
                    SELECT id FROM checkpoints WHERE session_id = ?
                    ORDER BY sequence ASC LIMIT ?
                )""",
                (session_id, excess),
            )
            conn.commit()

    def close(self) -> None:
        """关闭数据库连接（Close the database connection）。

        在程序退出前应调用此方法，确保 SQLite 文件被正确关闭。
        """
        if self._conn:
            self._conn.close()
            self._conn = None
