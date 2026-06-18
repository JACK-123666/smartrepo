"""
过程笔记 (ProcessNotes) — 对话摘要、关键决策、错误日志。

============================================================
三种记忆的职责划分（回顾）
============================================================

SmartRepo 的 memory 系统包含三种记忆：
1. TaskMemory  (task_memory.py)  — 追踪"要做什么"：目标 → 子任务 → 进度
2. FileMemory  (file_memory.py)  — 缓存"文件里有什么"：路径 → 摘要/符号表
3. ProcessNotes (本文件)         — 记录"发生了什么"：决策/错误/洞察/里程碑

============================================================
设计背景与核心价值
============================================================

在 agent 执行过程中，大量"过程信息"会被产生：
  - "为什么要删除这个文件？" — 设计决策
  - "调用 API 时报了 403 错误" — 运行时错误
  - "这个模块采用了工厂模式" — 代码洞察
  - "重构第一阶段完成" — 里程碑
  - "对话的前半部分讨论了认证流程" — 对话摘要

这些信息不属于"任务进度"（那是 TaskMemory 的职责），也不属于"文件缓存"
（那是 FileMemory 的职责），但如果不记录，就会在对话历史被裁剪后永久丢失。

ProcessNotes 将这些信息按 category 分类存储，支持：
  - 事后追溯："当初为什么做了这个决定？"
  - 错误复盘："之前遇到了哪些错误？怎么解决的？"
  - 恢复上下文：从检查点恢复后，快速了解"之前发生了什么"

============================================================
笔记分类 (5 种)
============================================================
  - decision:  关键决策（如"选择 SQLite 而非 JSON 存储检查点"）
  - error:     运行时错误（如 API 调用失败、文件不存在）
  - insight:   洞察和观察（如"这个模块用了循环导入的 workaround"）
  - milestone: 里程碑事件（如"Phase 1 完成，开始 Phase 2"）
  - summary:   对话/会话摘要（定期生成，用于上下文恢复）
============================================================

Process notes — conversation summaries, key decisions, errors log.
Records "what happened" during agent execution, separate from task progress
and file caching. Survives context pruning and checkpoint restores.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from smart_repo.memory.store import MemoryStore


@dataclass
class Note:
    """一条过程笔记。

    Attributes:
        id: 笔记的唯一标识符（自动生成，格式为 "note_0001"）。
        category: 笔记类别，必须是以下之一：
                  "decision"  — 关键决策
                  "error"     — 错误记录
                  "insight"   — 洞察/发现
                  "milestone" — 里程碑
                  "summary"   — 会话摘要
        content: 笔记内容（自由文本）。
        timestamp: 创建时间戳（Unix epoch 秒）。
        related_task_id: 关联的任务 ID（可选），用于按任务过滤笔记。
        metadata: 自定义元数据字典。
    """

    id: str
    category: str  # "decision", "error", "insight", "milestone", "summary"
    content: str
    timestamp: float = field(default_factory=time.time)
    related_task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict，用于 JSON 持久化。"""
        return {
            "id": self.id,
            "category": self.category,
            "content": self.content,
            "timestamp": self.timestamp,
            "related_task_id": self.related_task_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Note:
        """从 dict 反序列化创建 Note。"""
        return cls(
            id=d["id"],
            category=d["category"],
            content=d["content"],
            timestamp=d.get("timestamp", time.time()),
            related_task_id=d.get("related_task_id"),
            metadata=d.get("metadata", {}),
        )


class ProcessNotes:
    """过程笔记 — 管理会话摘要、关键决策、错误记录。

    职责：
      - 按类别（decision/error/insight/milestone/summary）存储自由文本笔记
      - 支持按类别、时间、关联任务过滤和查询
      - 支持文本搜索
      - 自动生成全局摘要

    使用方法:
        store = MemoryStore(Path("notes.json"))
        pn = ProcessNotes(store)

        # 记录一条关键决策
        pn.add_decision("使用 gRPC 而非 REST 进行服务间通信", task_id="arch_1")

        # 记录一个错误
        pn.add_error("Redis 连接超时：ConnectionResetError", task_id="cache_3")

        # 搜索历史笔记
        results = pn.search("gRPC")

        # 查看最新笔记
        recent = pn.get_recent(10)

        # 查看整体摘要
        print(pn.summary())

    设计说明：
      与 TaskMemory 的区别在于，ProcessNotes 不管理"任务进度"，
      而是记录"过程中的关键事件"。一个 Task 可能对应多条 Note，
      一条 Note 也可以独立于任何 Task（如全局性的架构决策），
      方便事后回溯"当初为什么做了这个决定"或"之前遇到了哪些错误"。

    Process-level memory: conversation summaries, key decisions, errors.
    Separates process-level observations from task tracking and file caching.
    """

    CATEGORIES = ["decision", "error", "insight", "milestone", "summary"]

    def __init__(self, store: MemoryStore) -> None:
        """初始化过程笔记。

        Args:
            store: 底层 MemoryStore 实例，用于持久化笔记数据。
        """
        self.store = store
        self._notes: dict[str, Note] = {}  # 内存中的笔记字典，键为笔记 ID
        self._id_counter = 0  # 自动递增的 ID 计数器
        self._load()

    def _load(self) -> None:
        """从底层 store 加载笔记数据到内存。

        笔记数据在 store 中以键 "__notes__" 存储，值为 {note_id: note_dict} 的字典。
        加载时同时恢复 _id_counter，确保新增笔记的 ID 不会与已有笔记冲突。
        """
        data = self.store.get("__notes__", {})
        for nid, nd in data.items():
            self._notes[nid] = Note.from_dict(nd)
            # 从已有 ID 中恢复最大计数器值
            try:
                num = int(nid.split("_")[-1])
                self._id_counter = max(self._id_counter, num)
            except (ValueError, IndexError):
                pass

    def _save(self) -> None:
        """将内存中的笔记持久化到 store 并 flush 到磁盘。"""
        data = {nid: n.to_dict() for nid, n in self._notes.items()}
        self.store.set("__notes__", data)
        self.store.flush()

    def _next_id(self) -> str:
        """生成下一个笔记 ID（格式: note_0001, note_0002, ...）。"""
        self._id_counter += 1
        return f"note_{self._id_counter:04d}"

    def add(
        self,
        content: str,
        category: str = "insight",
        related_task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Note:
        """添加一条新笔记（通用方法）。

        如果 category 不在已知类别中，则自动归为 "insight"。

        Args:
            content: 笔记内容（自由文本）。
            category: 笔记类别，默认为 "insight"。
            related_task_id: 关联的任务 ID（可选），用于按任务过滤。
            metadata: 自定义元数据（可选）。

        Returns:
            新创建的 Note 对象。
        """
        if category not in self.CATEGORIES:
            category = "insight"

        note = Note(
            id=self._next_id(),
            category=category,
            content=content,
            related_task_id=related_task_id,
            metadata=metadata or {},
        )
        self._notes[note.id] = note
        self._save()
        return note

    def add_decision(self, content: str, task_id: str | None = None) -> Note:
        """记录一条关键决策（快捷方法，category="decision"）。

        例如："决定将所有 API 接口从 REST 迁移到 GraphQL。"
        """
        return self.add(content, "decision", task_id)

    def add_error(self, content: str, task_id: str | None = None) -> Note:
        """记录一个错误（快捷方法，category="error"）。

        例如："数据库迁移脚本执行失败：FK constraint violation。"
        """
        return self.add(content, "error", task_id)

    def add_insight(self, content: str, task_id: str | None = None) -> Note:
        """记录一个洞察或发现（快捷方法，category="insight"）。

        例如："auth 模块的 token 刷新逻辑与 session 模块有代码重复。"
        """
        return self.add(content, "insight", task_id)

    def add_milestone(self, content: str, task_id: str | None = None) -> Note:
        """记录一个里程碑（快捷方法，category="milestone"）。

        例如："v2.0 重构全部完成，所有集成测试通过。"
        """
        return self.add(content, "milestone", task_id)

    def add_summary(self, content: str) -> Note:
        """记录一条会话摘要（快捷方法，category="summary"）。

        通常在会话结束时调用，用于下次恢复时快速了解之前的上下文。
        """
        return self.add(content, "summary")

    def get(self, note_id: str) -> Note | None:
        """获取指定 ID 的笔记。

        Args:
            note_id: 笔记 ID（如 "note_0001"）。

        Returns:
            Note 对象，不存在则返回 None。
        """
        return self._notes.get(note_id)

    def get_by_category(self, category: str, limit: int = 50) -> list[Note]:
        """按类别获取笔记，按时间倒序（最新的在前）。

        Args:
            category: 笔记类别。
            limit: 返回的最大条数，默认 50。

        Returns:
            匹配的笔记列表，按时间降序排列。
        """
        notes = [n for n in self._notes.values() if n.category == category]
        notes.sort(key=lambda n: n.timestamp, reverse=True)
        return notes[:limit]

    def get_recent(self, limit: int = 20) -> list[Note]:
        """获取最近的笔记（跨所有类别），按时间倒序。

        Args:
            limit: 返回的最大条数，默认 20。

        Returns:
            最近的笔记列表。
        """
        notes = sorted(self._notes.values(), key=lambda n: n.timestamp, reverse=True)
        return notes[:limit]

    def get_for_task(self, task_id: str) -> list[Note]:
        """获取与指定任务关联的所有笔记。

        Args:
            task_id: 任务 ID。

        Returns:
            关联的笔记列表。
        """
        return [n for n in self._notes.values() if n.related_task_id == task_id]

    def search(self, query: str) -> list[Note]:
        """在笔记内容中进行简单的文本搜索（大小写不敏感）。

        Args:
            query: 搜索关键词。

        Returns:
            内容中包含关键词的笔记列表。
        """
        q = query.lower()
        return [n for n in self._notes.values() if q in n.content.lower()]

    def summary(self) -> dict[str, Any]:
        """返回所有笔记的摘要统计。

        Returns:
            包含以下键的字典：
              - total_notes: 笔记总数
              - by_category: 各类别的笔记数量（dict）
              - latest_note: 最新笔记的前 200 字符（无笔记时为 "No notes yet"）
        """
        by_cat = {}
        for cat in self.CATEGORIES:
            by_cat[cat] = len(self.get_by_category(cat))
        return {
            "total_notes": len(self._notes),
            "by_category": by_cat,
            "latest_note": (
                max(self._notes.values(), key=lambda n: n.timestamp).content[:200]
                if self._notes else "No notes yet"
            ),
        }

    def delete(self, note_id: str) -> bool:
        """删除指定 ID 的笔记。

        Args:
            note_id: 笔记 ID。

        Returns:
            True 表示笔记存在并已删除，False 表示笔记不存在。
        """
        if note_id in self._notes:
            del self._notes[note_id]
            self._save()
            return True
        return False

    def clear_old(self, before_timestamp: float) -> int:
        """清除在指定时间戳之前创建的所有旧笔记。

        用于定期清理历史笔记，防止存储文件无限增长。

        Args:
            before_timestamp: 清除此时间戳之前的笔记（Unix epoch 秒）。

        Returns:
            被清除的笔记数量。
        """
        to_delete = [
            nid for nid, n in self._notes.items()
            if n.timestamp < before_timestamp
        ]
        for nid in to_delete:
            del self._notes[nid]
        if to_delete:
            self._save()
        return len(to_delete)
