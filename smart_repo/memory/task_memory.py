"""
任务记忆 (TaskMemory) — 追踪目标、子任务、进度和决策。

============================================================
三种记忆的职责划分
============================================================

SmartRepo 的 memory 系统包含三种记忆，各司其职：

1. TaskMemory (任务记忆) — 本文件
   职责：追踪"要做什么"
   内容：目标 → 子任务 → 进度状态
   场景：agent 在执行复杂多步任务（如"重构整个模块"）时，需要
         记住当前做到哪一步了、还有哪些子任务未完成。
         即使对话中断后恢复，也能从上次的进度继续。
   数据结构：Task 对象，通过 parent_id 形成树形层次结构。

2. FileMemory (文件记忆) — file_memory.py
   职责：缓存"文件里有什么"
   内容：文件路径 → 缓存摘要/符号表/修改时间
   场景：agent 频繁读取同一批文件时，避免重复磁盘 I/O 和 LLM 调用。
         详见 file_memory.py 模块注释。

3. ProcessNotes (过程笔记) — notes.py
   职责：记录"发生了什么"
   内容：关键决策、错误日志、洞察、里程碑、会话摘要
   场景：debug 时回溯"为什么当初做了这个决定"，
         或恢复对话时了解"之前遇到了什么错误"。
   数据类型：Note 对象，按 category 分类（decision/error/insight/milestone/summary）。

============================================================
TaskMemory 设计要点
============================================================

- 层次结构：Task 通过 parent_id 形成树形结构。
  例如：根任务 "重构 user 模块" 下有三个子任务 "拆分 service 层"、
  "迁移测试"、"更新文档"。

- 状态机：Task 有 5 种状态：
    pending      — 待处理
    in_progress  — 进行中
    completed    — 已完成
    blocked      — 被阻塞（依赖其他任务或等待外部条件）
    cancelled    — 已取消

- 持久化：通过 MemoryStore 将任务树序列化为 JSON 持久化，
  进程重启后可从磁盘恢复所有任务和进度。

- 进度统计：progress_summary() 提供整体进度视图，
  便于向用户汇报或让 agent 自我评估完成度。
============================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from smart_repo.memory.store import MemoryStore


@dataclass
class Task:
    """单个任务或子任务。

    Attributes:
        id: 任务的唯一标识符（建议使用有意义的字符串 ID）。
        description: 任务描述。
        status: 任务状态，可选值：
                "pending"      — 待处理
                "in_progress"  — 进行中
                "completed"    — 已完成
                "blocked"      — 被阻塞
                "cancelled"    — 已取消
        parent_id: 父任务 ID，None 表示根任务。
        created_at: 创建时间戳（Unix epoch 秒）。
        completed_at: 完成时间戳，未完成时为 None。
        notes: 任务相关的备注列表。
        metadata: 自定义元数据字典（如优先级、标签、指派人等）。
    """

    id: str
    description: str
    status: str = "pending"  # pending | in_progress | completed | blocked | cancelled
    parent_id: str | None = None
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """将 Task 序列化为 dict，用于 JSON 持久化。"""
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status,
            "parent_id": self.parent_id,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "notes": self.notes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Task:
        """从 dict 反序列化创建 Task 对象。"""
        return cls(
            id=d["id"],
            description=d["description"],
            status=d.get("status", "pending"),
            parent_id=d.get("parent_id"),
            created_at=d.get("created_at", time.time()),
            completed_at=d.get("completed_at"),
            notes=d.get("notes", []),
            metadata=d.get("metadata", {}),
        )


class TaskMemory:
    """任务记忆 — 管理目标、子任务和进度跟踪。

    职责：
      - 以层次结构存储任务（通过 parent_id 形成树）
      - 跟踪每个任务的状态变化
      - 统计整体进度
      - 支持任务的中断恢复

    使用方法:
        store = MemoryStore(Path("tasks.json"))
        tm = TaskMemory(store)
        tm.add_task("root_1", "重构 user 模块")
        tm.add_task("sub_1", "拆分 service 层", parent_id="root_1")
        tm.update_status("sub_1", "completed")
        print(tm.progress_summary())  # 查看整体进度
    """

    def __init__(self, store: MemoryStore) -> None:
        """初始化任务记忆。

        Args:
            store: 底层 MemoryStore 实例，用于持久化任务数据。
        """
        self.store = store
        self._tasks: dict[str, Task] = {}  # 内存中的任务字典，键为任务 ID
        self._load()

    def _load(self) -> None:
        """从底层 store 加载任务数据到内存。

        任务数据在 store 中以键 "__tasks__" 存储，值为 {task_id: task_dict} 的字典。
        """
        data = self.store.get("__tasks__", {})
        for tid, td in data.items():
            self._tasks[tid] = Task.from_dict(td)

    def _save(self) -> None:
        """将内存中的任务数据持久化到 store 并 flush 到磁盘。

        每次增删改操作后自动调用，保证数据不丢失。
        """
        data = {tid: t.to_dict() for tid, t in self._tasks.items()}
        self.store.set("__tasks__", data)
        self.store.flush()

    def add_task(
        self, task_id: str, description: str,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Task:
        """添加一个新任务。

        Args:
            task_id: 任务的唯一标识符。
            description: 任务描述。
            parent_id: 父任务 ID，None 表示根任务。
            metadata: 自定义元数据（如优先级、标签等）。

        Returns:
            新创建的 Task 对象。
        """
        task = Task(
            id=task_id,
            description=description,
            parent_id=parent_id,
            metadata=metadata or {},
        )
        self._tasks[task_id] = task
        self._save()
        return task

    def update_status(self, task_id: str, status: str) -> Task | None:
        """更新任务的状态。

        当状态改为 "completed" 时，自动记录完成时间。

        Args:
            task_id: 任务 ID。
            status: 新状态（pending / in_progress / completed / blocked / cancelled）。

        Returns:
            更新后的 Task 对象，如果任务不存在则返回 None。
        """
        task = self._tasks.get(task_id)
        if task is None:
            return None
        task.status = status
        if status == "completed":
            task.completed_at = time.time()  # 自动记录完成时间
        self._save()
        return task

    def add_note(self, task_id: str, note: str) -> Task | None:
        """给任务添加一条备注。

        Args:
            task_id: 任务 ID。
            note: 备注文本（如 "已确认 API 接口定义，等待后端部署"）。

        Returns:
            更新后的 Task 对象，如果任务不存在则返回 None。
        """
        task = self._tasks.get(task_id)
        if task is None:
            return None
        task.notes.append(note)
        self._save()
        return task

    def get_task(self, task_id: str) -> Task | None:
        """获取指定 ID 的任务。

        Args:
            task_id: 任务 ID。

        Returns:
            Task 对象，不存在则返回 None。
        """
        return self._tasks.get(task_id)

    def get_subtasks(self, parent_id: str) -> list[Task]:
        """获取指定父任务的所有子任务。

        Args:
            parent_id: 父任务 ID。

        Returns:
            子任务列表（按添加顺序，不保证额外排序）。
        """
        return [t for t in self._tasks.values() if t.parent_id == parent_id]

    def get_root_tasks(self) -> list[Task]:
        """获取所有根任务（没有父任务的任务）。"""
        return [t for t in self._tasks.values() if t.parent_id is None]

    def get_by_status(self, status: str) -> list[Task]:
        """获取指定状态的所有任务。

        Args:
            status: 状态值（如 "in_progress"、"blocked"）。

        Returns:
            匹配的任务列表。
        """
        return [t for t in self._tasks.values() if t.status == status]

    def list_all(self) -> list[Task]:
        """列出所有任务，按创建时间升序排列。"""
        return sorted(self._tasks.values(), key=lambda t: t.created_at)

    def progress_summary(self) -> dict[str, Any]:
        """返回任务进度的摘要统计。

        Returns:
            包含以下键的字典：
              - total: 任务总数
              - completed: 已完成数
              - in_progress: 进行中数
              - blocked: 被阻塞数
              - pending: 待处理数
              - progress_pct: 进度百分比（0-100）
        """
        all_tasks = list(self._tasks.values())
        total = len(all_tasks)
        completed = sum(1 for t in all_tasks if t.status == "completed")
        in_progress = sum(1 for t in all_tasks if t.status == "in_progress")
        blocked = sum(1 for t in all_tasks if t.status == "blocked")
        return {
            "total": total,
            "completed": completed,
            "in_progress": in_progress,
            "blocked": blocked,
            "pending": total - completed - in_progress - blocked,
            "progress_pct": (completed / total * 100) if total > 0 else 0,
        }

    def delete_task(self, task_id: str) -> bool:
        """删除一个任务及其所有子任务（级联删除）。

        Args:
            task_id: 要删除的任务 ID。

        Returns:
            True 表示成功删除，False 表示任务不存在。
        """
        if task_id not in self._tasks:
            return False
        # 级联删除所有子任务
        subtasks = self.get_subtasks(task_id)
        for st in subtasks:
            del self._tasks[st.id]
        del self._tasks[task_id]
        self._save()
        return True
