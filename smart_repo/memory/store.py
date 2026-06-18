"""
内存存储器 (MemoryStore) — 基于 JSON 文件的持久化键值存储。

============================================================
为什么需要 MemoryStore？
============================================================

在 agent 系统中，需要跨对话回合记住以下信息：
  - 当前任务状态（task memory）
  - 已读取文件的缓存（file memory）
  - 关键决策和错误日志（process notes）

这些信息不能只放在内存中——对话回合结束时进程可能退出。
因此需要一个轻量级的持久化层。选择 JSON 文件而非 SQLite/Redis 的原因：
  1. 零依赖：不需要安装数据库驱动或启动外部服务
  2. 可调试：JSON 文件可以直接用文本编辑器查看和修改
  3. 够用：agent 的记忆数据量通常在 KB-MB 级别，JSON 完全能胜任

============================================================
设计要点
============================================================

1. TTL 过期机制：
   每个存储条目可以设置 TTL（Time To Live），到期后自动视为不存在。
   这用于防止旧缓存无限增长——例如文件缓存超过 5 分钟后重新读取磁盘。

2. 原子写入：
   先写入 .tmp 临时文件，再通过 rename 替换正式文件。
   这能防止写入过程中进程崩溃导致文件损坏（POSIX rename 是原子的）。

3. 延迟写入 (dirty flag)：
   通过 _dirty 标记跟踪是否有未持久化的修改，
   只有显式调用 flush() 或下一次 set/delete 时才写盘，
   避免频繁的磁盘 I/O。

4. 元数据包装：
   每个值被包装为 {"_value": ..., "_created": ..., "_ttl": ...} 结构，
   这样可以在不改变调用者代码的情况下支持 TTL 和创建时间追踪。
============================================================
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class MemoryStore:
    """基于 JSON 文件的持久化键值存储。

    职责：
      - 提供类似 dict 的接口（get/set/delete/exists/keys/values/items）
      - 支持 TTL 过期（条目到期后自动清理）
      - 原子写入保证数据安全
      - 每个记忆类别（task、file、notes）拥有独立的 MemoryStore 实例

    使用方法:
        store = MemoryStore(Path("task_memory.json"), ttl_seconds=3600)
        store.set("task_1", {"name": "fix login bug", "status": "pending"})
        task = store.get("task_1")  # 自动检查 TTL，过期返回 None
    """

    def __init__(self, file_path: Path, ttl_seconds: int = 0) -> None:
        """初始化存储器。

        Args:
            file_path: JSON 文件的路径（文件不存在时会自动创建）。
            ttl_seconds: 条目的默认 TTL（秒），0 表示永不过期。
                         也可以在 set() 时为单个条目覆盖此值。
        """
        self.file_path = file_path
        self.ttl_seconds = ttl_seconds
        self._data: dict[str, dict[str, Any]] = {}  # 内存中的数据字典
        self._dirty = False  # 是否有未持久化的修改
        self._load()

    def _load(self) -> None:
        """从磁盘加载数据到内存。

        如果文件不存在或 JSON 格式损坏，初始化为空字典。
        JSON 损坏的情况：手动编辑文件时写错格式，这时放弃加载而非崩溃。
        """
        if self.file_path.exists():
            try:
                self._data = json.loads(self.file_path.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                # JSON 格式错误或文件读取失败 → 初始化为空
                self._data = {}
        else:
            self._data = {}

    def _save(self) -> None:
        """将数据原子化持久化到磁盘。

        原子写入策略：
          1. 确保父目录存在
          2. 将数据写入 .tmp 临时文件
          3. 调用 Path.replace() 替换正式文件（在 POSIX 上是原子操作）
        这能防止写盘中途崩溃导致文件损坏。
        """
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.file_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        tmp_path.replace(self.file_path)
        self._dirty = False

    def get(self, key: str, default: Any = None) -> Any:
        """获取键对应的值。键不存在或已过期时返回 default。

        TTL 检查逻辑：
          - 如果条目有 _ttl > 0 且 _created + _ttl < 当前时间，视为过期
          - 过期条目会被自动删除（惰性删除策略）

        Args:
            key: 键名。
            default: 键不存在或过期时的默认返回值。

        Returns:
            存储的值（已解包，不含元数据），或 default。
        """
        entry = self._data.get(key)
        if entry is None:
            return default
        # TTL 过期检查
        ttl = entry.get("_ttl", 0) if isinstance(entry, dict) else 0
        created = entry.get("_created", 0) if isinstance(entry, dict) else 0
        if ttl > 0 and created > 0:
            if time.time() - created > ttl:
                # 惰性删除：过期条目在访问时清理
                del self._data[key]
                self._dirty = True
                return default
        # 返回解包后的值（去除元数据包装）
        if isinstance(entry, dict) and "_value" in entry:
            return entry["_value"]
        return entry

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """设置键值对，可选指定 TTL。

        Args:
            key: 键名。
            value: 要存储的值（任意可 JSON 序列化的 Python 对象）。
            ttl_seconds: 该条目的 TTL（秒），为 None 时使用存储器的默认 TTL。
                         0 表示永不过期。
        """
        ttl = ttl_seconds if ttl_seconds is not None else self.ttl_seconds
        # 包装为元数据结构：值 + 创建时间 + TTL
        self._data[key] = {
            "_value": value,
            "_created": time.time(),
            "_ttl": ttl,
        }
        self._dirty = True

    def delete(self, key: str) -> bool:
        """删除一个键。返回 True 表示键存在并已删除。"""
        if key in self._data:
            del self._data[key]
            self._dirty = True
            return True
        return False

    def exists(self, key: str) -> bool:
        """检查键是否存在且未过期。"""
        return self.get(key) is not None

    def keys(self, prefix: str = "") -> list[str]:
        """列出所有键，可选择按前缀过滤。

        注意：会过滤掉已过期的键（惰性删除 + 过滤）。
        """
        all_keys = [k for k in self._data if k.startswith(prefix)]
        # 过滤已过期的键（get() 会触发惰性删除）
        valid = [k for k in all_keys if self.get(k) is not None]
        return sorted(valid)

    def values(self, prefix: str = "") -> list[Any]:
        """列出所有值，可选择按键前缀过滤。"""
        return [self.get(k) for k in self.keys(prefix)]

    def items(self, prefix: str = "") -> list[tuple[str, Any]]:
        """列出所有 (键, 值) 对，可选择按键前缀过滤。"""
        return [(k, self.get(k)) for k in self.keys(prefix)]

    def clear(self) -> None:
        """清空所有条目。"""
        self._data.clear()
        self._dirty = True

    def flush(self) -> None:
        """强制将脏数据写入磁盘。

        通常在关键操作后调用（如任务状态变更），确保数据不丢失。
        如果数据没有变更（_dirty=False），则跳过写入。
        """
        if self._dirty:
            self._save()

    def __len__(self) -> int:
        return len(self.keys())

    def __contains__(self, key: str) -> bool:
        return self.exists(key)

    def __getitem__(self, key: str) -> Any:
        """支持 store[key] 语法。键不存在时抛出 KeyError。"""
        val = self.get(key)
        if val is None:
            raise KeyError(key)
        return val

    def __setitem__(self, key: str, value: Any) -> None:
        """支持 store[key] = value 语法（使用默认 TTL）。"""
        self.set(key, value)

    def __delitem__(self, key: str) -> None:
        """支持 del store[key] 语法。键不存在时抛出 KeyError。"""
        if not self.delete(key):
            raise KeyError(key)
