"""
文件记忆 (FileMemory) — 代码库文件缓存，消除重复的文件读取。

============================================================
三种记忆的职责划分（回顾）
============================================================

SmartRepo 的 memory 系统包含三种记忆：

1. TaskMemory  (task_memory.py)  — 追踪"要做什么"：目标 → 子任务 → 进度
2. FileMemory  (本文件)          — 缓存"文件里有什么"：路径 → 摘要/符号表
3. ProcessNotes (notes.py)       — 记录"发生了什么"：决策/错误/洞察/里程碑

============================================================
FileMemory 设计背景与核心价值
============================================================

核心洞察：在 agent-based coding 场景中，LLM 经常重复读取相同的文件。
例如：用户在对话中多次提到同一个模块，agent 每次都要调用 Read 工具
读取文件内容，然后将文件内容放入上下文发送给 LLM。

这带来了两个问题：
  1. 重复磁盘 I/O：同一文件被多次读取，浪费时间和资源
  2. 重复 token 消耗：同一文件内容多次出现在上下文窗口中，
     挤占了有效对话历史的空间

FileMemory 的解决方案：
  - 缓存文件的关键元数据（摘要、符号表、行数、大小、修改时间）
  - 当 agent 需要"读取"一个文件时，先检查缓存
  - 如果文件自上次缓存后未修改，直接返回缓存的摘要（而非完整内容）
  - 仅在文件确实被修改后才重新读取和缓存

缓存失效策略（两层保护）：
  1. TTL 过期：超过 ttl_seconds（默认 300秒 = 5分钟）的缓存自动失效
  2. 文件修改检测：比较磁盘上文件的 mtime（修改时间），
     如果比缓存中的 last_modified 更新，则缓存失效

============================================================
数据结构
============================================================

FileCacheEntry 存储以下信息：
  - path:            文件相对路径
  - content_hash:    文件内容的 SHA256 哈希（前 16 字符），用于快速比对
  - summary:         文件摘要（可能是 LLM 生成的，也可能是自动生成的简短描述）
  - line_count:      文件行数
  - size_bytes:      文件字节数
  - last_modified:   文件在磁盘上的最后修改时间（mtime）
  - cached_at:       缓存创建时间
  - symbols:         代码符号列表（函数名、类名），通过正则提取
  - metadata:        自定义元数据（扩展用）

符号提取（_extract_symbols）：
  对于 .py 文件，使用正则表达式提取 def/class/async def 定义，
  生成符号列表。这有助于 agent 快速了解文件的结构，而不必读取全部内容。
============================================================
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from smart_repo.memory.store import MemoryStore


@dataclass
class FileCacheEntry:
    """文件缓存条目 — 缓存一个文件的元数据。

    Attributes:
        path: 文件的相对路径（相对于工作区根目录）。
        content_hash: 文件内容的 SHA256 哈希值前 16 字符。
        summary: 文件摘要（可能是 LLM 生成或自动生成）。
        line_count: 文件行数。
        size_bytes: 文件字节数。
        last_modified: 文件在磁盘上的最后修改时间戳。
        cached_at: 缓存条目的创建时间戳。
        symbols: 代码符号列表（函数名、类名等），通过正则提取。
        metadata: 自定义元数据字典。
    """

    path: str
    content_hash: str
    summary: str
    line_count: int
    size_bytes: int
    last_modified: float
    cached_at: float = field(default_factory=time.time)
    symbols: list[str] = field(default_factory=list)  # 函数名/类名列表
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def age_seconds(self) -> float:
        """缓存条目的年龄（秒）。"""
        return time.time() - self.cached_at

    def is_stale(self, ttl_seconds: int = 300,
                 current_mtime: float | None = None) -> bool:
        """检查缓存是否过期。

        两种过期条件（满足任一即过期）：
          1. TTL 过期：缓存年龄超过 ttl_seconds
          2. 文件更新：磁盘文件修改时间比缓存记录的更晚

        Args:
            ttl_seconds: TTL 阈值（秒），默认 300 秒。
            current_mtime: 磁盘上文件的当前修改时间戳，None 表示不检查。

        Returns:
            True 表示缓存过期，应重新读取文件。
        """
        if self.age_seconds > ttl_seconds:
            return True
        if current_mtime is not None and current_mtime > self.last_modified:
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict，用于 JSON 持久化。"""
        return {
            "path": self.path,
            "content_hash": self.content_hash,
            "summary": self.summary,
            "line_count": self.line_count,
            "size_bytes": self.size_bytes,
            "last_modified": self.last_modified,
            "cached_at": self.cached_at,
            "symbols": self.symbols,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FileCacheEntry:
        """从 dict 反序列化创建 FileCacheEntry。"""
        return cls(
            path=d["path"],
            content_hash=d["content_hash"],
            summary=d["summary"],
            line_count=d["line_count"],
            size_bytes=d["size_bytes"],
            last_modified=d["last_modified"],
            cached_at=d.get("cached_at", time.time()),
            symbols=d.get("symbols", []),
            metadata=d.get("metadata", {}),
        )


class FileMemory:
    """文件记忆 — 代码库文件缓存，消除重复文件读取。

    职责：
      - 缓存文件元数据（路径 → 摘要、符号表、修改时间）
      - 自动检测文件变更（mtime 对比 + TTL 过期）
      - 自动提取代码符号（Python 的函数/类定义）
      - 提供缓存统计信息

    使用方法:
        store = MemoryStore(Path("file_cache.json"))
        fm = FileMemory(store, ttl_seconds=300, workspace=Path("/path/to/project"))

        # 缓存一个文件
        entry = fm.put("src/main.py", content, summary="主入口模块")

        # 读取缓存（自动检查是否过期）
        cached = fm.get("src/main.py")
        if cached is None:
            # 缓存未命中或过期，需要重新读取文件
            pass
    """

    def __init__(
        self,
        store: MemoryStore,
        ttl_seconds: int = 300,
        workspace: Path | None = None,
    ) -> None:
        """初始化文件记忆。

        Args:
            store: 底层 MemoryStore 实例，用于持久化缓存数据。
            ttl_seconds: 缓存 TTL（秒），默认 300 秒（5 分钟）。
                         过期缓存会被自动清除。
            workspace: 工作区根目录，用于拼接文件的完整磁盘路径。
                       为 None 时使用当前工作目录。
        """
        self.store = store
        self.ttl_seconds = ttl_seconds
        self.workspace = workspace or Path.cwd()
        self._cache: dict[str, FileCacheEntry] = {}  # 内存中的缓存字典，键为文件相对路径
        self._load()

    def _load(self) -> None:
        """从底层 store 加载缓存数据到内存。

        缓存数据在 store 中以键 "__file_cache__" 存储，值为 {path: entry_dict} 的字典。
        """
        data = self.store.get("__file_cache__", {})
        for path, entry_dict in data.items():
            self._cache[path] = FileCacheEntry.from_dict(entry_dict)

    def _save(self) -> None:
        """将内存缓存持久化到 store 并 flush 到磁盘。"""
        data = {p: e.to_dict() for p, e in self._cache.items()}
        self.store.set("__file_cache__", data)
        self.store.flush()

    def _hash_content(self, content: str) -> str:
        """计算文件内容的 SHA256 哈希值前 16 字符。

        用于快速比对文件内容是否有变化（不依赖于 mtime）。
        """
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _extract_symbols(self, content: str, file_path: str) -> list[str]:
        """从源代码中提取函数名和类名（符号提取）。

        当前仅支持 Python 文件（.py），通过正则匹配 def/class/async def。
        对于其他文件类型，返回空列表。
        未来可扩展支持更多语言（如 JS/TS 的 function/class/const）。

        Args:
            content: 文件内容。
            file_path: 文件路径（用于判断文件类型）。

        Returns:
            符号名列表（按定义顺序）。
        """
        symbols = []
        suffix = Path(file_path).suffix.lower()
        if suffix == ".py":
            import re
            for match in re.finditer(
                r'^\s*(?:def|class|async def)\s+(\w+)',
                content, re.MULTILINE,
            ):
                symbols.append(match.group(1))
        return symbols

    def get(self, file_path: str) -> FileCacheEntry | None:
        """获取文件的缓存信息。

        自动执行缓存失效检查：
          1. 缓存不存在 → 返回 None
          2. TTL 过期 → 删除缓存 → 返回 None
          3. 磁盘文件已更新（mtime 更新） → 删除缓存 → 返回 None
          4. 缓存有效 → 返回 FileCacheEntry

        Args:
            file_path: 文件的相对路径（相对于 workspace）。

        Returns:
            FileCacheEntry（缓存有效时），None（缓存未命中或已失效时）。
        """
        entry = self._cache.get(file_path)
        if entry is None:
            return None

        # 检查磁盘上的文件是否被修改（mtime 对比）
        full_path = self.workspace / file_path
        current_mtime = full_path.stat().st_mtime if full_path.exists() else None

        if entry.is_stale(self.ttl_seconds, current_mtime):
            # 缓存已过期，删除并返回 None
            del self._cache[file_path]
            self._save()
            return None

        return entry

    def put(
        self,
        file_path: str,
        content: str,
        summary: str = "",
        symbols: list[str] | None = None,
    ) -> FileCacheEntry:
        """缓存一个文件的信息。

        如果未提供 summary，则自动生成简短摘要（文件名 + 行数 + 字节数）。
        如果未提供 symbols，则自动从源码中提取。
        自动计算 content_hash 用于快速比对。

        Args:
            file_path: 文件的相对路径。
            content: 文件的完整内容。
            summary: 文件摘要（可选，留空则自动生成）。
            symbols: 代码符号列表（可选，留空则自动提取）。

        Returns:
            新创建的 FileCacheEntry。
        """
        full_path = self.workspace / file_path
        mtime = full_path.stat().st_mtime if full_path.exists() else time.time()

        if symbols is None:
            symbols = self._extract_symbols(content, file_path)

        if not summary:
            # 自动生成简短摘要：文件名 + 统计信息
            lines = content.splitlines()
            summary = (
                f"{Path(file_path).name}: {len(lines)} lines, "
                f"{len(content)} bytes"
            )
            if symbols:
                summary += f", symbols: {', '.join(symbols[:20])}"

        entry = FileCacheEntry(
            path=file_path,
            content_hash=self._hash_content(content),
            summary=summary,
            line_count=len(content.splitlines()),
            size_bytes=len(content.encode()),
            last_modified=mtime,
            symbols=symbols,
        )

        self._cache[file_path] = entry
        self._save()
        return entry

    def invalidate(self, file_path: str) -> bool:
        """使指定文件的缓存失效。

        Args:
            file_path: 文件的相对路径。

        Returns:
            True 表示缓存存在并已删除，False 表示缓存原本就不存在。
        """
        if file_path in self._cache:
            del self._cache[file_path]
            self._save()
            return True
        return False

    def invalidate_all(self) -> None:
        """清空所有文件缓存。

        适用于工作区切换或强制刷新全部缓存的场景。
        """
        self._cache.clear()
        self._save()

    def list_cached(self) -> list[str]:
        """列出所有已缓存文件的路径（排序后）。"""
        return sorted(self._cache.keys())

    def cache_stats(self) -> dict[str, Any]:
        """返回缓存统计信息。

        Returns:
            包含以下键的字典：
              - cached_files: 缓存文件数量
              - total_size_bytes: 缓存文件的总大小（字节）
              - avg_age_seconds: 缓存条目的平均年龄（秒）
              - ttl_seconds: 当前 TTL 设置
        """
        total_size = sum(e.size_bytes for e in self._cache.values())
        return {
            "cached_files": len(self._cache),
            "total_size_bytes": total_size,
            "avg_age_seconds": (
                sum(e.age_seconds for e in self._cache.values()) / len(self._cache)
                if self._cache else 0
            ),
            "ttl_seconds": self.ttl_seconds,
        }
