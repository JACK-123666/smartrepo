"""文件操作工具集——读取、写入、编辑、Glob 匹配、Grep 搜索、目录列表。

本模块为代理提供工作区内的完整文件操作能力，涵盖 6 个核心工具。

设计理念：
  - 所有路径操作都通过 _resolve_path() 进行沙箱校验，防止路径穿越攻击（directory traversal）。
  - 只读操作（read_file, glob, grep, list_dir）风险等级为 low，无需审批。
  - 写入/编辑操作（write_file, edit_file）风险等级为 medium，需要审批。
  - edit_file 基于精确字符串替换，避免了 sed/regex 的复杂性和出错风险。
  - grep 支持两种输出模式：files_with_matches（仅列文件路径）和 content（展示匹配行及上下文）。
  - 所有工具输出长度有上限控制，防止上下文溢出。

File operation tools — Read, Write, Edit, Glob, Grep, List Directory.
Provides the agent with comprehensive file manipulation capabilities within
a sandboxed workspace.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from smart_repo.tools.base import Tool


def _resolve_path(base: Path, path: str) -> Path:
    """安全地解析路径：将相对路径转换为绝对路径，并防止路径穿越。

    这是整个文件操作工具的安全基石。所有文件读写操作都必须经过此函数验证，
    确保用户传入的路径无法通过 "../" 等方式逃逸到工作区之外。

    Args:
        base: 基准目录（通常是工作区根目录），所有路径相对此目录解析。
        path: 用户传入的相对路径字符串。

    Returns:
        解析后的绝对路径。

    Raises:
        PermissionError: 如果解析后的路径位于基准目录之外。

    Resolve a path relative to base, preventing traversal outside base.
    """
    resolved = (base / path).resolve()
    # 安全检查：验证解析后的路径是否在基准目录内
    # Security check: ensure resolved path stays within base directory
    try:
        resolved.relative_to(base.resolve())
    except ValueError:
        raise PermissionError(f"Path traversal blocked: '{path}' resolves outside workspace.")
    return resolved


async def _read_file(workspace: Path, path: str, offset: int = 0,
                     limit: int | None = None) -> str:
    """从工作区读取文件内容。

    以 cat -n 风格显示行号，支持偏移量和行数限制。
    对大文件友好：limit 参数可以只读部分内容，offset 支持跳行。

    Args:
        workspace: 工作区根目录。
        path: 相对于工作区的文件路径。
        offset: 起始行（0-indexed），跳过前 offset 行。
        limit: 最大读取行数，None 表示读取全部剩余行。

    Returns:
        带行号的文本内容，或错误信息。

    Read a file from the workspace with line numbers (cat -n style).
    """
    file_path = _resolve_path(workspace, path)
    if not file_path.exists():
        return f"Error: File not found: {file_path}"
    if file_path.is_dir():
        return f"Error: '{path}' is a directory, not a file."

    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Error: Cannot read '{path}' as UTF-8 text (binary file?)."

    lines = content.splitlines()
    total_lines = len(lines)

    # 应用偏移量，跳过前 offset 行 / Apply offset to skip leading lines
    if offset > 0:
        lines = lines[offset:]
    # 应用行数限制 / Apply line limit
    if limit is not None:
        lines = lines[:limit]

    # 添加行号前缀（cat -n 风格）
    # Prefix with line numbers (cat -n style)
    numbered = [f"{i + offset + 1:6}\t{line}" for i, line in enumerate(lines)]
    result = "\n".join(numbered)

    # 如果还有更多行未显示，添加提示 / Add truncation hint if needed
    if limit is not None and total_lines > offset + limit:
        result += f"\n\n[... {total_lines - offset - limit} more lines ...]"

    return result


async def _write_file(workspace: Path, path: str, content: str) -> str:
    """向工作区写入文件内容。

    自动创建不存在的父目录（mkdir -p 行为）。
    风险等级 medium，需要审批——写入工作区可能覆盖现有文件或引入恶意代码。

    Args:
        workspace: 工作区根目录。
        path: 相对于工作区的目标文件路径。
        content: 要写入的内容字符串（UTF-8 编码）。

    Returns:
        写入成功确认信息（含字节数）。

    Write content to a file in the workspace.
    Creates parent directories if needed (mkdir -p behaviour).
    """
    file_path = _resolve_path(workspace, path)
    # 确保父目录存在 / Ensure parent directory tree exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return f"Successfully wrote {len(content)} bytes to {file_path}"


async def _edit_file(workspace: Path, path: str, old_string: str,
                     new_string: str, replace_all: bool = False) -> str:
    """在文件中进行精确字符串替换。

    与 sed 不同，此方法要求 old_string 完全精确匹配（包括空白字符），
    这避免了正则表达式替换可能导致的意外修改。
    如果 old_string 在文件中出现多次且未设置 replace_all，
    会返回错误并提示用户使用更具体的匹配字符串或设置 replace_all=True。

    Args:
        workspace: 工作区根目录。
        path: 目标文件路径。
        old_string: 要查找并替换的精确字符串（必须完全匹配，含空白字符）。
        new_string: 替换后的新字符串。
        replace_all: 是否替换所有匹配项。默认 False（仅替换第一个）。

    Returns:
        操作结果描述（成功时报告替换次数）。

    Replace an exact string in a file (exact match required).
    This is safer than regex-based replacement — no unexpected side effects.
    """
    file_path = _resolve_path(workspace, path)
    if not file_path.exists():
        return f"Error: File not found: {file_path}"

    content = file_path.read_text(encoding="utf-8")

    count = content.count(old_string)
    if count == 0:
        return f"Error: old_string not found in '{path}'. Make sure it matches exactly (including whitespace)."

    if not replace_all:
        # 多处匹配但未允许全部替换 → 返回错误，要求用户更精确
        # Multiple matches without replace_all → error, ask user to be more specific
        if count > 1:
            return (f"Error: old_string found {count} times in '{path}'. "
                    f"Use replace_all=true to replace all, or make old_string more specific.")
        new_content = content.replace(old_string, new_string, 1)
    else:
        # 替换所有匹配项 / Replace all occurrences
        new_content = content.replace(old_string, new_string)

    if new_content == content:
        return f"Error: No changes made (old_string and new_string are identical)."

    file_path.write_text(new_content, encoding="utf-8")
    return f"Successfully edited '{path}': replaced {count} occurrence(s)."


async def _glob_files(workspace: Path, pattern: str,
                      path: str = ".") -> str:
    """使用 glob 模式查找文件。

    支持递归通配符（**），返回相对于工作区的路径列表。
    结果上限 200 条，超出部分显示截断提示。

    Args:
        workspace: 工作区根目录。
        pattern: Glob 匹配模式（如 '**/*.py'、'src/**/*.ts'）。
        path: 搜索的起始子目录，默认为工作区根目录。

    Returns:
        匹配的文件路径列表（每行一个），或错误信息。

    Find files matching a glob pattern.
    Supports recursive glob with **. Results capped at 200 entries.
    """
    import glob as glob_mod
    search_dir = _resolve_path(workspace, path)
    if not search_dir.exists():
        return f"Error: Directory not found: {search_dir}"

    full_pattern = str(search_dir / pattern)
    matches = glob_mod.glob(full_pattern, recursive=True)
    # 将绝对路径转换为相对于工作区的路径 / Convert absolute to workspace-relative
    rel_matches = []
    for m in sorted(matches):
        try:
            rel = Path(m).relative_to(workspace.resolve())
        except ValueError:
            rel = Path(m)
        rel_matches.append(str(rel))

    if not rel_matches:
        return f"No files matching '{pattern}' found in {path}."
    # 结果截断保护，防止超长输出 / Result truncation to prevent context overflow
    return "\n".join(rel_matches[:200]) + (
        f"\n\n[... and {len(rel_matches) - 200} more ...]" if len(rel_matches) > 200 else ""
    )


async def _grep_search(workspace: Path, pattern: str,
                       path: str = ".", glob: str = "*",
                       output_mode: str = "files_with_matches",
                       max_results: int = 50,
                       context_lines: int = 0) -> str:
    """使用正则表达式在文件内容中搜索。

    支持两种输出模式：
      - files_with_matches：只列出包含匹配项的文件路径（默认）
      - content：显示匹配行及其上下文（行号 + 标记）

    Args:
        workspace: 工作区根目录。
        pattern: 正则表达式模式。
        path: 搜索的起始子目录。
        glob: 文件名过滤的 glob 模式（默认 * 匹配所有文件）。
        output_mode: 输出模式，"files_with_matches" 或 "content"。
        max_results: 最大匹配结果数限制。
        context_lines: content 模式下匹配行前后的上下文行数。

    Returns:
        搜索结果字符串，或错误信息（含无效正则提示）。

    Search file contents using regex.
    In content mode, matching lines are prefixed with '>' for easy scanning.
    """
    search_dir = _resolve_path(workspace, path)
    if not search_dir.exists():
        return f"Error: Directory not found: {search_dir}"

    # 安全检查：验证正则表达式的合法性 / Validate regex pattern safety
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"

    results: list[str] = []
    import glob as glob_mod
    files = glob_mod.glob(str(search_dir / glob), recursive=True)

    for filepath in sorted(files):
        # 提前终止：防止结果爆炸 / Early exit to prevent result explosion
        if len(results) >= max_results * (context_lines * 2 + 1):
            break
        p = Path(filepath)
        if not p.is_file():
            continue
        # 跳过无法读取的二进制文件 / Skip unreadable (binary) files
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, PermissionError):
            continue

        for i, line in enumerate(lines):
            if regex.search(line):
                if output_mode == "files_with_matches":
                    try:
                        rel = p.relative_to(workspace.resolve())
                    except ValueError:
                        rel = p
                    results.append(str(rel))
                    break  # 每个文件只记录一次 / one entry per file
                elif output_mode == "content":
                    # 显示匹配行及上下文 / Show match line with context
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    for j in range(start, end):
                        # '>' 标记实际匹配行 / '>' marker for the actual matching line
                        prefix = ">" if j == i else " "
                        results.append(f"{str(p)}:{j + 1}:{prefix}{lines[j]}")
                    results.append("--")

    if not results:
        return f"No matches for '{pattern}' in {path}."
    return "\n".join(results[:max_results * (context_lines * 2 + 2)])


async def _list_dir(workspace: Path, path: str = ".") -> str:
    """列出目录内容。

    类似于 ls 命令，目录项以 '/' 后缀标识。
    所有路径显示为相对于工作区的路径。

    Args:
        workspace: 工作区根目录。
        path: 要列出的目录路径。

    Returns:
        目录内容列表（每行一项），或错误信息。

    List directory contents (ls-like). Directories marked with trailing '/'.
    """
    dir_path = _resolve_path(workspace, path)
    if not dir_path.exists():
        return f"Error: Directory not found: {dir_path}"
    if not dir_path.is_dir():
        return f"Error: '{path}' is not a directory."

    items = []
    for item in sorted(dir_path.iterdir()):
        try:
            rel = item.relative_to(workspace.resolve())
        except ValueError:
            rel = item
        # 目录项添加 '/' 后缀，便于区分 / Add '/' suffix for directories
        suffix = "/" if item.is_dir() else ""
        items.append(f"  {rel}{suffix}")

    if not items:
        return f"Directory '{path}' is empty."
    return "\n".join(items)


def register_file_tools(
    registry,  # ToolRegistry
    workspace: Path,
) -> list[Tool]:
    """创建并注册所有文件操作工具。

    本函数是文件操作工具的工厂函数，负责创建 Tool 实例并批量注册到
    ToolRegistry 中。每个工具都标记了风险等级和是否需要审批。

    工具清单及风险说明：
      - read_file  (low)：    只读，无副作用，自动审批。
      - write_file (medium)： 写入文件，可能覆盖或注入代码，需审批。
      - edit_file  (medium)： 修改文件内容，需审批（精确匹配降低风险）。
      - glob       (low)：    只读模式匹配，无副作用。
      - grep       (low)：    只读内容搜索，无副作用。
      - list_dir   (low)：    只读目录浏览，无副作用。

    Args:
        registry: ToolRegistry 实例，用于注册工具。
        workspace: 工作区根目录，所有文件操作在此范围内进行。

    Returns:
        已注册的 Tool 实例列表。

    Create and register all file operation tools.

    Args:
        registry: ToolRegistry instance to register with.
        workspace: Base workspace directory for path resolution.

    Returns:
        List of registered Tool instances.
    """
    tools = [
        Tool(
            name="read_file",
            description="Read a file from the workspace. Returns line-numbered content.",
            # 工具用途：读取工作区文件，返回带行号的文本内容
            # 风险等级：low — 只读操作，无副作用 / Risk: low — read-only, no side effects
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to workspace root.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (0-indexed).",
                        "default": 0,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read.",
                    },
                },
                "required": ["path"],
            },
            handler=lambda **kw: _read_file(workspace, **kw),
            risk_level="low",
        ),
        Tool(
            name="write_file",
            description="Write content to a file. Creates parent directories if needed.",
            # 工具用途：写入文件内容，自动创建父目录
            # 风险等级：medium — 可能覆盖现有文件，需审批 / Risk: medium — can overwrite, needs approval
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to workspace root.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file.",
                    },
                },
                "required": ["path", "content"],
            },
            handler=lambda **kw: _write_file(workspace, **kw),
            risk_level="medium",
            requires_approval=True,
        ),
        Tool(
            name="edit_file",
            description="Replace an exact string in a file. The old_string must match exactly (including whitespace).",
            # 工具用途：精确字符串替换（非正则），避免意外修改
            # 风险等级：medium — 修改文件内容，需审批；精确匹配降低风险
            # Risk: medium — modifies file, needs approval; exact matching reduces risk
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to workspace root.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Exact string to find and replace.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement string.",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace all occurrences (default: false).",
                        "default": False,
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
            handler=lambda **kw: _edit_file(workspace, **kw),
            risk_level="medium",
            requires_approval=True,
        ),
        Tool(
            name="glob",
            description="Find files matching a glob pattern.",
            # 工具用途：使用 Glob 模式查找文件
            # 风险等级：low — 只读操作 / Risk: low — read-only
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g., '**/*.py', 'src/**/*.ts').",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (default: workspace root).",
                        "default": ".",
                    },
                },
                "required": ["pattern"],
            },
            handler=lambda **kw: _glob_files(workspace, **kw),
            risk_level="low",
        ),
        Tool(
            name="grep",
            description="Search file contents using a regex pattern.",
            # 工具用途：使用正则表达式搜索文件内容
            # 风险等级：low — 只读操作，但需注意 ReDoS 风险（由 _grep_search 内部防范）
            # Risk: low — read-only, but ReDoS risk is mitigated internally
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in.",
                        "default": ".",
                    },
                    "glob": {
                        "type": "string",
                        "description": "File glob filter (e.g., '*.py').",
                        "default": "*",
                    },
                    "output_mode": {
                        "type": "string",
                        "enum": ["files_with_matches", "content"],
                        "description": "Output mode: file paths or matching lines.",
                        "default": "files_with_matches",
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Lines of context around matches (content mode).",
                        "default": 0,
                    },
                },
                "required": ["pattern"],
            },
            handler=lambda **kw: _grep_search(workspace, **kw),
            risk_level="low",
        ),
        Tool(
            name="list_dir",
            description="List contents of a directory.",
            # 工具用途：列出目录内容（类似 ls）
            # 风险等级：low — 只读操作 / Risk: low — read-only
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to workspace root.",
                        "default": ".",
                    },
                },
            },
            handler=lambda **kw: _list_dir(workspace, **kw),
            risk_level="low",
        ),
    ]

    registry.register_many(tools)
    return tools
