"""Git 操作工具集——状态查看、差异对比、日志、分支、提交。

本模块为代理提供 Git 仓库的完整操作能力，覆盖常用 Git 命令。

设计理念：
  - 所有 Git 子命令通过统一的 _run_git() 助手执行，减少重复代码。
  - 只读操作（status, diff, log, branch, show）风险等级为 low，自动审批。
  - 写入操作（commit）风险等级为 high，需要审批——提交会永久修改仓库历史。
  - 未提供 push 工具——push 是网络操作且不可逆，应当由人工执行。
  - diff 和 show 输出上限 8000 字符，防止大 commit 撑爆上下文。
  - 使用 asyncio.create_subprocess_exec 直接调用 git 二进制，不经过 Shell，
    避免命令注入风险（参数以列表形式传递，不会被 Shell 解释）。

Git operation tools — status, diff, log, branch, commit, show.
Provides the agent with Git repository awareness and controlled mutation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from smart_repo.tools.base import Tool


async def _run_git(workspace: Path, args: list[str],
                   timeout: int = 30_000) -> str:
    """执行 Git 命令并返回其输出。

    这是所有 Git 操作的基础执行函数。使用 subprocess_exec 而非 subprocess_shell，
    参数以列表形式传递——这从根本上避免了命令注入（Shell 注入）风险，
    因为 git 的参数不会被 Shell 解析器解释。

    安全要点：
      - 使用 exec（非 shell）避免注入
      - 超时保护防止长时间运行的 Git 操作
      - 异常捕获防止 Git 未安装导致崩溃

    Args:
        workspace: Git 仓库的工作目录。
        args: Git 命令参数列表（不含 'git' 本身），如 ["status", "--short"]。
        timeout: 超时时间（毫秒），默认 30 秒。

    Returns:
        Git 命令的标准输出 + 标准错误。

    Run a git command and return its output.
    Uses subprocess_exec (not shell) — arguments are passed as a list, preventing
    shell injection since nothing is interpreted by a shell parser.
    """
    try:
        # 使用 exec（非 shell）执行 git，避免 Shell 注入
        # Use exec (not shell) to prevent command injection
        proc = await asyncio.create_subprocess_exec(
            "git", *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(workspace),
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout / 1000,
            )
        except asyncio.TimeoutError:
            # 超时后强制终止，防止资源泄露
            # Kill on timeout to prevent hanging subprocess
            proc.kill()
            await proc.wait()
            return f"TIMEOUT: Git command exceeded timeout."

        result = stdout.decode("utf-8", errors="replace")
        if stderr:
            err = stderr.decode("utf-8", errors="replace")
            if err.strip():
                result += f"\n[git stderr]\n{err}"

        return result.strip() or "(no output)"

    except FileNotFoundError:
        return "Error: Git is not installed or not found in PATH."
    except Exception as e:
        return f"Error running git: {type(e).__name__}: {e}"


async def _git_status(workspace: Path) -> str:
    """显示工作树状态。

    对应 git status --short --branch，简洁格式 + 当前分支信息。
    风险等级：low（只读操作）。

    Show working tree status (short format with branch info).
    """
    return await _run_git(workspace, ["status", "--short", "--branch"])


async def _git_diff(workspace: Path, staged: bool = False,
                    path: str = "") -> str:
    """显示工作树中的变更。

    默认显示未暂存的变更，staged=True 时显示已暂存的变更。
    path 参数可限制到特定文件。
    输出上限 8000 字符。

    Args:
        workspace: Git 仓库工作目录。
        staged: True 显示已暂存变更，False（默认）显示未暂存变更。
        path: 可选，限制 diff 到特定文件路径。

    Returns:
        diff 文本内容。

    Show changes (unstaged by default, --staged if staged=True).
    """
    args = ["diff"]
    if staged:
        args.append("--staged")
    if path:
        args.extend(["--", path])
    output = await _run_git(workspace, args)
    # 输出截断保护 / Output truncation
    return output[:8000] + (
        f"\n\n[... diff truncated at 8000 chars ...]"
        if len(output) > 8000 else ""
    )


async def _git_log(workspace: Path, max_count: int = 10,
                   oneline: bool = False) -> str:
    """显示提交历史。

    默认显示最近 10 条，格式为 "hash 日期 作者: 摘要"。
    oneline 模式只显示 hash + 标题。

    Args:
        workspace: Git 仓库工作目录。
        max_count: 最大显示条数，默认 10。
        oneline: True 使用单行格式，False 使用详细格式。

    Show commit history.
    """
    # 防御性检查：max_count 必须为正整数，否则 Git 会报错
    if max_count < 1:
        max_count = 1
    args = ["log", f"-{max_count}"]
    if oneline:
        args.append("--oneline")
    else:
        args.extend(["--pretty=format:%h %ad %an: %s", "--date=short"])
    return await _run_git(workspace, args)


async def _git_branch(workspace: Path) -> str:
    """列出所有分支（本地 + 远程）。

    风险等级：low（只读操作）。

    List branches (local and remote). Risk: low, read-only.
    """
    return await _run_git(workspace, ["branch", "-a"])


async def _git_commit(workspace: Path, message: str,
                      files: list[str] | None = None) -> str:
    """暂存并提交变更。

    这是唯一会修改仓库的 Git 工具。先 git add（指定文件或全部），再 git commit。
    风险等级：high（修改仓库历史，不可轻易撤销）。

    注意：本工具不做 git push，push 是网络操作且不可逆，应由人工执行。

    Args:
        workspace: Git 仓库工作目录。
        message: 提交信息。
        files: 可选，指定要暂存的具体文件列表。None 表示暂存所有变更（git add -A）。

    Returns:
        git add + git commit 的合并输出。

    Stage and commit changes.
    Risk: high — permanently modifies repository history. Does NOT push.
    """
    if files:
        # 只暂存指定文件 / Stage only specified files
        add_result = await _run_git(workspace, ["add"] + files)
    else:
        # 暂存所有变更 / Stage all changes
        add_result = await _run_git(workspace, ["add", "-A"])

    result = add_result + "\n"
    result += await _run_git(workspace, ["commit", "-m", message])
    return result


async def _git_show(workspace: Path, ref: str = "HEAD") -> str:
    """显示某个提交的详细信息。

    默认显示 HEAD（最新提交），可指定任意 git 引用（hash、分支、标签）。
    输出上限 8000 字符。

    Args:
        workspace: Git 仓库工作目录。
        ref: Git 引用（commit hash、分支名、标签等），默认 "HEAD"。

    Returns:
        提交详细信息。

    Show a commit's details. Risk: low, read-only.
    """
    output = await _run_git(workspace, ["show", ref])
    return output[:8000] + (
        f"\n\n[... output truncated at 8000 chars ...]"
        if len(output) > 8000 else ""
    )


def register_git_tools(
    registry,  # ToolRegistry
    workspace: Path,
) -> list[Tool]:
    """创建并注册所有 Git 操作工具。

    工具清单及风险说明：
      - git_status (low)：  查看工作树状态，只读，自动审批。
      - git_diff   (low)：  查看未暂存/已暂存变更，只读，自动审批。
      - git_log    (low)：  查看提交历史，只读，自动审批。
      - git_branch (low)：  列出所有分支，只读，自动审批。
      - git_commit (high)： 暂存并提交变更，修改仓库历史，需审批（不可逆）。
      - git_show   (low)：  查看单个提交详情，只读，自动审批。

    Args:
        registry: ToolRegistry 实例。
        workspace: Git 仓库的工作目录。

    Returns:
        已注册的 Tool 实例列表。

    Create and register all git operation tools.
    """
    tools = [
        Tool(
            name="git_status",
            description="Show the working tree status (short format with branch info).",
            # 工具用途：查看工作树状态
            # 风险等级：low — 只读 / Risk: low — read-only
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda **kw: _git_status(workspace),
            risk_level="low",
        ),
        Tool(
            name="git_diff",
            description="Show changes in the working tree (unstaged by default).",
            # 工具用途：查看文件变更差异
            # 风险等级：low — 只读 / Risk: low — read-only
            parameters={
                "type": "object",
                "properties": {
                    "staged": {
                        "type": "boolean",
                        "description": "Show staged changes instead of unstaged.",
                        "default": False,
                    },
                    "path": {
                        "type": "string",
                        "description": "Limit diff to a specific file path.",
                        "default": "",
                    },
                },
            },
            handler=lambda staged=False, path="", **kw: _git_diff(
                workspace, staged=staged, path=path,
            ),
            risk_level="low",
        ),
        Tool(
            name="git_log",
            description="Show commit history.",
            # 工具用途：查看提交历史
            # 风险等级：low — 只读 / Risk: low — read-only
            parameters={
                "type": "object",
                "properties": {
                    "max_count": {
                        "type": "integer",
                        "description": "Maximum number of commits to show.",
                        "default": 10,
                    },
                    "oneline": {
                        "type": "boolean",
                        "description": "One-line format.",
                        "default": False,
                    },
                },
            },
            handler=lambda max_count=10, oneline=False, **kw: _git_log(
                workspace, max_count=max_count, oneline=oneline,
            ),
            risk_level="low",
        ),
        Tool(
            name="git_branch",
            description="List all branches (local and remote).",
            # 工具用途：列出所有分支
            # 风险等级：low — 只读 / Risk: low — read-only
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda **kw: _git_branch(workspace),
            risk_level="low",
        ),
        Tool(
            name="git_commit",
            description="Stage and commit changes. Use with caution — this modifies the repository.",
            # 工具用途：暂存并提交变更
            # 风险等级：high — 修改仓库，需审批；不做 push / Risk: high — mutates repo; no push
            parameters={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Commit message.",
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific files to stage (default: all changes).",
                    },
                },
                "required": ["message"],
            },
            handler=lambda message, files=None, **kw: _git_commit(
                workspace, message=message, files=files,
            ),
            risk_level="high",
            requires_approval=True,
        ),
        Tool(
            name="git_show",
            description="Show details of a commit.",
            # 工具用途：查看提交详情
            # 风险等级：low — 只读 / Risk: low — read-only
            parameters={
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": "Git reference (commit hash, branch, tag).",
                        "default": "HEAD",
                    },
                },
            },
            handler=lambda ref="HEAD", **kw: _git_show(workspace, ref=ref),
            risk_level="low",
        ),
    ]

    registry.register_many(tools)
    return tools
