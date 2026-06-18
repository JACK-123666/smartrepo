"""Shell 执行工具——在沙箱环境中执行命令。

本模块提供代理在受控环境中执行 Shell 命令的能力。

设计理念：
  - Shell 是最高风险的工具（risk_level="high"），始终需要人工审批。
  - 多层安全防护：
    1. 命令模式黑名单（blocked_patterns）——阻止危险命令如 rm -rf /、fork bomb 等
    2. 工作目录白名单（allowed_dirs）——限制命令只能在指定目录内运行
    3. 环境变量剥离——清除所有 API Key、Token、Password 等敏感变量
    4. 超时强制终止——防止无限运行耗尽资源
    5. 输出截断——限制最大输出 8000 字符
  - 使用 asyncio.create_subprocess_shell 异步执行，不阻塞事件循环。
  - 标准输出和标准错误分开捕获，exit code 始终附加在结果末尾。

Shell execution tool — sandboxed command execution.
The shell tool is the most powerful and dangerous capability the agent has.
Every invocation passes through multiple safety layers before reaching the OS.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from smart_repo.tools.base import Tool


async def _shell_exec(
    workspace: Path,
    command: str,
    timeout: int = 120_000,
    allowed_dirs: list[Path] | None = None,
    blocked_patterns: list[str] | None = None,
) -> str:
    """在沙箱环境中执行 Shell 命令。

    此函数是 Shell 安全体系的核心，在命令到达操作系统之前执行多层检查。

    安全防护顺序：
      1. 命令模式黑名单检查 → 阻止已知危险模式
      2. 工作目录白名单验证 → 确保命令在许可范围内运行
      3. 环境变量清理       → 剥离所有可能的凭据泄露
      4. 子进程执行 + 超时   → 异步执行，超时强杀
      5. 输出截断           → 防止上下文溢出

    Args:
        workspace: 工作目录，命令在此目录下执行。
        command: 要执行的 Shell 命令字符串。
        timeout: 超时时间（毫秒），默认 120 秒。
        allowed_dirs: 允许执行命令的目录白名单（可选）。
        blocked_patterns: 命令模式黑名单列表（可选）。

    Returns:
        命令执行结果字符串（stdout + stderr + exit code），或受阻/超时/错误信息。

    Execute a shell command in a sandboxed environment.

    Safety checks:
    1. Block dangerous command patterns     — 阻止危险命令模式
    2. Restrict to allowed directories      — 限制在允许目录内执行
    3. Enforce timeout                      — 强制超时终止
    4. Strip sensitive env vars             — 剥离敏感环境变量
    """
    # ==================== 安全检查 1：命令模式黑名单 ====================
    # Security check 1: Block dangerous command patterns
    cmd_lower = command.lower()
    for pattern in (blocked_patterns or []):
        if pattern.lower() in cmd_lower:
            return (
                f"BLOCKED: Command matches blocked pattern '{pattern}'.\n"
                f"Command was: {command}\n"
                f"This operation requires explicit approval."
            )

    # ==================== 安全检查 2：目录白名单验证 ====================
    # Security check 2: Validate working directory is within allowed dirs
    if allowed_dirs:
        cwd = workspace.resolve()
        allowed = False
        for ad in allowed_dirs:
            try:
                cwd.relative_to(ad.resolve())
                allowed = True
                break
            except ValueError:
                continue
        if not allowed:
            return f"BLOCKED: Working directory '{cwd}' is outside allowed directories."

    # ==================== 安全检查 3：构建安全环境变量 ====================
    # Security check 3: Build a safe environment — strip all secrets
    # 危险的敏感环境变量名集合 / Set of dangerous/sensitive environment variable names
    safe_env = {}
    dangerous_vars = {
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "CLAUDE_API_KEY",
        "AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN", "GH_TOKEN",
        "DATABASE_URL", "PGPASSWORD", "DOCKER_PASSWORD",
    }
    for k, v in os.environ.items():
        # 跳过包含 KEY、TOKEN、SECRET、PASSWORD 的变量，防止凭据泄露
        # Skip any var whose name hints at secrets/tokens/passwords
        if k.upper() not in dangerous_vars and "SECRET" not in k.upper() \
                and "TOKEN" not in k.upper() and "PASSWORD" not in k.upper() \
                and "KEY" not in k.upper():
            safe_env[k] = v
    # 始终保留 PATH 和 HOME，这是大多数命令正常运行的前提
    # Always preserve PATH and HOME — essential for most commands
    safe_env["PATH"] = os.environ.get("PATH", "/usr/bin:/bin")
    safe_env["HOME"] = os.environ.get("HOME", str(Path.home()))

    # ==================== 安全检查 4-5：异步执行 + 超时 + 输出截断 ====================
    # Security check 4-5: Async execution with timeout + output truncation
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(workspace),
            env=safe_env,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout / 1000,
            )
        except asyncio.TimeoutError:
            # 安全措施：超时后强制终止子进程，防止资源泄露
            # Kill on timeout to prevent resource leaks
            proc.kill()
            await proc.wait()
            return f"TIMEOUT: Command exceeded {timeout}ms limit and was killed.\nCommand: {command}"

        result_parts = []
        if stdout:
            result_parts.append(stdout.decode("utf-8", errors="replace"))
        if stderr:
            result_parts.append(f"[stderr]\n{stderr.decode('utf-8', errors='replace')}")

        output = "\n".join(result_parts).strip()
        exit_info = f"\n\n[exit code: {proc.returncode}]"

        if not output:
            return f"(no output){exit_info}"
        # 输出截断保护：防止超长输出撑爆 LLM 上下文窗口
        # Truncate output to prevent context overflow
        return output[:8000] + (
            f"\n\n[... output truncated at 8000 chars ...]{exit_info}"
            if len(output) > 8000 else exit_info
        )

    except FileNotFoundError:
        return f"Error: Shell not available or command not found. Try using bash-compatible syntax."
    except asyncio.CancelledError:
        # 安全检查：任务被取消时确保子进程被终止，防止僵尸进程泄漏
        # Security: kill subprocess on task cancellation to prevent zombie processes
        try:
            proc.kill()
            await proc.wait()
        except Exception:
            pass
        raise  # 重新抛出 CancelledError，让上层处理取消逻辑
    except Exception as e:
        return f"Error executing command: {type(e).__name__}: {e}"


def register_shell_tools(
    registry,  # ToolRegistry
    workspace: Path,
    allowed_dirs: list[Path] | None = None,
    blocked_patterns: list[str] | None = None,
) -> list[Tool]:
    """创建并注册 Shell 执行工具。

    这是整个工具系统中风险最高的工具。Shell 命令可以执行任意代码，
    因此设计上：
      - 风险等级为 "high"，始终需要审批（requires_approval=True）
      - 支持多层安全配置：允许目录白名单 + 命令模式黑名单
      - 环境变量自动清理，防止凭据泄露
      - 输出长度限制，防止上下文溢出

    Args:
        registry: ToolRegistry 实例。
        workspace: 命令执行的工作目录。
        allowed_dirs: 可选的目录白名单，限制命令只能在指定目录下执行。
        blocked_patterns: 可选的命令模式黑名单（如 "rm -rf /", "fork bomb" 等）。

    Returns:
        已注册的 Tool 实例列表（目前仅一个 Shell 工具）。

    Create and register shell execution tools.
    The shell tool is the highest-risk capability — always gated behind approval.
    """
    tool = Tool(
        name="shell",
        description=(
            "Execute a shell command in a sandboxed environment. "
            "Commands run within the workspace directory. "
            "Dangerous commands (rm -rf /, etc.) are automatically blocked. "
            "All secrets are stripped from the environment. "
            "Timeout: 120 seconds."
        ),
        # 工具用途：在沙箱环境中执行 Shell 命令
        # 风险等级：high — 可执行任意代码，始终需审批；多层沙箱保护
        # Risk: high — arbitrary code execution possible; always requires approval;
        #        multiple sandbox layers mitigate risk
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in milliseconds (default: 120000).",
                    "default": 120_000,
                },
            },
            "required": ["command"],
        },
        handler=lambda command, timeout=120_000, **kw: _shell_exec(
            workspace=workspace,
            command=command,
            timeout=timeout,
            allowed_dirs=allowed_dirs,
            blocked_patterns=blocked_patterns,
        ),
        risk_level="high",
        requires_approval=True,
    )

    registry.register(tool)
    return [tool]
