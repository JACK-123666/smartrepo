"""安全沙箱——工作区隔离与操作门控。

本模块是 SmartRepo 安全体系的第一道防线，负责工作区隔离和路径/命令级别的访问控制。

设计理念：
  - 工作区隔离（Workspace Isolation）：所有文件操作被限制在允许的目录集合内，
    通过 resolve() + relative_to() 检测路径穿越攻击。
  - 分层安全策略：路径校验 + 命令黑名单 + 审计日志，纵深防御。
  - 审计追踪：所有安全检查决策都记录到 audit_log，便于事后审查。
  - 防御路径穿越（Directory Traversal）：`../` 等相对路径符号在 resolve() 后
    如果超出了 allowed_dirs 范围，会被拒绝并抛出 PermissionError。

Security sandbox — workspace isolation and operation gating.
Acts as the first line of defence: every file path and shell command passes
through this module before reaching the filesystem or OS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class SecuritySandbox:
    """强制执行工作区隔离和安全策略。

    职责：
      - 限制文件操作在允许的目录范围内（路径校验）
      - 验证路径合法性，防止路径穿越攻击
      - 门控危险操作（命令黑名单）
      - 审计追踪所有操作决策

    使用方式：
        sandbox = SecuritySandbox(
            workspace_dir=Path("/safe/workdir"),
            allowed_directories=[Path("/safe/workdir"), Path("/safe/data")],
            blocked_commands=["rm -rf /", "fork bomb"],
        )

        # 验证文件路径
        safe_path = sandbox.validate_path("subdir/file.txt")

        # 检查命令是否允许
        allowed, reason = sandbox.is_command_allowed("ls -la")

        # 查看审计摘要
        summary = sandbox.get_audit_summary()

    Enforces workspace isolation and security policies.

    Responsibilities:
      - Restrict file operations to allowed directories
      - Validate paths (prevent traversal attacks)
      - Gate dangerous operations
      - Track all operations for audit
    """

    def __init__(
        self,
        workspace_dir: Path,
        allowed_directories: list[Path] | None = None,
        blocked_commands: list[str] | None = None,
    ) -> None:
        # 工作区根目录（解析为绝对路径） / Workspace root (resolved to absolute)
        self.workspace_dir = workspace_dir.resolve()
        # 允许的目录白名单，默认仅包含工作区根目录
        # Allowed directory whitelist; defaults to workspace only
        self.allowed_dirs = [
            d.resolve() for d in (allowed_directories or [workspace_dir])
        ]
        # 命令模式黑名单 / Command pattern blacklist
        self.blocked_commands = blocked_commands or []
        # 审计日志：记录所有安全决策 / Audit log: records every security decision
        self.audit_log: list[dict[str, Any]] = []

    def is_path_allowed(self, path: str | Path) -> bool:
        """检查路径是否在允许的目录范围内。

        安全检查：路径穿越防范——将路径相对工作区解析后，验证其是否
        位于 allowed_dirs 的某个目录内。不在范围内返回 False。

        Args:
            path: 要检查的路径（可以是相对路径字符串或 Path 对象）。

        Returns:
            True 如果路径在允许的目录范围内，False 否则。

        Check if a path is within allowed directories.
        """
        try:
            resolved = (self.workspace_dir / path).resolve()
        except (ValueError, OSError):
            return False

        # 安全检查：逐项验证解析后路径是否在某允许目录内
        # Security check: verify resolved path is within at least one allowed dir
        for allowed in self.allowed_dirs:
            try:
                resolved.relative_to(allowed)
                return True
            except ValueError:
                continue
        return False

    def validate_path(self, path: str | Path) -> Path:
        """验证并解析路径，路径穿越时抛出异常。

        安全检查：严格的路径验证（抛出异常版本）。
        如果解析后的路径逃逸到允许目录之外，抛出 PermissionError。
        这是"先验证，后使用"模式——不安全的路径会在此被拦截，不会到达文件系统。

        Args:
            path: 要验证的路径。

        Returns:
            解析后的安全绝对路径。

        Raises:
            PermissionError: 如果路径逃逸出允许目录范围（路径穿越攻击）。

        Validate and resolve a path.

        Raises PermissionError if the path escapes the allowed directories.
        This is the "validate-then-use" gate — unsafe paths are stopped here.
        """
        try:
            resolved = (self.workspace_dir / path).resolve()
        except (ValueError, OSError) as e:
            raise PermissionError(f"Invalid path: {path} — {e}")

        # 安全检查：逐项验证解析后路径是否在某允许目录内
        # Security check: ensure resolved path stays within an allowed directory
        for allowed in self.allowed_dirs:
            try:
                resolved.relative_to(allowed)
                return resolved
            except ValueError:
                continue

        # 安全检查：路径穿越被阻断！
        # Security check: Path traversal blocked!
        raise PermissionError(
            f"Path traversal blocked: '{path}' resolves to '{resolved}', "
            f"which is outside allowed directories: {self.allowed_dirs}"
        )

    def is_command_allowed(self, command: str) -> tuple[bool, str]:
        """检查 Shell 命令是否被允许执行。

        安全检查：命令黑名单匹配——将命令转为小写后与 blocked_commands 中的
        每个模式做子串匹配。匹配到任何模式则拒绝执行。

        Args:
            command: 要检查的 Shell 命令字符串。

        Returns:
            (是否允许, 拒绝原因) —— 允许时 reason 为空字符串。
            若被阻止，reason 包含匹配到的危险模式信息。

        Check if a shell command is allowed.

        Returns:
            (allowed, reason) — reason is empty if allowed.
        """
        cmd_lower = command.lower()
        # 安全检查：遍历黑名单，子串匹配 / Security check: blacklist substring matching
        for pattern in self.blocked_commands:
            if pattern.lower() in cmd_lower:
                # 审计：记录被阻止的命令 / Audit: log the blocked command
                self._audit("command_blocked", command=command, pattern=pattern)
                return False, f"Command matches blocked pattern: '{pattern}'"

        # 审计：记录被放行的命令 / Audit: log the allowed command
        self._audit("command_allowed", command=command)
        return True, ""

    def _audit(self, action: str, **details: Any) -> None:
        """记录一条审计日志。

        每条日志包含操作类型、UTC 时间戳和详细上下文。
        审计日志用于事后审查安全决策，所有操作（允许/拒绝）都会被记录。

        Args:
            action: 操作类型标签（如 "command_blocked", "command_allowed"）。
            **details: 操作相关的键值对上下文信息。

        Record an audit entry with timestamp and operation details.
        """
        from datetime import datetime, timezone
        self.audit_log.append({
            "action": action,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details,
        })

    def get_audit_summary(self) -> dict[str, int]:
        """获取按操作类型统计的审计摘要。

        返回每种操作（如 "command_blocked", "command_allowed"）的计数。
        用于快速了解安全事件的分布情况。

        Get counts of actions by type for quick security overview.
        """
        counts: dict[str, int] = {}
        for entry in self.audit_log:
            action = entry["action"]
            counts[action] = counts.get(action, 0) + 1
        return counts
