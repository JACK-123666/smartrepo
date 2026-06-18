"""Standardized tool system."""

from smart_repo.tools.base import Tool, ToolResult
from smart_repo.tools.registry import ToolRegistry
from smart_repo.tools.file_tools import register_file_tools
from smart_repo.tools.shell import register_shell_tools
from smart_repo.tools.git_tools import register_git_tools

__all__ = [
    "Tool", "ToolResult", "ToolRegistry",
    "register_file_tools", "register_shell_tools", "register_git_tools",
]
