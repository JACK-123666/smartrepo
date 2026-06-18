"""SmartRepo 命令行入口。

子命令:
  run <任务>              运行一个新任务（核心命令）
  resume <会话ID>         从检查点恢复中断的任务
  list                    列出所有已保存的会话
  checkpoints <会话ID>    列出某会话的全部检查点
  stats                   显示运行统计（工具/缓存/审批）

输出用 Rich 渲染；argparse 是标准库，无额外依赖。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Windows GBK 终端下 Rich 的 spinner/unicode 会编码崩溃，强制 stdout/stderr 用 utf-8
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Confirm

from smart_repo.config import Config
from smart_repo.core.runtime import SmartRepo
from smart_repo.core.session import SessionState
from smart_repo.security.approval import ApprovalDecision, ApprovalRequest

console = Console()


def build_parser():
    """构建 argparse 解析器。"""
    import argparse

    parser = argparse.ArgumentParser(
        prog="smart-repo",
        description="SmartRepo — 本地多模型智能体工具，面向代码仓库的 AI 助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""使用示例:
  smart-repo run "找出所有 TODO 注释"
  smart-repo run -m gpt-4o -p openai "解释认证流程"
  smart-repo run -m deepseek-chat -p deepseek "重构 src/auth.py"
  smart-repo run -i "删除未使用的 import"      # -i 高风险步骤交互确认
  smart-repo list                               # 列出已保存会话
  smart-repo resume abc123def456                # 恢复中断的任务
  smart-repo checkpoints abc123def456           # 查看某会话的检查点
  smart-repo stats                              # 运行统计
        """,
    )

    sub = parser.add_subparsers(dest="command", help="可用子命令")

    # run —— 运行新任务
    run_parser = sub.add_parser("run", help="运行一个新任务")
    run_parser.add_argument("task", help="任务描述（自然语言）")
    run_parser.add_argument("--model", "-m", default=None, help="模型名（如 claude-sonnet-4-6, gpt-4o, deepseek-chat）")
    run_parser.add_argument("--provider", "-p", default=None, help="提供商：claude / openai / deepseek")
    run_parser.add_argument("--workspace", "-w", default=".", help="工作区目录，默认当前目录")
    run_parser.add_argument("--max-turns", "-t", type=int, default=100, help="最大对话轮次，默认 100")
    run_parser.add_argument("--temperature", type=float, default=0.7, help="采样温度 0-2，越高越随机，默认 0.7")
    run_parser.add_argument("--system-prompt", "-s", default="", help="自定义系统提示词（覆盖默认）")
    run_parser.add_argument("--interactive", "-i", action="store_true", help="交互模式：高风险操作需人工审批")

    # resume —— 恢复中断的会话
    resume_parser = sub.add_parser("resume", help="从检查点恢复一个中断的会话")
    resume_parser.add_argument("session_id", help="要恢复的会话 ID")
    resume_parser.add_argument("--checkpoint", "-c", default=None, help="指定恢复的检查点 ID（默认用最新）")
    resume_parser.add_argument("--workspace", "-w", default=".", help="工作区目录，默认当前目录")

    # list —— 列出会话
    sub.add_parser("list", help="列出所有已保存的会话")

    # stats —— 运行统计
    sub.add_parser("stats", help="显示运行统计（工具数、缓存、审批等）")

    # checkpoints —— 查看检查点
    ckpt_parser = sub.add_parser("checkpoints", help="列出指定会话的所有检查点")
    ckpt_parser.add_argument("session_id", help="要查看检查点的会话 ID")

    return parser


async def interactive_approval(req: ApprovalRequest) -> ApprovalDecision:
    """交互审批回调：高风险操作在终端弹确认，返回 APPROVED/DENIED。"""
    console.print()
    risk_color = {"low": "green", "medium": "yellow", "high": "red"}.get(
        req.risk_level, "white"
    )
    console.print(Panel(
        f"[bold]工具:[/bold] {req.tool_name}\n"
        f"[bold]风险等级:[/bold] [{risk_color}]{req.risk_level}[/{risk_color}]\n"
        f"[bold]参数:[/bold]\n{_format_args(req.tool_args)}",
        title="需要人工审批",
        border_style=risk_color,
    ))

    # 低风险默认批准，中高风险默认拒绝
    decision = Confirm.ask(
        f"是否批准执行 [bold]{req.tool_name}[/bold]？",
        default=(req.risk_level == "low"),
    )
    return ApprovalDecision.APPROVED if decision else ApprovalDecision.DENIED


def _format_args(args: dict, max_len: int = 500) -> str:
    """把参数字典格式化成 JSON（超长截断），用于审批面板展示。"""
    import json
    text = json.dumps(args, indent=2, ensure_ascii=False)
    if len(text) > max_len:
        text = text[:max_len] + "\n... (已截断)"
    return text


async def cmd_run(args) -> int:
    """运行新任务：初始化 agent → 跑主循环 → 展示结果与会话摘要。

    返回 0=完成，130=Ctrl+C 中断（检查点已保存，可用 resume 继续）。
    """
    workspace = Path(args.workspace).resolve()
    config = Config(workspace_dir=workspace)

    # with 保证 SQLite 检查点连接在退出时关闭
    with SmartRepo(
        workspace_dir=workspace,
        config=config,
        model=args.model or config.default_model,
        provider=args.provider or config.default_provider,
    ) as sr:
        if args.interactive:
            sr.approval.approval_callback = interactive_approval

        console.print(Panel(
            f"[bold]任务:[/bold] {args.task}\n"
            f"[bold]模型:[/bold] {sr.model} ({sr.provider_name})\n"
            f"[bold]工作区:[/bold] {sr.workspace_dir}",
            title="SmartRepo",
            border_style="blue",
        ))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task_id = progress.add_task("智能体运行中...", total=None)

            try:
                session = await sr.run(
                    task=args.task,
                    system_prompt=args.system_prompt,
                    max_turns=args.max_turns,
                    temperature=args.temperature,
                )
            except KeyboardInterrupt:
                # Ctrl+C：检查点已由 agent 保存，可稍后 resume
                progress.update(task_id, description="已中断 — 正在保存检查点...")
                console.print("\n[yellow]已中断。会话已保存到检查点。[/yellow]")
                if sr._current_session:
                    console.print(f"会话 ID: [cyan]{sr._current_session.id}[/cyan]")
                    console.print(f"恢复命令: smart-repo resume {sr._current_session.id}")
                return 130

        # 展示结果
        console.print()
        if session.state == SessionState.COMPLETED:
            console.print("[green][完成] 任务已成功完成[/green]")
        elif session.state == SessionState.ERROR:
            console.print(f"[red][错误] 错误信息: {session.error_message}[/red]")

        last_msg = session.get_last_message()
        if last_msg and last_msg.role == "assistant" and last_msg.content:
            console.print()
            console.print(Panel(
                Markdown(last_msg.content[:3000]),  # 截断过长回复
                title="回复",
                border_style="green",
            ))

        # 会话摘要
        table = Table(title="会话摘要")
        table.add_column("指标", style="cyan")
        table.add_column("数值")
        table.add_row("会话 ID", session.id)
        table.add_row("状态", session.state.value)
        table.add_row("对话轮次", str(session.turn_count))
        table.add_row("消息总数", str(len(session.messages)))
        table.add_row("Token 消耗", f"{session.total_tokens_used:,}")
        table.add_row("耗时", f"{session.duration_seconds:.1f}s")
        console.print(table)

        return 0


async def cmd_resume(args) -> int:
    """从检查点恢复中断的会话并继续执行。

    返回 0=完成，1=会话不存在，130=再次中断。
    """
    workspace = Path(args.workspace).resolve()
    config = Config(workspace_dir=workspace)

    with SmartRepo(workspace_dir=workspace, config=config) as sr:
        console.print(f"正在恢复会话 [cyan]{args.session_id}[/cyan]...")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task_id = progress.add_task("恢复智能体中...", total=None)

            try:
                session = await sr.resume(
                    args.session_id,
                    checkpoint_id=args.checkpoint,
                )
            except KeyboardInterrupt:
                progress.update(task_id, description="已中断 — 检查点已保存")
                console.print("\n[yellow]已中断。会话已保存。[/yellow]")
                return 130

        if session is None:
            console.print(f"[red]会话 '{args.session_id}' 未找到。[/red]")
            return 1

        console.print(f"[green]会话已恢复并完成。[/green]")
        console.print(f"对话轮次: {session.turn_count}, Token 消耗: {session.total_tokens_used:,}")
        return 0


def cmd_list(args) -> int:
    """列出所有已保存的会话（从检查点库扫描）。"""
    config = Config()
    with SmartRepo(config=config) as sr:
        sessions = sr.list_sessions()
        if not sessions:
            console.print("[yellow]未找到已保存的会话。[/yellow]")
            return 0

        table = Table(title="已保存的会话")
        table.add_column("会话 ID", style="cyan")
        table.add_column("检查点数", style="green")

        for sid in sessions:
            checkpoints = sr.list_checkpoints(sid)
            table.add_row(sid, str(len(checkpoints)))

        console.print(table)
        return 0


def cmd_checkpoints(args) -> int:
    """列出某会话的全部检查点（按序号），便于选恢复点。"""
    config = Config()
    with SmartRepo(config=config) as sr:
        checkpoints = sr.list_checkpoints(args.session_id)
        if not checkpoints:
            console.print(f"[yellow]未找到会话 '{args.session_id}' 的检查点。[/yellow]")
            return 0

        table = Table(title=f"会话 {args.session_id} 的检查点列表")
        table.add_column("ID", style="cyan")
        table.add_column("序号", justify="right")
        table.add_column("状态")
        table.add_column("摘要")

        for ckpt in checkpoints:
            table.add_row(
                ckpt["id"],
                str(ckpt["sequence"]),
                ckpt["state"],
                ckpt.get("summary", "")[:80],  # 摘要截断到 80 字符
            )

        console.print(table)
        return 0


def cmd_stats(args) -> int:
    """显示运行统计：工作区、模型、已注册工具、缓存、审批计数。"""
    config = Config()
    with SmartRepo(config=config) as sr:
        stats = sr.get_stats()

        console.print(Panel(
            f"[bold]工作区:[/bold] {stats['workspace']}\n"
            f"[bold]模型:[/bold] {stats['model']} ({stats['provider']})\n"
            f"[bold]已注册工具:[/bold] {stats['tools_registered']} 个\n"
            f"[bold]文件缓存:[/bold] {stats['file_cache']['cached_files']} 个文件已缓存\n"
            f"[bold]审批统计:[/bold] {stats['approval_stats']['approved']} 批准 / "
            f"{stats['approval_stats']['denied']} 拒绝",
            title="SmartRepo 统计信息",
            border_style="blue",
        ))

        if stats["tools_registered"] > 0:
            console.print("\n[bold]已注册的工具:[/bold]")
            for name in stats["tool_names"]:
                console.print(f"  - {name}")

        return 0


def main():
    """入口：解析参数，按子命令分发（run/resume 走 asyncio，其余同步）。"""
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "run":
            return asyncio.run(cmd_run(args))
        elif args.command == "resume":
            return asyncio.run(cmd_resume(args))
        elif args.command == "list":
            return cmd_list(args)
        elif args.command == "checkpoints":
            return cmd_checkpoints(args)
        elif args.command == "stats":
            return cmd_stats(args)
        else:
            parser.print_help()
            return 0
    except KeyboardInterrupt:
        # Ctrl+C 优雅退出
        console.print("\n[yellow]已中断。[/yellow]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
