"""审批管理器——基于风险等级的 Human-in-the-Loop 操作门控。

Approval manager — risk-based operation gating with human-in-the-loop.
Decides whether a pending tool execution can proceed automatically or needs
a human to sign off, based on the tool's risk classification.

============================================================================
设计背景与原理
============================================================================

本模块是 SmartRepo 安全体系中"人机协作"的关键环节。LLM Agent 系统最大的
安全挑战之一是：当代理拥有执行 Shell 命令、修改文件、提交代码等能力时，
如何防止误操作或恶意操作造成不可逆的破坏？答案是在执行前插入人工审批节点——
即 Human-in-the-Loop（人机回路）。

当代理尝试执行中高风险操作时，由本模块根据风险等级和运行模式决定：
  1. 自动放行（低风险）
  2. 弹出确认提示（中风险 + 交互模式）
  3. 直接拒绝（高风险 + 无人值守模式）
  4. 等待显式审批（高风险 + 有审批回调）

============================================================================
三级风险分类及审批规则详解
============================================================================

风险等级由工具的"副作用程度"和"不可逆性"决定。SMARTREPO 将工具分为三级：

┌──────────┬──────────────────────────────────────────────────────────────┐
│ 风险等级  │ 审批规则                                                      │
├──────────┼──────────────────────────────────────────────────────────────┤
│          │ **判定标准**：只读操作、无副作用、不影响系统状态。               │
│          │ **典型工具**：read_file, grep, glob, git_status,              │
│ low      │              git_log, list_directory 等                       │
│ (低风险)  │ **审批规则**：                                                │
│          │   - auto_approve_low=True（默认）→ 自动批准，记录到历史        │
│          │   - auto_approve_low=False → 走正常审批流程                   │
│          │ **设计理由**：只读操作不会造成任何破坏，无需阻塞用户。          │
│          │              自动批准可显著降低审批疲劳（approval fatigue）。   │
├──────────┼──────────────────────────────────────────────────────────────┤
│          │ **判定标准**：有副作用但影响范围有限，或写操作在受限空间内。     │
│          │ **典型工具**：write_file, edit_file, pip_install（工作区内）,  │
│ medium    │              git_branch_create, tool_config_update 等        │
│ (中风险)  │ **审批规则**：                                                │
│          │   - 有回调（交互模式）→ 调用回调，由用户确认                    │
│          │   - 无回调（自动化/无人值守）→ 拒绝执行                        │
│          │ **设计理由**：中风险操作可能有破坏性但可逆。在有人的场景下      │
│          │              确认一下即可；无人时不允许，避免自动脚本误操作。    │
├──────────┼──────────────────────────────────────────────────────────────┤
│          │ **判定标准**：高破坏性、高不可逆性，直接影响系统或代码仓库。     │
│          │ **典型工具**：shell (bash), git_commit, git_push,             │
│ high      │              git_force_push, file_delete, rm 等              │
│ (高风险)  │ **审批规则**：                                                │
│          │   - 始终需要显式人工审批（无论是否有回调）                      │
│          │   - 无回调时直接拒绝（自动化模式不允许高风险操作）              │
│          │ **设计理由**：这些操作可能造成不可逆损害（如 rm -rf、           │
│          │               force push 覆盖远程历史）。必须有人类确认。       │
└──────────┴──────────────────────────────────────────────────────────────┘

============================================================================
决策矩阵（request_approval 完整逻辑）
============================================================================

                    ┌─────────────────────────────────┐
                    │ request_approval(risk_level)     │
                    └─────────────────────────────────┘
                                    │
                    是否为 low + auto_approve_low？
                                    │
                     ┌──────────────┼──────────────┐
                     │ YES                          │ NO
                     ▼                              ▼
              自动批准并记录           创建 ApprovalRequest
              (APPROVED)              (状态: PENDING)
                                              │
                                  是否有 approval_callback？
                                              │
                     ┌─────────────────────────┼─────────────────────────┐
                     │ YES                                              │ NO
                     ▼                                                  ▼
              调用回调获取人工决策                        安全检查：无人值守模式
              callback(req) → decision                       │
              记录历史 → 返回 decision          ┌────────────┼────────────┐
                                              │ low                     │ medium/high
                                              ▼                         ▼
                                        自动批准 (APPROVED)      拒绝 (DENIED)
                                                               中高风险操作必须有
                                                               人类确认

============================================================================
设计理念总结
============================================================================

  - 三级风险自动决策：
    - low：    自动批准（只读操作，无副作用）
    - medium： 交互模式下需确认，非交互模式下拒绝
    - high：   始终需要显式审批（如 Shell 执行、Git 提交）
  - 可插拔的审批回调：支持自定义审批 UI（CLI 交互、Web 审批面板等），
    通过 approval_callback 注入。
  - 完整的审批历史：所有请求（含自动批准和拒绝）都记录在历史列表中，
    支持事后审计和统计（stats() 方法）。
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


class ApprovalDecision(enum.Enum):
    """审批决策枚举 — 表示一次审批请求的最终决策结果。

    Approval decision enum — the outcome of an approval request.

    三种状态的生命周期：
      PENDING  → 请求创建时的初始状态，等待人类或系统决策
      APPROVED → 审批通过，工具可以继续执行
      DENIED   → 审批拒绝，工具执行被阻止（安全检查生效）
    """
    APPROVED = "approved"   # 批准执行 / Approved — tool may proceed
    DENIED = "denied"       # 拒绝执行 / Denied — tool execution blocked
    PENDING = "pending"     # 等待审批 / Pending — awaiting human/system decision


@dataclass
class ApprovalRequest:
    """一次待审批的工具执行请求（数据容器）。

    A pending approval request for a tool execution.
    Encapsulates everything a human reviewer needs to make an informed decision.

    包含审批所需的所有上下文信息：哪个工具、什么参数、风险等级、原因等。
    status 初始为 PENDING，由审批者调用 approve() 或 deny() 后改变。

    这是一个纯数据类（dataclass），不包含业务逻辑。
    业务决策逻辑在 ApprovalManager.request_approval() 中。

    Attributes:
        id:           唯一请求标识符（格式: "approval_0001"）。
                      Unique request ID (format: "approval_0001").
        tool_name:    要执行的工具名称，如 "shell", "write_file"。
                      Name of the tool to be executed.
        tool_args:    工具调用参数，传给审批回调供人类审查。
                      Tool arguments — presented to the human reviewer.
        risk_level:   风险等级: "low" / "medium" / "high"。
                      Risk classification for this operation.
        reason:       请求审批的原因说明（可选）。
                      Optional human-readable reason for the request.
        created_at:   请求创建时间戳（time.time() 返回值）。
                      Unix timestamp when the request was created.
        status:       当前审批状态，初始为 PENDING，通过 approve()/deny() 改变。
                      Current decision status; starts as PENDING.
    """

    id: str
    tool_name: str
    tool_args: dict[str, Any]
    risk_level: str
    reason: str
    created_at: float
    status: ApprovalDecision = ApprovalDecision.PENDING

    def approve(self) -> None:
        """标记此请求为已批准。

        安全检查：此方法不包含权限检查——调用者（ApprovalManager）
        应在调用此方法前已完成风险等级评估和人类确认。
        Mark this request as approved.
        """
        self.status = ApprovalDecision.APPROVED

    def deny(self) -> None:
        """标记此请求为已拒绝。

        安全检查：拒绝意味着工具执行被阻止。此方法终止了该次操作的生命周期。
        Mark this request as denied — execution will be blocked.
        """
        self.status = ApprovalDecision.DENIED


# 异步审批回调的类型签名：接收 ApprovalRequest，返回 ApprovalDecision
# Type for async approval callbacks: takes an ApprovalRequest, returns an ApprovalDecision
ApprovalCallback = Callable[[ApprovalRequest], Awaitable[ApprovalDecision]]


class ApprovalManager:
    """审批工作流管理器——风险操作的 Human-in-the-Loop 门控。

    Manages approval workflow for risky operations.

    职责：
      - 根据风险等级自动决策（低风险自动批准、中高风险需要审批）
      - 支持可插拔的自定义审批回调（CLI 提示、Web 面板等任意 UI）
      - 维护完整的审批请求历史，支持事后审计和统计分析

    核心设计：审批回调（approval_callback）
      - 是一个异步函数: async (ApprovalRequest) -> ApprovalDecision
      - 可注入任意审批 UI：CLI input()、WebSocket 推送、消息队列等
      - 当回调存在时，所有非自动批准路径都通过回调获取人类决策
      - 当回调为 None 时，进入无人值守模式（仅批准 low，拒绝 medium/high）

    风险分类策略：
      - low：    自动批准（只读操作：read_file、grep、git_status 等）
      - medium： 交互模式下需用户确认，非交互模式（无回调）拒绝
      - high：   始终要求显式人工审批（shell、git_commit、push 等）

    使用方式：
        # 无回调模式（自动化/无人值守场景）—— low 自动通过，其他拒绝
        mgr = ApprovalManager(auto_approve_low=True)

        # 有回调模式（交互场景）—— 中等风险弹出确认
        async def my_callback(req: ApprovalRequest) -> ApprovalDecision:
            answer = input(f"Allow {req.tool_name}? [y/N] ")
            return ApprovalDecision.APPROVED if answer.lower() == "y"
            else ApprovalDecision.DENIED

        mgr = ApprovalManager(approval_callback=my_callback)

        decision = await mgr.request_approval("shell", {"cmd": "ls"}, "high")
        if decision == ApprovalDecision.APPROVED:
            await execute_tool(...)  # 安全：通过审批后再执行
        else:
            log.warning("操作被审批管理器拒绝")  # 安全检查生效

        # 审计和统计
        print(mgr.stats())  # {'approved': 15, 'denied': 2, 'total': 17}
        for req in mgr.get_history(10):
            print(f"{req.id}: {req.tool_name} → {req.status.value}")

    The approval callback is an async function that receives the
    ApprovalRequest and returns an ApprovalDecision.
    """

    def __init__(
        self,
        auto_approve_low: bool = True,
        auto_approve_in_workspace: bool = True,
        approval_callback: ApprovalCallback | None = None,
    ) -> None:
        """初始化审批管理器。

        Args:
            auto_approve_low: 是否自动批准低风险操作（默认 True）。
                              设为 False 可对低风险操作也执行审批回调。
            auto_approve_in_workspace: 是否自动批准工作区内的操作（默认 True）。
                                       预留参数，用于工作区隔离的审批策略。
            approval_callback: 自定义异步审批回调函数（可选）。
                               传入 None 表示无人值守模式（仅自动批准 low）。
        """
        # 是否自动批准低风险操作
        # Whether to auto-approve low-risk tools (read-only, no side effects)
        self.auto_approve_low = auto_approve_low
        # 是否自动批准工作区内的操作（预留扩展）
        # Whether to auto-approve operations confined to the workspace
        self.auto_approve_in_workspace = auto_approve_in_workspace
        # 自定义审批回调（异步函数），None 表示无人值守自动化模式
        # Custom async approval callback; None → unattended automated mode
        self.approval_callback = approval_callback
        # 待审批请求表（按 ID 索引，用于快速查找）
        # Pending requests indexed by ID for O(1) lookup
        self._pending: dict[str, ApprovalRequest] = {}
        # 审批历史记录（全量，按时间顺序追加）
        # Full approval history — chronological append-only log for auditing
        self._history: list[ApprovalRequest] = []
        # 请求 ID 自增计数器（4 位零填充，如 approval_0001）
        # Auto-incrementing request ID counter (zero-padded, e.g. approval_0001)
        self._id_counter = 0

    def _next_id(self) -> str:
        """生成唯一且有序的审批请求 ID。

        格式: "approval_0001", "approval_0002", ...
        自增计数器确保 ID 唯一且在审计日志中可排序。
        Generate the next unique approval request ID.
        """
        self._id_counter += 1
        return f"approval_{self._id_counter:04d}"

    async def request_approval(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        risk_level: str = "low",
        reason: str = "",
    ) -> ApprovalDecision:
        """请求工具操作的审批——审批流程的主入口。

        Request approval for a tool operation — main entry point.

        ================================================================
        决策流程（按优先级顺序）
        ================================================================

        优先级 1 — 低风险自动批准：
            IF risk_level == "low" AND self.auto_approve_low == True:
                → 创建 ApprovalRequest (status=APPROVED)，记录到历史
                → 直接返回 APPROVED（跳过所有人工交互）

        优先级 2 — 自定义回调审批（交互模式）：
            IF self.approval_callback is not None:
                → 创建 ApprovalRequest (status=PENDING)
                → 调用 await self.approval_callback(req)
                → 将回调返回的决策写入 req.status，记录到历史
                → 返回回调的决策结果

        优先级 3 — 无人值守自动化模式（无回调）：
            IF risk_level == "low":
                → 自动批准，记录到历史，返回 APPROVED
            ELSE (medium / high):
                安全检查：中高风险操作在无人值守模式下直接拒绝，
                确保自动化脚本不能私自执行 shell 命令或修改代码。
                → 拒绝，记录到历史，返回 DENIED

        ================================================================
        安全设计要点
        ================================================================
        - 低风险 + auto_approve_low 是最常见的路径，避免审批疲劳
        - 中高风险在无人值守时始终拒绝，这是安全底线
        - 审批回调是可插拔的，支持 CLI 确认、Web 面板等任意 UI
        - 所有决策（包括自动通过和拒绝）都记录到 _history，支持审计

        安全检查：中高风险操作在没有回调的自动化模式下会被拒绝，
        确保无人值守的环境中不会执行危险操作（如 rm -rf、git push --force）。

        Args:
            tool_name: 要执行的工具名称（如 "shell", "write_file"）。
            tool_args: 工具调用参数（传递给审批回调供人类审查）。
            risk_level: 风险等级——"low" / "medium" / "high"。
                        默认 "low"，调用者必须为危险操作显式指定更高级别。
            reason: 请求原因说明（可选），展示给审批者帮助其决策。

        Returns:
            ApprovalDecision.APPROVED —— 批准执行
            ApprovalDecision.DENIED   —— 拒绝执行（安全阻止）
        """
        # ============================================================
        # 优先级 1: 低风险自动批准
        # Priority 1: Low-risk auto-approve fast path
        # 安全检查：只读操作无副作用，自动放行以减少审批疲劳。
        # ============================================================
        if risk_level == "low" and self.auto_approve_low:
            req = ApprovalRequest(
                id=self._next_id(),
                tool_name=tool_name,
                tool_args=tool_args,
                risk_level=risk_level,
                reason=reason,
                created_at=time.time(),
                status=ApprovalDecision.APPROVED,
            )
            self._history.append(req)
            return ApprovalDecision.APPROVED

        # ============================================================
        # 创建待审批请求（状态初始为 PENDING）
        # Create pending request for medium/high risk or when auto_approve_low is off
        # ============================================================
        req = ApprovalRequest(
            id=self._next_id(),
            tool_name=tool_name,
            tool_args=tool_args,
            risk_level=risk_level,
            reason=reason,
            created_at=time.time(),
        )

        # ============================================================
        # 优先级 2: 调用自定义审批回调（交互模式）
        # Priority 2: Delegate to custom approval callback (interactive mode)
        # ============================================================
        if self.approval_callback:
            decision = await self.approval_callback(req)
            req.status = decision
            self._history.append(req)
            return decision

        # ============================================================
        # 优先级 3: 无人值守模式——仅批准低风险，拒绝中/高风险
        # Priority 3: Automated mode (no callback)
        # ============================================================
        if risk_level == "low":
            # 安全检查：低风险 + 无回调 → 自动批准（无副作用的只读操作）
            # Low risk without callback → auto-approve (safe, read-only)
            req.approve()
            self._history.append(req)
            return ApprovalDecision.APPROVED

        # 安全检查：中/高风险操作在无人值守模式下被拒绝。
        # 没有人类确认的情况下，shell 执行、文件写入、git 提交等操作
        # 一律拒绝。这是安全底线——自动化脚本不能私自执行危险操作。
        # Security check: medium/high risk is DENIED in unattended mode.
        # Without a human in the loop, dangerous operations (shell exec,
        # file writes, git commits) must not proceed automatically.
        req.deny()
        self._history.append(req)
        return ApprovalDecision.DENIED

    def get_pending(self) -> list[ApprovalRequest]:
        """获取所有待审批的请求列表。

        Get all currently pending (unresolved) approval requests.

        从完整历史记录中筛选出状态仍为 PENDING 的请求。
        这些是已创建但尚未通过审批回调做出决策的请求，
        通常需要外部系统（如 Web 审批面板）定时轮询此方法获取待处理项。

        Returns:
            状态为 PENDING 的审批请求列表（可能为空）。
        """
        return [r for r in self._history
                if r.status == ApprovalDecision.PENDING]

    def get_history(self, limit: int = 20) -> list[ApprovalRequest]:
        """获取最近 N 条审批历史记录（用于审计和安全态势分析）。

        Get the most recent N approval history entries.

        历史记录包含所有已处理的请求（APPROVED 和 DENIED），按时间正序排列。
        默认返回最近 20 条，适用场景：
          - 在 CLI 中展示最近的审批活动摘要
          - 在 Web 面板中渲染审批时间线
          - 审计日志导出（可传较大 limit 或 0 获取全部）

        Args:
            limit: 返回的最大记录数，默认 20。
                   设为 0 或负数时返回全部历史。

        Returns:
            按时间排序的最新 N 条审批请求（最新的在列表末尾）。
        """
        if limit <= 0:
            return list(self._history)
        return self._history[-limit:]

    def stats(self) -> dict[str, int]:
        """返回审批统计信息（用于安全态势评估）。

        Return approval statistics for security posture assessment.

        统计维度：
          - approved: 批准的操作总数
          - denied:   被安全拦截的操作总数
          - total:    所有审批请求总数 (= approved + denied + pending)
          - 通过率:   approved / total（可由调用者自行计算）

        安全检查：denied 计数异常升高可能表明：
          - 代理行为异常（频繁尝试危险操作）
          - 配置错误（工具的风险等级标记不当）
          - 潜在的安全攻击（恶意构造的 tool call）
        建议结合日志监控此统计指标。

        返回值示例:
            {"approved": 42, "denied": 3, "total": 45}
        """
        approved = sum(1 for r in self._history
                       if r.status == ApprovalDecision.APPROVED)
        denied = sum(1 for r in self._history
                     if r.status == ApprovalDecision.DENIED)
        return {"approved": approved, "denied": denied, "total": len(self._history)}
