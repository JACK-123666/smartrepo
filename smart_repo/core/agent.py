"""主 Agent 循环 — 协调模型调用、工具执行和检查点保存。

Main agent loop — orchestrates model calls, tool execution, and checkpointing.

本模块是 SmartRepo 的"大脑"：它实现了 Agent 主循环，负责反复执行以下流程：
  1. 上下文治理（裁剪过长的对话历史）
  2. 调用大模型（Claude/OpenAI）
  3. 解析响应：如果是文本则结束本轮，如果是工具调用则逐个执行
  4. 对工具调用进行参数校验、安全审批、执行、输出脱敏和截断
  5. 定期将会话状态保存为检查点（checkpoint），支持断点续跑

设计原则：
  - 将"模型调用"和"工具执行"解耦为独立方法，便于测试和扩展
  - 所有工具调用都必须经过 validator → approval → execute 三道关卡
  - 上下文治理在每次模型调用前执行，确保 token 不超限
  - 参数强制转换（_coerce_args）用来容忍模型偶尔的类型错误
"""

from __future__ import annotations

import json
import time
from typing import Any

from smart_repo.config import Config
from smart_repo.context.governor import ContextGovernor, GovernedContext
from smart_repo.core.checkpoint import CheckpointManager
from smart_repo.core.session import Session, SessionState
from smart_repo.memory.process_notes import ProcessNotes
from smart_repo.memory.task_memory import TaskMemory
from smart_repo.memory.file_memory import FileMemory
from smart_repo.models.base import BaseProvider, Message, ProviderResponse
from smart_repo.models.registry import ModelRegistry
from smart_repo.security.approval import ApprovalDecision, ApprovalManager
from smart_repo.security.secret_sanitizer import SensitiveDataSanitizer
from smart_repo.tools.base import ToolResult
from smart_repo.tools.registry import ToolRegistry
from smart_repo.security.param_validator import ParameterValidator


class Agent:
    """核心 Agent 循环（Core agent loop）。

    职责：协调以下组件完成"用户任务→模型推理→工具执行→结果返回"的闭环：
      - Model provider (Claude/OpenAI)           —— 大模型提供商
      - Context governance (pruning)             —— 上下文裁剪
      - Tool execution (with security gating)    —— 工具调用（含安全门禁）
      - Checkpoint persistence                   —— 检查点持久化
      - Memory updates                           —— 记忆更新

    用法（Usage）:
        agent = Agent(config, provider, registry, checkpoint_mgr, memory...)
        session = await agent.run(session)
    """

    def __init__(
        self,
        config: Config,
        provider: BaseProvider,
        tool_registry: ToolRegistry,
        checkpoint_manager: CheckpointManager,
        task_memory: TaskMemory | None = None,
        file_memory: FileMemory | None = None,
        process_notes: ProcessNotes | None = None,
        approval_manager: ApprovalManager | None = None,
    ) -> None:
        """初始化 Agent 实例。

        参数（Args）:
            config: 全局配置对象，包含模型、路径、策略等所有设置。
            provider: 大模型提供商实例，封装了 API 调用细节。
            tool_registry: 工具注册表，管理所有可用工具的元信息和执行逻辑。
            checkpoint_manager: 检查点管理器，负责会话状态的保存和恢复。
            task_memory: 可选，任务记忆——用于跨会话记住用户偏好。
            file_memory: 可选，文件记忆——缓存已读取的文件内容以避免重复 I/O。
            approval_manager: 可选，安全审批管理器——控制高风险操作是否需要人工确认。
        """
        self.config = config
        self.provider = provider
        self.tool_registry = tool_registry
        self.checkpoint_mgr = checkpoint_manager
        self.task_memory = task_memory
        self.file_memory = file_memory
        self.process_notes = process_notes
        # 安全审批：默认自动批准低风险操作，workspace 内的操作也可自动批准
        self.approval = approval_manager or ApprovalManager(
            auto_approve_low=True,
            auto_approve_in_workspace=config.auto_approve_in_workspace,
        )

        # 上下文治理器：在每次模型调用前裁剪超长历史
        self.governor = ContextGovernor(config)
        # 参数校验器：确保工具调用的参数符合 schema 定义
        self.validator = ParameterValidator()
        self._should_stop = False

    async def run(self, session: Session) -> Session:
        """执行 Agent 主循环（Execute the agent loop for a session）。

        主循环流程（The loop）:
          第1步（Step 1）: 上下文治理（govern context）——裁剪过长的对话历史，防止 token 超限。
          第2步（Step 2）: 调用模型（call the model）——将消息和工具列表发送给 LLM。
          第3步（Step 3）: 如果模型返回纯文本且 finish_reason=="stop" → 任务完成，结束循环。
          第4步（Step 4）: 如果模型返回工具调用 → 逐个执行：校验参数、审批风险、执行工具、脱敏输出、保存检查点。
          第5步（Step 5）: 循环直到任务完成或达到最大轮次上限。

        参数（Args）:
            session: 要运行的会话（可以是新建的，也可以是从检查点恢复的）。

        返回（Returns）:
            完成后的会话对象，包含所有消息和历史记录。
        """
        # --- 处理初始状态：新会话则启动，中断会话则恢复 ---
        if session.state == SessionState.IDLE and session.config:
            session.start(session.config)
        elif session.state == SessionState.INTERRUPTED:
            # Resume from checkpoint —— 从检查点恢复运行
            session.set_state(SessionState.RUNNING)
            if self.process_notes:
                self.process_notes.add_insight(
                    f"Resumed session {session.id} from checkpoint at turn {session.turn_count}"
                )

        self._should_stop = False

        # 断点恢复一致性修复：会话可能在一次工具调用回合中途被中断（assistant 已
        # 含 tool_calls，但部分 tool_result 尚未写入）。若不补齐，下一轮把历史发给
        # 模型时，未配对的 tool_use 会触发 API 400。这里为所有缺失配对的
        # tool_call_id 补写占位 tool_result，保证 tool_use/tool_result 严格配对。
        self._repair_trailing_tool_pairs(session)

        # 记录 Agent 启动里程碑
        if self.process_notes:
            self.process_notes.add_milestone(
                f"Agent started: {session.config.task[:200] if session.config else 'resumed'}",
            )

        try:
            # ================================================================
            # 主循环：反复执行直到任务完成、出错或达到轮次上限
            # ================================================================
            while not self._should_stop and session.is_active:
                # --- 轮次上限检查（Check turn limit）---
                if session.config and session.turn_count >= session.config.max_turns:
                    if self.process_notes:
                        self.process_notes.add_error(
                            f"Max turns ({session.config.max_turns}) reached"
                        )
                    session.set_error(f"Max turns ({session.config.max_turns}) reached")
                    break

                # --- 第1步（Step 1）：上下文治理 —— 裁剪过长历史防 token 超限 ---
                governed = self._govern_context(session)

                # --- 第2步（Step 2）：调用模型 ---
                response = await self._call_model(governed, session)

                # 模型返回错误 → 终止循环
                if response.finish_reason == "error":
                    session.set_error(response.content)
                    break

                # --- 第3步（Step 3）：处理模型响应 ---
                if response.has_tool_calls:
                    # 情况A：模型要求调用工具 → 记录助手消息，然后逐个执行工具
                    # Record assistant message with tool calls
                    session.add_assistant_message(
                        content=response.content,
                        tool_calls=response.tool_calls,
                        usage=response.usage,
                    )

                    # 逐个执行工具调用（Execute each tool call）
                    for tc in response.tool_calls:
                        executed = await self._execute_tool(tc, session)
                        if not executed:
                            # 工具被拒绝或严重失败 —— _execute_tool 内部已为该 tool_call_id
                            # 追加了一条含具体原因的 tool_result。此处不可再 add_tool_result，
                            # 否则同一 tool_call_id 会出现两条 tool_result，破坏 tool_use/
                            # tool_result 一一配对，导致后续 Claude/OpenAI API 返回 400。
                            if self.process_notes:
                                self.process_notes.add_insight(
                                    f"Tool call "
                                    f"{tc.get('function', {}).get('name', 'unknown')} "
                                    f"denied/failed; error already recorded as tool_result."
                                )
                else:
                    # 情况B：纯文本响应（无工具调用）
                    # Text response — no tool calls this turn
                    session.add_assistant_message(
                        content=response.content,
                        usage=response.usage,
                    )

                    if response.finish_reason == "stop":
                        # 模型正常结束 → 任务完成
                        session.set_state(SessionState.COMPLETED)
                        break
                    elif response.finish_reason in ("length", "max_tokens"):
                        # 输出被 max_tokens 截断：提示模型续写，而非静默重发同一段
                        # 历史（否则会稳定空转、反复截断，烧 token 直到 max_turns）。
                        session.add_user_message(
                            "Your previous response was truncated by max_tokens. "
                            "Please continue concisely from where you left off."
                        )
                        # 不 break，继续循环让模型续写
                    # 其它 finish_reason（如 stop_sequence）同样继续循环

        except Exception as e:
            # 捕获主循环中的任何异常，记录并设置错误状态
            if self.process_notes:
                self.process_notes.add_error(f"Agent loop error: {e}")
            session.set_error(f"{type(e).__name__}: {e}")

        # 记录会话结束摘要
        if self.process_notes:
            self.process_notes.add_summary(
                f"Session {session.id} ended: {session.state.value}, "
                f"{session.turn_count} turns, {session.total_tokens_used} tokens"
            )

        return session

    def stop(self) -> None:
        """请求 Agent 循环优雅停止（Request the agent loop to stop gracefully）。"""
        self._should_stop = True

    def _repair_trailing_tool_pairs(self, session: Session) -> None:
        """补齐会话中缺失配对的 tool_result（resume 一致性修复）。

        扫描消息历史中所有带 tool_calls 的 assistant 消息，对每个 tool_call_id，
        若后续没有对应的 role=tool 消息，则补写一条占位 tool_result。这保证
        tool_use/tool_result 严格一一配对，避免恢复后首次调用 API 时返回 400
        （Claude: "tool_use ids found without tool_result"；OpenAI 同理）。

        对新会话（刚 start，只有 system+user）无害——没有 tool_calls 则不补。
        """
        if not session.messages:
            return

        # 收集所有已配对的 tool_call_id（来自 tool 角色消息）
        answered_ids: set[str] = set()
        for msg in session.messages:
            if msg.role == "tool" and msg.tool_call_id:
                answered_ids.add(msg.tool_call_id)

        # 找出所有 assistant.tool_calls 中未被配对的 tool_call_id
        to_fill: list[tuple[str, str]] = []  # (tool_call_id, tool_name)
        for msg in session.messages:
            if msg.role == "assistant" and msg.tool_calls:
                for tc in msg.tool_calls:
                    tc_id = tc.get("id", "")
                    if tc_id and tc_id not in answered_ids:
                        name = tc.get("function", {}).get("name", "unknown")
                        to_fill.append((tc_id, name))

        for tc_id, name in to_fill:
            session.add_tool_result(
                tool_call_id=tc_id,
                tool_name=name,
                result="[interrupted: tool did not complete before checkpoint]",
            )

    def _govern_context(self, session: Session) -> GovernedContext:
        """在发送给模型之前执行上下文治理。

        上下文治理的作用：当对话历史过长时，裁剪掉最旧的消息，
        确保总 token 数不超过模型的上下文窗口限制。

        Apply context governance before sending to model.
        """
        # 获取所有已注册工具的参数 schema（用于模型了解可用工具）
        # Get all registered tool schemas
        tool_schemas = self.tool_registry.get_schemas()

        # 从 session.messages 中提取系统提示词（第一条消息，role="system"），
        # 传入 governor 以便正确计算 token 预算。
        # Extract system prompt from session messages for accurate token budget calculation.
        system_prompt = ""
        if session.messages and session.messages[0].role == "system":
            sp = session.messages[0].content
            system_prompt = sp if isinstance(sp, str) else str(sp)

        return self.governor.govern(
            system_prompt=system_prompt,
            messages=session.messages[1:] if system_prompt else session.messages,
            tool_schemas=tool_schemas,
            model=session.config.model if session.config else self.config.default_model,
        )

    async def _call_model(
        self,
        governed: GovernedContext,
        session: Session,
    ) -> ProviderResponse:
        """将消息发送给大模型并获取响应（Send messages to the model and get a response）。

        参数（Args）:
            governed: 经上下文治理后的消息和工具列表。
            session: 当前会话，用于获取温度和 token 上限等配置。

        返回（Returns）:
            模型的响应对象，包含文本内容或工具调用请求。
        """
        # 从会话配置中获取温度和响应 token 上限
        temp = session.config.temperature if session.config else 0.7
        max_tok = session.config.max_tokens_per_response if session.config else 4096

        # 获取已注册工具的完整对象列表（而不仅仅是 schema）供 provider 使用
        # Get tool objects for the provider
        tools = [self.tool_registry.get(n) for n in self.tool_registry.list_names()]
        tools = [t for t in tools if t is not None]

        return await self.provider.chat(
            messages=governed.messages,
            tools=tools,
            temperature=temp,
            max_tokens=max_tok,
        )

    async def _execute_tool(
        self,
        tool_call: dict[str, Any],
        session: Session,
    ) -> bool:
        """执行单个工具调用，包含完整的校验、审批和脱敏流程。

        执行流程（Execution flow）:
          第1步：解析工具名称和参数
          第2步：查找工具定义
          第3步：校验参数合法性
          第4步：检查是否需要安全审批
          第5步：实际执行工具
          第6步：对输出进行敏感数据脱敏
          第7步：截断过长输出
          第8步：更新文件记忆缓存（如适用）
          第9步：保存检查点（如达到保存间隔）

        Execute a single tool call with validation and approval.

        参数（Args）:
            tool_call: 模型返回的工具调用描述，包含 id、function name 和 arguments。
            session: 当前会话。

        返回（Returns）:
            True 表示执行成功，False 表示被拒绝或失败。
        """
        # --- 第1步：解析工具名称和 ID ---
        func = tool_call.get("function", tool_call)
        tool_name = func.get("name", "")
        tool_call_id = tool_call.get("id", "")

        # --- 第2步：解析参数（Parse arguments）---
        args_str = func.get("arguments", "{}")
        if isinstance(args_str, dict):
            # 已经是 dict，无需解析
            args = args_str
        else:
            # 尝试从 JSON 字符串解析
            try:
                args = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                session.add_tool_result(
                    tool_call_id, tool_name,
                    f"Error: Invalid JSON arguments: {args_str[:200]}",
                )
                return False

        # --- 第3步：获取工具定义（Get the tool definition）---
        tool = self.tool_registry.get(tool_name)
        if tool is None:
            session.add_tool_result(
                tool_call_id, tool_name,
                f"Error: Unknown tool '{tool_name}'",
            )
            return False

        # --- 第4步：校验参数合法性（Validate parameters）---
        valid, error = self.validator.validate(tool_name, args, tool.parameters)
        if not valid:
            # 参数不合规 → 尝试自动修正常见问题（如类型不匹配）
            # Attempt to fix common issues
            args = self._coerce_args(args, tool.parameters)
            valid, error = self.validator.validate(tool_name, args, tool.parameters)
            if not valid:
                # 修正失败 → 返回错误
                if self.process_notes:
                    self.process_notes.add_error(
                        f"Validation failed for {tool_name}: {error}"
                    )
                session.add_tool_result(tool_call_id, tool_name, f"Validation error: {error}")
                return False

        # --- 第5步：安全检查 —— 是否需要人工审批（Check approval）---
        risk = tool.risk_level
        if tool.requires_approval or risk in ("medium", "high"):
            # 构建审批理由，包含工具名、风险等级和参数快照
            approval_reason = f"Tool '{tool_name}' (risk: {risk}) called with: {json.dumps(args, ensure_ascii=False)[:300]}"
            decision = await self.approval.request_approval(
                tool_name=tool_name,
                tool_args=args,
                risk_level=risk,
                reason=approval_reason,
            )
            if decision == ApprovalDecision.DENIED:
                # 审批被拒绝 → 记录并返回失败
                if self.process_notes:
                    self.process_notes.add_decision(
                        f"Denied execution of {tool_name}", None
                    )
                session.add_tool_result(
                    tool_call_id, tool_name,
                    f"DENIED: Operation '{tool_name}' requires approval and was denied.",
                )
                return False

        # --- 第6步：实际执行工具（Execute the tool）---
        result: ToolResult = await self.tool_registry.execute(
            tool_name, tool_call_id=tool_call_id, **args
        )

        # --- 第7步：对输出进行敏感数据脱敏（Sanitize output）---
        sanitized_output = SensitiveDataSanitizer.sanitize(result.output)
        if sanitized_output != result.output:
            if self.process_notes:
                self.process_notes.add_insight(
                    f"Sanitized sensitive data in output of {tool_name}"
                )
            result.output = sanitized_output

        # --- 第8步：截断过长输出，防止撑爆上下文窗口（Truncate if too long）---
        if len(result.output) > 8000:
            result = result.truncate(8000)

        # --- 第9步：将会话结果记录到 session ---
        # Record in session
        output = result.output
        if result.error:
            output = f"Error: {result.error}\n\n{output}"

        session.add_tool_result(tool_call_id, tool_name, output)

        # --- 第10步：如果工具是 read_file，更新文件记忆缓存 ---
        # Update file memory cache if it was a read_file
        if tool_name == "read_file" and self.file_memory and result.success:
            file_path = args.get("path", "")
            if file_path:
                self.file_memory.put(
                    file_path=file_path,
                    content=result.output,
                )

        # --- 第11步：按间隔保存检查点（Save checkpoint after each tool call）---
        if self.config.checkpoint_interval > 0:
            if session.turn_count % self.config.checkpoint_interval == 0:
                self.checkpoint_mgr.save(
                    session,
                    summary=f"Turn {session.turn_count}: executed {tool_name}",
                )

        return True

    def _coerce_args(
        self, args: dict[str, Any], schema: dict[str, Any],
    ) -> dict[str, Any]:
        """尝试将参数类型强制转换为与 schema 期望一致的类型。

        背景：大模型有时会返回类型不匹配的参数（例如用字符串 "5" 代替整数 5）。
        此方法在参数校验失败后作为"第二道防线"，尝试自动修正这些类型错误。

        Attempt to coerce argument types to match schema expectations.

        参数（Args）:
            args: 原始参数字典。
            schema: 工具的 JSON Schema 定义，包含每个参数的期望类型。

        返回（Returns）:
            类型修正后的参数字典。
        """
        properties = schema.get("properties", {})
        coerced = dict(args)

        for name, prop in properties.items():
            if name not in coerced:
                continue
            expected_type = prop.get("type", "string")
            val = coerced[name]

            # 期望字符串 → 将非字符串值转为字符串
            if expected_type == "string" and not isinstance(val, str):
                coerced[name] = str(val)
            # 期望整数 → 尝试将值转为 int
            elif expected_type == "integer" and not isinstance(val, int):
                try:
                    coerced[name] = int(val)
                except (ValueError, TypeError):
                    pass
            # 期望数字 → 尝试将值转为 float
            elif expected_type == "number" and not isinstance(val, (int, float)):
                try:
                    coerced[name] = float(val)
                except (ValueError, TypeError):
                    pass
            # 期望布尔 → 智能解析布尔值（支持字符串 "true"/"1"/"yes"）
            elif expected_type == "boolean" and not isinstance(val, bool):
                if isinstance(val, str):
                    coerced[name] = val.lower() in ("true", "1", "yes")
                else:
                    coerced[name] = bool(val)

        return coerced
