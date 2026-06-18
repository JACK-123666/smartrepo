"""18 benchmark tests for SmartRepo capabilities.

Each test is an async function returning bool or dict with 'passed' key.
Tests MUST NOT require real API keys — they test infrastructure, not live models.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path


# ============================================================================
# Test 1: Basic file read/write roundtrip
# ============================================================================
async def test_benchmark_01_file_read_write_roundtrip(workspace: Path) -> dict:
    """Verify the file_tools tools can read and write correctly."""
    from smart_repo.tools.file_tools import _read_file, _write_file, _edit_file

    test_content = "Hello, SmartRepo!\nLine 2\nLine 3\n"

    # Write
    result = await _write_file(workspace, "test_bench_01.txt", test_content)
    assert "Successfully wrote" in result, f"Write failed: {result}"

    # Read full
    content = await _read_file(workspace, "test_bench_01.txt")
    assert "Hello, SmartRepo!" in content, f"Read failed: {content}"
    assert "Line 2" in content
    assert "Line 3" in content

    # Read with offset and limit
    partial = await _read_file(workspace, "test_bench_01.txt", offset=1, limit=1)
    assert "Line 2" in partial and "Line 3" not in partial, f"Partial read failed: {partial}"

    # Edit
    edit_result = await _edit_file(workspace, "test_bench_01.txt", "Line 2", "Modified Line 2")
    assert "Successfully edited" in edit_result, f"Edit failed: {edit_result}"

    # Verify edit
    edited = await _read_file(workspace, "test_bench_01.txt")
    assert "Modified Line 2" in edited

    # Cleanup
    (workspace / "test_bench_01.txt").unlink(missing_ok=True)

    return {"passed": True, "operations": 5}


# ============================================================================
# Test 2: Multi-turn conversation with tool calls
# ============================================================================
async def test_benchmark_02_multi_turn_conversation(workspace: Path) -> dict:
    """Verify the session correctly manages multi-turn conversations."""
    from smart_repo.core.session import Session, SessionConfig, SessionState
    from smart_repo.models.base import Message

    session = Session()
    config = SessionConfig(
        task="Test task",
        model="claude-sonnet-4-6",
        max_turns=5,
    )
    session.start(config)

    # Simulate a multi-turn interaction
    session.add_assistant_message(
        content="Let me read the file.",
        tool_calls=[{
            "id": "call_1",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "test.txt"}'},
        }],
    )
    session.add_tool_result("call_1", "read_file", "File contents here...")
    session.add_assistant_message(content="The file contains what we need.")

    # Verify state
    assert session.turn_count == 2, f"Expected 2 turns, got {session.turn_count}"
    assert len(session.messages) == 5, f"Expected 5 messages (system + user + 2 asst + tool), got {len(session.messages)}"
    assert session.state == SessionState.RUNNING

    # Serialize roundtrip
    data = session.to_dict()
    restored = Session.from_dict(data)
    assert restored.id == session.id
    assert restored.turn_count == session.turn_count
    assert len(restored.messages) == len(session.messages)

    session.set_state(SessionState.COMPLETED)
    assert session.state == SessionState.COMPLETED

    return {"passed": True, "turns": session.turn_count, "messages": len(session.messages)}


# ============================================================================
# Test 3: Checkpoint save and resume mid-task
# ============================================================================
async def test_benchmark_03_checkpoint_save_resume(workspace: Path) -> dict:
    """Verify checkpoint save/restore preserves full session state."""
    import tempfile
    from smart_repo.core.session import Session, SessionConfig
    from smart_repo.core.checkpoint import CheckpointManager

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.db"
        mgr = CheckpointManager(db_path, max_checkpoints=10)

        # Create a session with some history
        session = Session()
        session.start(SessionConfig(task="Test checkpoint task", max_turns=10))
        session.add_assistant_message(content="Step 1: reading file...")
        session.add_tool_result("call_1", "read_file", "content line 1\ncontent line 2")
        session.add_assistant_message(content="Step 2: analyzing...")

        turns_before = session.turn_count

        # Save checkpoint
        ckpt_id = mgr.save(session, summary="Mid-task checkpoint")
        assert ckpt_id, "No checkpoint ID returned"

        # Verify checkpoint was saved
        checkpoints = mgr.list_checkpoints(session.id)
        assert len(checkpoints) == 1, f"Expected 1 checkpoint, got {len(checkpoints)}"

        # Restore
        restored = mgr.restore(session.id)
        assert restored is not None, "Restore returned None"
        assert restored.id == session.id
        assert restored.turn_count == turns_before
        assert len(restored.messages) == len(session.messages)
        assert restored.config.task == "Test checkpoint task"

        # Verify message content preserved
        last_msg = restored.get_last_message()
        assert last_msg is not None
        assert "Step 2" in str(last_msg.content)

        mgr.close()

    return {"passed": True, "checkpoints": len(checkpoints)}


# ============================================================================
# Test 4: Context pruning preserves critical info
# ============================================================================
async def test_benchmark_04_context_pruning_preserves_info(workspace: Path) -> dict:
    """Verify context pruner compresses while preserving key information."""
    from smart_repo.context.token_counter import TokenCounter
    from smart_repo.context.pruner import ContextPruner
    from smart_repo.models.base import Message

    counter = TokenCounter()
    pruner = ContextPruner(token_counter=counter, target_compression=0.35)

    # Build a long conversation
    messages = [Message.system("You are a helpful coding assistant. Always be thorough.")]
    for i in range(40):
        messages.append(Message.user(f"Question {i}: " + "explain in detail " * 20))
        messages.append(Message.assistant(f"Answer {i}: " + "the answer is " * 30))
        if i % 5 == 0:
            messages.append(Message.tool(
                content="file contents: " + "data " * 100,
                tool_call_id=f"call_{i}",
                name="read_file",
            ))

    original_tokens = counter.count(messages, "gpt-4o")

    # Prune to 50% of original tokens
    target = original_tokens // 2
    result = pruner.prune(messages, max_tokens=target, model="gpt-4o")

    assert result.pruned_tokens < original_tokens, "No compression achieved"
    assert result.pruned_tokens <= target, f"Still over budget: {result.pruned_tokens} > {target}"

    compression = result.compression_ratio
    # Verify system message preserved (first message)
    assert result.messages[0].role == "system", "System prompt was removed!"

    return {
        "passed": True,
        "original_tokens": original_tokens,
        "pruned_tokens": result.pruned_tokens,
        "compression_ratio": round(compression * 100, 1),
        "actions": result.actions_taken,
    }


# ============================================================================
# Test 5: Multi-model switching
# ============================================================================
async def test_benchmark_05_multi_model_switching(workspace: Path) -> dict:
    """Verify model registry can switch between Claude and OpenAI providers."""
    from smart_repo.models.registry import ModelRegistry
    from smart_repo.models.claude import ClaudeProvider
    from smart_repo.models.openai import OpenAIProvider

    registry = ModelRegistry()

    # Test Claude models
    claude_provider = registry.get_provider_class("claude-sonnet-4-6")
    assert claude_provider is ClaudeProvider, f"Expected ClaudeProvider, got {claude_provider}"

    claude2 = registry.get_provider_class("claude-opus-4-8")
    assert claude2 is ClaudeProvider

    # Test OpenAI models
    openai_provider = registry.get_provider_class("gpt-4o")
    assert openai_provider is OpenAIProvider, f"Expected OpenAIProvider, got {openai_provider}"

    openai2 = registry.get_provider_class("gpt-4o-mini")
    assert openai2 is OpenAIProvider

    # Test heuristic fallback
    unknown = registry.get_provider_class("claude-unknown-future-model")
    assert unknown is ClaudeProvider, f"Heuristic should match Claude, got {unknown}"

    unknown_oai = registry.get_provider_class("gpt-future-model")
    assert unknown_oai is OpenAIProvider

    # List all models
    all_models = registry.list_models()
    assert len(all_models) >= 15, f"Expected at least 15 models, got {len(all_models)}"

    return {"passed": True, "models_count": len(all_models)}


# ============================================================================
# Test 6: Security — path traversal blocked
# ============================================================================
async def test_benchmark_06_path_traversal_blocked(workspace: Path) -> dict:
    """Verify the security sandbox blocks path traversal attacks."""
    from smart_repo.security.sandbox import SecuritySandbox

    sandbox = SecuritySandbox(
        workspace_dir=workspace,
        allowed_directories=[workspace],
    )

    # Normal path should be allowed
    assert sandbox.is_path_allowed("test.txt") is True
    assert sandbox.is_path_allowed("subdir/test.txt") is True

    # Path traversal should be blocked
    assert sandbox.is_path_allowed("../../../etc/passwd") is False, \
        "Path traversal not blocked!"

    # Absolute path outside workspace
    assert sandbox.is_path_allowed("/etc/passwd") is False, \
        "Absolute path outside workspace not blocked!"

    # validate_path should raise on traversal
    try:
        sandbox.validate_path("../../../etc/passwd")
        assert False, "validate_path should have raised PermissionError"
    except PermissionError:
        pass  # Expected

    # validate_path should succeed for valid paths
    valid = sandbox.validate_path("test.txt")
    expected = (workspace / "test.txt").resolve()
    assert valid == expected, f"Expected {expected}, got {valid}"

    return {"passed": True, "blocked_attacks": 2}


# ============================================================================
# Test 7: Security — API key detection & masking
# ============================================================================
async def test_benchmark_07_api_key_detection_masking(workspace: Path) -> dict:
    """Verify sensitive data sanitizer detects and masks secrets."""
    from smart_repo.security.secret_sanitizer import SensitiveDataSanitizer

    sanitizer = SensitiveDataSanitizer

    # Test: Anthropic API key
    text_with_anthropic = "My key is sk-ant-api03-abc123def456ghi789jkl012mno345pqr678stu901vwx234"
    sanitized = sanitizer.sanitize(text_with_anthropic)
    assert "sk-ant-***" in sanitized, f"Anthropic key not masked: {sanitized}"
    assert "sk-ant-api03" not in sanitized, "Original key still present!"

    # Test: OpenAI API key
    text_with_openai = "OPENAI_API_KEY=sk-proj-abc123def456ghi789jkl012mno345pqr678stu"
    sanitized2 = sanitizer.sanitize(text_with_openai)
    assert "sk-***" in sanitized2 or "sk-proj" not in sanitized2, f"OpenAI key not masked: {sanitized2}"

    # Test: AWS key
    text_with_aws = "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"
    sanitized3 = sanitizer.sanitize(text_with_aws)
    assert "AKIA***" in sanitized3, f"AWS key not masked: {sanitized3}"

    # Test: JWT token
    text_with_jwt = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8s"
    sanitized4 = sanitizer.sanitize(text_with_jwt)
    assert "***.***.***" in sanitized4, f"JWT not masked: {sanitized4}"

    # Test: detection mode
    findings = sanitizer.detect(text_with_anthropic)
    assert len(findings) >= 1, f"Should detect at least 1 secret, got {len(findings)}"

    # Test: has_secrets quick check
    assert sanitizer.has_secrets(text_with_anthropic) is True
    assert sanitizer.has_secrets("normal text without secrets") is False

    return {"passed": True, "detected_types": len(findings)}


# ============================================================================
# Test 8: Memory — file cache prevents re-read
# ============================================================================
async def test_benchmark_08_file_cache_prevents_reread(workspace: Path) -> dict:
    """Verify file memory cache works correctly."""
    from smart_repo.memory.store import MemoryStore
    from smart_repo.memory.file_memory import FileMemory

    # Create a test file
    test_file = workspace / "test_cache.txt"
    test_file.write_text("Cache test content\n" * 50)

    store = MemoryStore(workspace / ".benchmark_cache.json")
    file_memory = FileMemory(store, ttl_seconds=60, workspace=workspace)

    # First read: should not be in cache
    cached = file_memory.get("test_cache.txt")
    assert cached is None, "Should be cache miss on first access"

    # Put content in cache
    entry = file_memory.put(
        file_path="test_cache.txt",
        content=test_file.read_text(),
        summary="test_cache.txt: 50 lines, 950 bytes",
    )
    assert entry is not None

    # Second read: should hit cache
    cached = file_memory.get("test_cache.txt")
    assert cached is not None, "Should be cache hit"
    assert cached.path == "test_cache.txt"
    assert cached.line_count == 50

    # Check cache stats
    stats = file_memory.cache_stats()
    assert stats["cached_files"] == 1

    # Invalidate and check miss
    file_memory.invalidate("test_cache.txt")
    cached = file_memory.get("test_cache.txt")
    assert cached is None, "Should be cache miss after invalidation"

    # Cleanup
    test_file.unlink(missing_ok=True)
    store.clear()
    store.flush()
    (workspace / ".benchmark_cache.json").unlink(missing_ok=True)

    return {"passed": True, "cache_hit_verified": True}


# ============================================================================
# Test 9: Task memory tracks progress across sessions
# ============================================================================
async def test_benchmark_09_task_memory_progress_tracking(workspace: Path) -> dict:
    """Verify task memory correctly tracks goals and progress."""
    from smart_repo.memory.store import MemoryStore
    from smart_repo.memory.task_memory import TaskMemory

    store = MemoryStore(workspace / ".benchmark_tasks.json")
    tm = TaskMemory(store)

    # Add root task
    root = tm.add_task("task_1", "Analyze the codebase for bugs")
    assert root.status == "pending"

    # Add subtasks
    sub1 = tm.add_task("task_1a", "Check for SQL injection", parent_id="task_1")
    sub2 = tm.add_task("task_1b", "Check for XSS vulnerabilities", parent_id="task_1")

    # Update status
    tm.update_status("task_1", "in_progress")
    tm.update_status("task_1a", "completed")
    tm.add_note("task_1a", "Found 3 potential SQL injection points")

    # Check progress
    progress = tm.progress_summary()
    assert progress["total"] == 3
    assert progress["completed"] == 1
    assert progress["in_progress"] == 1
    assert progress["pending"] == 1

    # Check subtask retrieval
    subtasks = tm.get_subtasks("task_1")
    assert len(subtasks) == 2

    # Verify task note
    task_1a = tm.get_task("task_1a")
    assert task_1a is not None
    assert "SQL injection" in task_1a.notes[0]

    # Simulate persistence across "sessions" — create a new TaskMemory from same store
    tm2 = TaskMemory(store)
    restored = tm2.get_task("task_1")
    assert restored is not None
    assert restored.status == "in_progress"

    # Cleanup
    store.clear()
    store.flush()
    (workspace / ".benchmark_tasks.json").unlink(missing_ok=True)

    return {"passed": True, "progress": progress}


# ============================================================================
# Test 10: Shell command sandbox (blocked commands)
# ============================================================================
async def test_benchmark_10_shell_command_sandbox(workspace: Path) -> dict:
    """Verify shell tool blocks dangerous commands."""
    from smart_repo.tools.shell import _shell_exec

    blocked = ["rm -rf /", "shutdown", "> /dev/sda", "mkfs.ext4 /dev/sda"]

    for cmd in blocked:
        result = await _shell_exec(
            workspace,
            command=cmd,
            blocked_patterns=["rm -rf /", "shutdown", "> /dev/sda", "mkfs."],
        )
        assert "BLOCKED" in result, f"Command '{cmd}' should be blocked, got: {result[:200]}"

    # Test that safe commands work
    result = await _shell_exec(workspace, command="echo hello world", timeout=5000)
    assert "hello world" in result, f"Safe command failed: {result[:200]}"

    # Test timeout
    # (skip on CI — may cause hangs)
    # result = await _shell_exec(workspace, command="sleep 10", timeout=100)
    # assert "TIMEOUT" in result

    return {"passed": True, "blocked_commands": len(blocked)}


# ============================================================================
# Test 11: Approval flow for high-risk operations
# ============================================================================
async def test_benchmark_11_approval_flow_high_risk(workspace: Path) -> dict:
    """Verify the approval manager correctly gates operations by risk level."""
    from smart_repo.security.approval import (
        ApprovalManager, ApprovalDecision, ApprovalRequest,
    )

    # Test: auto-approve low risk
    mgr = ApprovalManager(auto_approve_low=True)
    decision = await mgr.request_approval(
        "read_file", {"path": "test.txt"}, risk_level="low",
    )
    assert decision == ApprovalDecision.APPROVED, f"Low risk should be auto-approved, got {decision}"

    # Test: medium risk without callback = denied
    decision = await mgr.request_approval(
        "write_file", {"path": "test.txt", "content": "test"},
        risk_level="medium", reason="Writing file",
    )
    assert decision == ApprovalDecision.DENIED, f"Medium risk without callback should be denied, got {decision}"

    # Test: high risk without callback = denied
    decision = await mgr.request_approval(
        "shell", {"command": "rm -rf ."}, risk_level="high",
        reason="Dangerous shell command",
    )
    assert decision == ApprovalDecision.DENIED, f"High risk should be denied, got {decision}"

    # Test: with callback that approves
    async def approve_all(req: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.APPROVED

    mgr2 = ApprovalManager(
        auto_approve_low=True,
        approval_callback=approve_all,
    )
    decision = await mgr2.request_approval(
        "write_file", {"path": "test.txt"}, risk_level="medium",
    )
    assert decision == ApprovalDecision.APPROVED, "With callback, medium should be approved"

    # Test: with callback that denies
    async def deny_all(req: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.DENIED

    mgr3 = ApprovalManager(auto_approve_low=False, approval_callback=deny_all)
    decision = await mgr3.request_approval(
        "read_file", {"path": "test.txt"}, risk_level="low",
    )
    assert decision == ApprovalDecision.DENIED, "With deny callback, even low should be denied"

    # Check stats
    stats = mgr.stats()
    assert stats["total"] == 3

    return {"passed": True, "stats": stats}


# ============================================================================
# Test 12: End-to-end — agent loop with mocked model
# ============================================================================
async def test_benchmark_12_end_to_end_agent_loop(workspace: Path) -> dict:
    """End-to-end test of the full agent loop with a mock model provider."""
    import tempfile
    from smart_repo.config import Config
    from smart_repo.core.agent import Agent
    from smart_repo.core.checkpoint import CheckpointManager
    from smart_repo.core.session import Session, SessionConfig, SessionState
    from smart_repo.models.base import BaseProvider, Message, ProviderResponse
    from smart_repo.tools.registry import ToolRegistry
    from smart_repo.tools.file_tools import register_file_tools
    from smart_repo.tools.shell import register_shell_tools
    from smart_repo.tools.git_tools import register_git_tools
    from smart_repo.security.approval import ApprovalManager

    # Create mock provider that simulates a simple tool-use interaction
    class MockProvider(BaseProvider):
        def __init__(self, model="mock-model", api_key=""):
            super().__init__(model=model, api_key=api_key)
            self.call_count = 0

        @property
        def model_name(self) -> str:
            return self.model

        async def chat(self, messages, tools=None, temperature=0.7, max_tokens=4096):
            self.call_count += 1
            if self.call_count == 1:
                # First call: request to read a file
                return ProviderResponse(
                    content="Let me read the file first.",
                    tool_calls=[{
                        "id": "mock_call_1",
                        "type": "function",
                        "function": {
                            "name": "list_dir",
                            "arguments": '{"path": "."}',
                        },
                    }],
                    finish_reason="tool_calls",
                    usage={"input_tokens": 100, "output_tokens": 50},
                )
            elif self.call_count == 2:
                # Second call: final response
                return ProviderResponse(
                    content="I have analyzed the directory. Task complete.",
                    finish_reason="stop",
                    usage={"input_tokens": 200, "output_tokens": 30},
                )
            return ProviderResponse(content="Done.", finish_reason="stop")

        def count_tokens(self, messages):
            return sum(len(str(m.content)) // 4 for m in messages)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        checkpoint_mgr = None
        try:
            config = Config(
                workspace_dir=workspace,
                checkpoint_dir=tmp_path / "checkpoints",
                memory_dir=tmp_path / "memory",
            )
            config.ensure_dirs()

            provider = MockProvider()
            tool_registry = ToolRegistry()
            register_file_tools(tool_registry, workspace)
            register_shell_tools(tool_registry, workspace)
            register_git_tools(tool_registry, workspace)

            checkpoint_mgr = CheckpointManager(
                tmp_path / "checks.db", max_checkpoints=5,
            )
            approval_mgr = ApprovalManager(auto_approve_low=True)

            agent = Agent(
                config=config,
                provider=provider,
                tool_registry=tool_registry,
                checkpoint_manager=checkpoint_mgr,
                approval_manager=approval_mgr,
            )

            session = Session()
            session.start(SessionConfig(
                task="List the directory and report back.",
                model="mock-model",
                max_turns=5,
            ))

            # Run the agent
            session = await agent.run(session)

            # Verify
            assert session.state == SessionState.COMPLETED, \
                f"Expected COMPLETED, got {session.state}"
            assert provider.call_count >= 2, \
                f"Expected at least 2 model calls, got {provider.call_count}"
            assert session.turn_count >= 1, \
                f"Expected at least 1 turn, got {session.turn_count}"

            # Verify messages include tool interactions
            roles = [m.role for m in session.messages]
            assert "tool" in roles, f"Expected tool messages in conversation, roles: {roles}"
            assert "assistant" in roles
        finally:
            if checkpoint_mgr:
                checkpoint_mgr.close()

    return {
        "passed": True,
        "turns": session.turn_count,
        "messages": len(session.messages),
        "model_calls": provider.call_count,
    }


# ============================================================================
# Additional conformance tests
# ============================================================================

async def test_benchmark_13_tool_registry_operations(workspace: Path) -> dict:
    """Verify tool registry registration, lookup, and schema generation."""
    from smart_repo.tools.base import Tool
    from smart_repo.tools.registry import ToolRegistry

    registry = ToolRegistry()

    tool = Tool(
        name="test_tool",
        description="A test tool",
        parameters={
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
        },
        handler=lambda **kw: "test_output",  # Not awaitable in sync context. We'll test registration only.
        risk_level="low",
    )

    # Need to make the handler async for execute to work
    async def async_handler(**kw):
        return "test_output"

    tool.handler = async_handler

    registry.register(tool)
    assert "test_tool" in registry
    assert len(registry) == 1

    # Get schemas
    schemas = registry.get_schemas()
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "test_tool"

    # Execute
    result = await registry.execute("test_tool", input="hello")
    assert result.success
    assert result.output == "test_output"

    # Unknown tool
    result = await registry.execute("nonexistent")
    assert not result.success
    assert "Unknown tool" in result.error

    # Unregister
    registry.unregister("test_tool")
    assert "test_tool" not in registry

    return {"passed": True}


# ============================================================================
# Test 14: Context pruning preserves tool_use/tool_result pairing
# ============================================================================
async def test_benchmark_14_pruning_preserves_tool_pairing(workspace: Path) -> dict:
    """裁剪后 tool_use 与 tool_result 必须保持一一配对，否则 Claude/OpenAI API 400。"""
    from smart_repo.context.token_counter import TokenCounter
    from smart_repo.context.pruner import ContextPruner
    from smart_repo.models.base import Message

    counter = TokenCounter()
    pruner = ContextPruner(token_counter=counter, target_compression=0.35)

    # 构造 30 轮带工具调用的对话：每轮 assistant(tool_use) + tool(result)
    messages = [Message.system("You are a coding agent.")]
    for i in range(30):
        messages.append(Message.user(f"Question {i}: " + "x " * 30))
        messages.append(Message.assistant(content="", tool_calls=[{
            "id": f"call_{i}", "type": "function",
            "function": {"name": "read_file", "arguments": '{"path":"f.py"}'},
        }]))
        messages.append(Message.tool(content="content " + "y " * 40,
                                     tool_call_id=f"call_{i}", name="read_file"))

    original = counter.count(messages, "gpt-4o")
    # 强制触发多层裁剪（预算压到 1/4）
    result = pruner.prune(messages, max_tokens=original // 4, model="gpt-4o")

    # 校验配对完整性：每个 tool_result 必须有对应 tool_use，反之亦然
    use_ids = {tc["id"] for m in result.messages
               if m.role == "assistant" and m.tool_calls
               for tc in m.tool_calls if tc.get("id")}
    tool_ids = {m.tool_call_id for m in result.messages
                if m.role == "tool" and m.tool_call_id}
    orphan_results = tool_ids - use_ids
    orphan_uses = use_ids - tool_ids

    assert not orphan_results, f"orphan tool_results after pruning: {orphan_results}"
    assert not orphan_uses, f"orphan tool_uses after pruning: {orphan_uses}"
    assert result.messages[0].role == "system", "system prompt was removed"

    return {"passed": True, "original_tokens": original,
            "pruned_tokens": result.pruned_tokens,
            "orphan_results": len(orphan_results), "orphan_uses": len(orphan_uses)}


# ============================================================================
# Test 15: Claude provider usage includes total_tokens
# ============================================================================
async def test_benchmark_15_claude_usage_has_total_tokens(workspace: Path) -> dict:
    """Claude provider 的 usage 必须含 total_tokens，否则会话 token 统计恒 0。"""
    from smart_repo.models.claude import ClaudeProvider

    class _FakeUsage:
        input_tokens = 100
        output_tokens = 50
    class _FakeBlock:
        type = "text"
        text = "hi"
    class _FakeResp:
        content = [_FakeBlock()]
        usage = _FakeUsage()
        stop_reason = "end_turn"

    provider = ClaudeProvider(model="claude-sonnet-4-6", api_key="fake")
    response = provider._parse_response(_FakeResp())

    assert response.usage.get("total_tokens") == 150, \
        f"total_tokens missing or wrong: {response.usage}"

    return {"passed": True, "total_tokens": response.usage["total_tokens"]}


# ============================================================================
# Test 16: Failed tool call produces exactly one tool_result (no duplicates)
# ============================================================================
async def test_benchmark_16_no_duplicate_tool_result(workspace: Path) -> dict:
    """失败的工具调用只能产生一条 tool_result；重复会破坏配对致后续 API 400。"""
    import tempfile
    from smart_repo.config import Config
    from smart_repo.core.agent import Agent
    from smart_repo.core.checkpoint import CheckpointManager
    from smart_repo.core.session import Session, SessionConfig
    from smart_repo.models.base import BaseProvider, ProviderResponse
    from smart_repo.tools.registry import ToolRegistry
    from smart_repo.tools.file_tools import register_file_tools
    from smart_repo.security.approval import ApprovalManager

    class FailProvider(BaseProvider):
        """第一轮返回 unknown tool 调用（会失败），第二轮 stop。"""
        def __init__(self):
            super().__init__(model="mock", api_key="")
            self.n = 0
        @property
        def model_name(self):
            return self.model
        async def chat(self, messages, tools=None, temperature=0.7, max_tokens=4096):
            self.n += 1
            if self.n == 1:
                return ProviderResponse(
                    content="", tool_calls=[{
                        "id": "call_1", "type": "function",
                        "function": {"name": "nonexistent_tool", "arguments": "{}"},
                    }],
                    finish_reason="tool_calls",
                )
            return ProviderResponse(content="done", finish_reason="stop")
        def count_tokens(self, messages):
            return 1

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        config = Config(workspace_dir=workspace,
                        checkpoint_dir=tmp_path / "c",
                        memory_dir=tmp_path / "m")
        config.ensure_dirs()
        registry = ToolRegistry()
        register_file_tools(registry, workspace)
        cm = CheckpointManager(tmp_path / "ck.db", max_checkpoints=5)
        try:
            agent = Agent(config=config, provider=FailProvider(),
                          tool_registry=registry, checkpoint_manager=cm,
                          approval_manager=ApprovalManager(auto_approve_low=True))
            session = Session()
            session.start(SessionConfig(task="test", model="mock", max_turns=5))
            session = await agent.run(session)

            count = sum(1 for m in session.messages
                        if m.role == "tool" and m.tool_call_id == "call_1")
            assert count == 1, \
                f"expected 1 tool_result for call_1, got {count} (duplicate bug)"
        finally:
            cm.close()

    return {"passed": True, "tool_result_count": count,
            "final_state": session.state.value}


# ============================================================================
# Test 17: Resume repairs missing tool_result after a mid-turn interrupt
# ============================================================================
async def test_benchmark_17_resume_repairs_tool_result(workspace: Path) -> dict:
    """中断在工具回合中途时，resume 必须补齐缺失的 tool_result，否则 API 400。"""
    import tempfile
    from smart_repo.config import Config
    from smart_repo.core.agent import Agent
    from smart_repo.core.checkpoint import CheckpointManager
    from smart_repo.core.session import Session, SessionConfig, SessionState
    from smart_repo.models.base import BaseProvider, ProviderResponse
    from smart_repo.tools.registry import ToolRegistry
    from smart_repo.tools.file_tools import register_file_tools
    from smart_repo.security.approval import ApprovalManager

    class StopProvider(BaseProvider):
        """直接返回 stop，让 _repair_trailing_tool_pairs 的效果可被观测。"""
        def __init__(self):
            super().__init__(model="mock", api_key="")
        @property
        def model_name(self):
            return self.model
        async def chat(self, messages, tools=None, temperature=0.7, max_tokens=4096):
            return ProviderResponse(content="resumed and done", finish_reason="stop")
        def count_tokens(self, messages):
            return 1

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        config = Config(workspace_dir=workspace,
                        checkpoint_dir=tmp_path / "c",
                        memory_dir=tmp_path / "m")
        config.ensure_dirs()
        registry = ToolRegistry()
        register_file_tools(registry, workspace)
        cm = CheckpointManager(tmp_path / "ck.db", max_checkpoints=5)
        try:
            agent = Agent(config=config, provider=StopProvider(),
                          tool_registry=registry, checkpoint_manager=cm,
                          approval_manager=ApprovalManager(auto_approve_low=True))
            # 模拟中断在工具回合中途：assistant 已含 tool_use，但无 tool_result
            session = Session()
            session.start(SessionConfig(task="test", model="mock", max_turns=5))
            session.add_assistant_message(content="let me read", tool_calls=[{
                "id": "call_X", "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"f.py"}'},
            }])
            session.set_state(SessionState.INTERRUPTED)
            session = await agent.run(session)  # 应触发 _repair_trailing_tool_pairs

            repaired = any(m.role == "tool" and m.tool_call_id == "call_X"
                           for m in session.messages)
            assert repaired, "resume did not repair missing tool_result for call_X"
        finally:
            cm.close()

    return {"passed": True, "repaired": repaired,
            "final_state": session.state.value}


# ============================================================================
# Test 18: DeepSeek provider registration & endpoint
# ============================================================================
async def test_benchmark_18_deepseek_provider(workspace: Path) -> dict:
    """DeepSeek provider 注册正确、启发式匹配、client 指向 DeepSeek 端点。"""
    from smart_repo.models.registry import ModelRegistry
    from smart_repo.models.deepseek import DeepSeekProvider, DEEPSEEK_BASE_URL

    registry = ModelRegistry()
    # 注册表精确匹配
    assert registry.get_provider_class("deepseek-chat") is DeepSeekProvider
    assert registry.get_provider_class("deepseek-reasoner") is DeepSeekProvider
    # 启发式前缀匹配：未注册的 deepseek-* 也走 DeepSeekProvider
    assert registry.get_provider_class("deepseek-future-model") is DeepSeekProvider

    # registry.create 能实例化 DeepSeekProvider
    provider = registry.create("deepseek-chat", api_key="fake-key")
    assert isinstance(provider, DeepSeekProvider)

    # client 指向 DeepSeek 端点（fake-key 不会触发请求，仅创建客户端对象）
    assert DEEPSEEK_BASE_URL in str(provider.client.base_url)
    # context limit
    assert provider.context_limit == 64_000

    return {"passed": True, "base_url": str(provider.client.base_url)}
