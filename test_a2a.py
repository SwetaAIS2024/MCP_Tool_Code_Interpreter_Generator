"""
A2A (Agent-to-Agent) integration test using the official a2a-sdk (a2a-sdk>=0.3.25).

SDK classes used:
  Server : AgentExecutor, RequestContext, EventQueue, TaskUpdater,
           InMemoryTaskStore, DefaultRequestHandler, A2AStarletteApplication
  Client : JsonRpcTransport (over httpx.ASGITransport -- no real HTTP port needed)
  Types  : AgentCard, AgentSkill, AgentCapabilities, Message, MessageSendParams,
           TaskState, TaskStatusUpdateEvent, TextPart

------------------------------------------------------------------------------
MODE 1 -- SINGLE-AGENT  (default)
------------------------------------------------------------------------------

  Test client
      |  MessageSendParams(query, data_path)
      |  JsonRpcTransport + httpx.ASGITransport -> Starlette app (via .build())
      v
  ToolGeneratorAgentExecutor  (Agent A)
      build_graph() LangGraph pipeline
      intent -> spec -> code -> validate -> execute -> promote
      TaskUpdater.complete(promoted_tool_json)
      |
      v
  Test client receives TaskStatusUpdateEvent(state=completed)

------------------------------------------------------------------------------
MODE 2 -- MULTI-AGENT  (--multi flag)
------------------------------------------------------------------------------

  Test client ONLY talks to Agent A.
  Agent A, after completing Phase 1, acts as an A2A client and calls Agent B
  directly via JsonRpcTransport + httpx.ASGITransport in-process (no real port).
  The test client never contacts Agent B.

  Test client
      |  A2A: MessageSendParams(q1, data_path)
      v
  +-----------------------------------------------------------+
  |  Agent A  (ToolGeneratorAgentExecutor, peer_app=agent_b)  |
  |                                                           |
  |  Phase 1: own build_graph() pipeline -> tool_1            |
  |                                                           |
  |  Phase 2: _call_peer()                                    |
  |    JsonRpcTransport + ASGITransport                       |
  |    MessageSendParams(q2, data_path)                       |
  |           |                                               |
  |           v                                               |
  |    +------------------------------------+                 |
  |    |  Agent B  (AgentExecutor)          |                 |
  |    |  own build_graph() pipeline        |                 |
  |    |  -> tool_2                         |                 |
  |    +------------------------------------+                 |
  |                                                           |
  |  composite = {tool_1, peer_agent_b: {tool_2}}            |
  |  TaskUpdater.complete(json.dumps(composite))              |
  +-----------------------------------------------------------+
      |
      v
  Test client receives composite result (tool_1 + tool_2)

------------------------------------------------------------------------------
Run commands
------------------------------------------------------------------------------
    python test_a2a.py --task-only           # verify SDK types only (instant)
    python test_a2a.py                        # single-agent full pipeline run
    python test_a2a.py --stream               # single-agent + verbose events
    python test_a2a.py --missing-data         # test input-required flow
    python test_a2a.py --multi                # two-agent direct A2A peer call
    python test_a2a.py --multi --stream       # two-agent with streaming events
"""


import argparse
import asyncio
import json
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from a2a.client.transports.jsonrpc import JsonRpcTransport
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps.jsonrpc.starlette_app import A2AStarletteApplication
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Message,
    MessageSendParams,
    Part,
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
)
from a2a.utils.message import new_agent_text_message

sys.path.insert(0, str(Path(__file__).parent))

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.pipeline import build_graph
from src.models import ToolGeneratorState


# ============================================================================
# Helpers
# ============================================================================

def _extract_text(message: Message) -> str:
    """Extract plain text from an official a2a Message."""
    if not message or not message.parts:
        return ""
    texts = []
    for p in message.parts:
        if hasattr(p, "root") and isinstance(p.root, TextPart):
            texts.append(p.root.text)
    return " ".join(texts)


def _make_message(text: str, context_id: Optional[str] = None,
                  task_id: Optional[str] = None) -> Message:
    return Message(
        role=Role.user,
        parts=[Part(root=TextPart(text=text))],
        message_id=str(uuid.uuid4()),
        kind="message",
        context_id=context_id,
        task_id=task_id,
    )


def _make_agent_card(name: str, url: str) -> AgentCard:
    return AgentCard(
        name=name,
        url=url,
        version="1.0.0",
        description="Autonomous MCP tool generator (LangGraph + DeepSeek/Qwen)",
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[
            AgentSkill(
                id="generate-mcp-tool",
                name="Generate MCP Tool",
                description=(
                    "Generates and promotes a validated Python MCP tool "
                    "from a natural language analysis query."
                ),
                tags=["code-generation", "mcp", "langgraph"],
            )
        ],
    )


def _decompose_query(query: str) -> List[str]:
    """Split compound query on AND / ALSO / THEN / semicolons."""
    parts = re.split(r"\s+(?:AND|ALSO|THEN)\s+|;\s*", query, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if len(parts) > 1 else [query, query]  # pad to 2 if undivided


# ============================================================================
# Official AgentExecutor -- wraps the LangGraph build_graph() pipeline
# ============================================================================

class ToolGeneratorAgentExecutor(AgentExecutor):
    """
    Implements the a2a-sdk AgentExecutor ABC.

    execute() runs the full Tool Generator pipeline (intent->spec->code->
    validate->execute->promote) for the incoming query.

    Multi-agent mode: when peer_app is set, after completing its own pipeline
    (Phase 1), the executor acts as an A2A CLIENT and calls the peer Agent B
    directly via JsonRpcTransport + httpx.ASGITransport (Phase 2, in-process,
    no real network).  Both tools are merged into the composite result before
    calling TaskUpdater.complete().
    """

    def __init__(
        self,
        peer_app: Optional[A2AStarletteApplication] = None,
        peer_query: str = "",
        peer_url: str = "http://agent-b",
    ):
        # Each instance owns its own compiled graph and checkpointer
        self.graph = build_graph(checkpointer=MemorySaver(), visualize=False)
        self._peer_app = peer_app
        self._peer_query = peer_query
        self._peer_url = peer_url

    # ------------------------------------------------------------------
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(
            event_queue,
            context.task_id or str(uuid.uuid4()),
            context.context_id or str(uuid.uuid4()),
        )
        await updater.start_work()

        query = context.get_user_input()
        data_path = (context.metadata or {}).get("data_path", "")

        # Validate data_path; trigger input-required if absent or missing
        if not data_path or not Path(data_path).exists():
            await updater.requires_input(
                new_agent_text_message(
                    "Please provide a valid path to the CSV dataset file.",
                    context_id=context.context_id,
                    task_id=context.task_id,
                )
            )
            return

        # -- Phase 1: own build_graph() pipeline --------------------------
        task_id = context.task_id or str(uuid.uuid4())
        initial_state: ToolGeneratorState = {
            "user_query": query,
            "data_path": data_path,
            "task_id": task_id,
            "extracted_intent": None,
            "has_gap": False,
            "matched_tool": None,
            "tool_spec": None,
            "generated_code": None,
            "draft_path": None,
            "validation_result": None,
            "repair_attempts": 0,
            "execution_output": None,
            "draft_output_path": None,
            "promoted_tool": None,
            "errors": None,
            "projected_tool_transcript": None,
            "projected_artifact_log": None,
            "projected_capability_gap": None,
            "projected_errors": None,
            "projected_warnings": None,
            "projected_final_artifacts": None,
            "messages": [HumanMessage(content=query)],
        }
        config = {"configurable": {"thread_id": task_id}}

        try:
            async for _ in self.graph.astream_events(
                initial_state, config=config, version="v2"
            ):
                pass  # pipeline runs; state is stored in MemorySaver

            final_state = self.graph.get_state(config).values
            promoted = final_state.get("promoted_tool")
            errors = final_state.get("errors") or []

            if not promoted:
                await updater.failed(
                    new_agent_text_message(
                        "; ".join(errors) or "Pipeline completed without promoting a tool.",
                        context_id=context.context_id,
                        task_id=context.task_id,
                    )
                )
                return

            composite: Dict[str, Any] = {
                "query": query,
                "promoted_tool": promoted,
                "artifact_log": final_state.get("projected_artifact_log"),
            }

            # -- Phase 2 (multi-agent): Agent A -> Agent B via A2A ---------
            if self._peer_app and self._peer_query:
                print(
                    f"\n  [Agent A -> Agent B]  A2A peer call  "
                    f"query={self._peer_query[:60]}"
                )
                peer_result = await self._call_peer(data_path)
                composite["peer_agent_b"] = peer_result

            await updater.complete(
                new_agent_text_message(
                    json.dumps(composite),
                    context_id=context.context_id,
                    task_id=context.task_id,
                )
            )

        except Exception as exc:
            await updater.failed(
                new_agent_text_message(
                    str(exc),
                    context_id=context.context_id,
                    task_id=context.task_id,
                )
            )

    # ------------------------------------------------------------------
    async def _call_peer(self, data_path: str) -> Dict[str, Any]:
        """
        Agent A calls Agent B directly via A2A.
        Uses httpx.ASGITransport -> no real TCP; fully in-process.
        self._peer_app is a pre-built Starlette ASGI app (from .build()).
        """
        peer_transport = httpx.ASGITransport(app=self._peer_app)
        peer_result: Dict[str, Any] = {"status": "unknown"}

        async with httpx.AsyncClient(
            transport=peer_transport, base_url=self._peer_url
        ) as http:
            rpc = JsonRpcTransport(http, url=self._peer_url)
            params = MessageSendParams(
                message=_make_message(self._peer_query),
                metadata={"data_path": data_path},
            )

            async for ev in rpc.send_message_streaming(params):
                if isinstance(ev, TaskStatusUpdateEvent):
                    state = ev.status.state if ev.status else None
                    if state:
                        peer_result["status"] = state.value
                    if ev.final:
                        if ev.status and ev.status.message:
                            raw = _extract_text(ev.status.message)
                            peer_result["raw"] = raw
                            try:
                                peer_result["data"] = json.loads(raw)
                            except json.JSONDecodeError:
                                pass
                        break
                elif isinstance(ev, Task):
                    state = ev.status.state if ev.status else None
                    if state in (
                        TaskState.completed,
                        TaskState.failed,
                        TaskState.input_required,
                    ):
                        peer_result["status"] = state.value
                        peer_result["task_id"] = ev.id
                        if ev.status and ev.status.message:
                            raw = _extract_text(ev.status.message)
                            peer_result["raw"] = raw
                            try:
                                peer_result["data"] = json.loads(raw)
                            except json.JSONDecodeError:
                                pass
                        break

        return peer_result

    # ------------------------------------------------------------------
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass


# ============================================================================
# App factory
# ============================================================================

from starlette.applications import Starlette


def build_agent_app(
    name: str,
    url: str = "http://testserver",
    peer_app: Optional[Starlette] = None,
    peer_query: str = "",
    peer_url: str = "http://agent-b",
) -> Starlette:
    """
    Build an A2A-compliant ASGI app (Starlette) wrapping the LangGraph pipeline.

    A2AStarletteApplication is a route-builder, not an ASGI callable.
    Calling .build() returns the real Starlette app that httpx.ASGITransport needs.
    """
    executor = ToolGeneratorAgentExecutor(
        peer_app=peer_app, peer_query=peer_query, peer_url=peer_url
    )
    store = InMemoryTaskStore()
    handler = DefaultRequestHandler(agent_executor=executor, task_store=store)
    a2a_app = A2AStarletteApplication(
        agent_card=_make_agent_card(name, url),
        http_handler=handler,
    )
    return a2a_app.build()


# ============================================================================
# Test-client helper -- calls an A2A app in-process via httpx.ASGITransport
# ============================================================================

async def run_agent(
    app: Starlette,
    query: str,
    data_path: str,
    *,
    stream: bool = False,
    label: str = "Agent",
    base_url: str = "http://testserver",
    resume_data_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Send a task to an A2A agent app without a real HTTP server.
    'app' must be the Starlette object returned by build_agent_app() (.build() result).
    Returns a dict with at minimum 'status' and optionally 'data' (parsed JSON).
    """
    transport = httpx.ASGITransport(app=app)

    print(f"\n{'='*70}")
    print(f"[{label}]  Sending task via official A2A")
    print(f"{'='*70}")
    print(f"  query     : {query[:70]}")
    print(f"  data_path : {data_path}")

    async with httpx.AsyncClient(transport=transport, base_url=base_url) as http:
        rpc = JsonRpcTransport(http, url=base_url)
        params = MessageSendParams(
            message=_make_message(query),
            metadata={"data_path": data_path},
        )

        result: Dict[str, Any] = {"status": "unknown"}
        input_required = False

        async for ev in rpc.send_message_streaming(params):
            ts = datetime.now().strftime("%H:%M:%S")

            if isinstance(ev, TaskStatusUpdateEvent):
                state = ev.status.state if ev.status else None
                label_s = state.value if state else "?"

                # Always print key transitions; node-level only with --stream
                if state in (
                    TaskState.working,
                    TaskState.completed,
                    TaskState.failed,
                    TaskState.input_required,
                ) or stream:
                    print(f"  [{ts}] status -> {label_s}")

                if state == TaskState.input_required:
                    input_required = True
                    msg_txt = (
                        _extract_text(ev.status.message)
                        if (ev.status and ev.status.message)
                        else ""
                    )
                    print(f"  [{ts}] INPUT-REQUIRED: {msg_txt}")

                if ev.final:
                    result["status"] = label_s
                    if ev.status and ev.status.message:
                        raw = _extract_text(ev.status.message)
                        result["raw"] = raw
                        try:
                            result["data"] = json.loads(raw)
                        except json.JSONDecodeError:
                            pass
                    break

            elif isinstance(ev, Task):
                state = ev.status.state if ev.status else None
                if state in (
                    TaskState.completed,
                    TaskState.failed,
                    TaskState.input_required,
                ):
                    result["status"] = state.value
                    result["task_id"] = ev.id
                    if ev.status and ev.status.message:
                        raw = _extract_text(ev.status.message)
                        result["raw"] = raw
                        try:
                            result["data"] = json.loads(raw)
                        except json.JSONDecodeError:
                            pass
                    if state == TaskState.input_required:
                        input_required = True
                    else:
                        break

        # Pretty-print terminal state
        if result["status"] == TaskState.completed.value:
            data = result.get("data") or {}
            tool = (data.get("promoted_tool") or {}).get("name", "--")
            print(f"  COMPLETED   tool={tool}")
            if "peer_agent_b" in data:
                pb = data["peer_agent_b"]
                pb_tool = ((pb.get("data") or {}).get("promoted_tool") or {}).get("name", "--")
                print(f"  Agent B     tool={pb_tool}  status={pb.get('status')}")
        elif result["status"] == TaskState.failed.value:
            print(f"  FAILED  {result.get('raw', '')[:80]}")

        print(f"{'='*70}")

    # Handle input-required: retry with resume_data_path
    if input_required and resume_data_path:
        print(f"\n[{label}]  Handling input-required -- retrying data_path={resume_data_path}")
        return await run_agent(
            app, query, resume_data_path,
            stream=stream, label=label, base_url=base_url,
        )

    return result


# ============================================================================
# Model / SDK type verification (no pipeline run)
# ============================================================================

def verify_a2a_models() -> None:
    """Verify that official a2a-sdk types import and construct correctly."""
    print("\n[A2A SDK MODEL CHECK]")

    card = _make_agent_card("test-agent", "http://localhost:9001")
    assert card.name == "test-agent"
    assert card.capabilities.streaming is True
    assert len(card.skills) == 1
    assert card.skills[0].id == "generate-mcp-tool"
    print("  [PASS] AgentCard + AgentSkill + AgentCapabilities")

    msg = _make_message("count crashes by day")
    assert _extract_text(msg) == "count crashes by day"
    print("  [PASS] Message + TextPart + _extract_text()")

    params = MessageSendParams(message=msg, metadata={"data_path": "test.csv"})
    assert params.metadata["data_path"] == "test.csv"
    print("  [PASS] MessageSendParams with metadata")

    states = [s.value for s in TaskState]
    for expected in ("submitted", "working", "input-required", "completed", "failed"):
        assert expected in states, f"Missing TaskState: {expected}"
    print(f"  [PASS] TaskState enum ({len(states)} states)")

    qs = _decompose_query("do X AND do Y")
    assert qs == ["do X", "do Y"], f"unexpected decomposition: {qs}"
    print("  [PASS] _decompose_query() AND-splitting")

    print()


# ============================================================================
# Entry point
# ============================================================================

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="A2A integration test (official a2a-sdk) for tool-generator"
    )
    parser.add_argument(
        "--query",
        default="count the number of crashes for each day of the week",
        help="Query for single-agent mode",
    )
    parser.add_argument(
        "--compound-query",
        default=(
            "count crashes by day of week "
            "AND find average injuries grouped by weather condition"
        ),
        help="Compound query for --multi mode (split on AND/ALSO/THEN)",
    )
    parser.add_argument(
        "--data",
        default="reference_files/sample_planner_output/traffic_accidents.csv",
        help="Path to the CSV dataset",
    )
    parser.add_argument("--stream", action="store_true",
                        help="Print all intermediate status events")
    parser.add_argument("--task-only", action="store_true",
                        help="Verify SDK models only -- no pipeline run")
    parser.add_argument("--missing-data", action="store_true",
                        help="Trigger input-required flow (omit data_path)")
    parser.add_argument(
        "--multi", action="store_true",
        help=(
            "Two-agent mode: Agent A calls Agent B directly via A2A "
            "(JsonRpcTransport + httpx.ASGITransport)"
        ),
    )
    args = parser.parse_args()

    verify_a2a_models()

    if args.task_only:
        print("[A2A] --task-only: skipping pipeline run.")
        return

    if args.multi:
        # ------------------------------------------------------------------
        # MULTI-AGENT MODE
        # Agent A (PeerAwareAgentExecutor) calls Agent B inside _call_peer()
        # using JsonRpcTransport + httpx.ASGITransport.
        # The test client only sends to Agent A.
        # ------------------------------------------------------------------
        sub_queries = _decompose_query(args.compound_query)
        q1, q2 = sub_queries[0], sub_queries[1]

        print("\n[MODE] Multi-Agent A2A -- Agent A -> Agent B (direct peer call)")
        print(f"  Agent A query : {q1}")
        print(f"  Agent B query : {q2}  (Agent A calls Agent B internally via A2A)")

        # Agent B -- standalone Tool Generator
        agent_b_app = build_agent_app(
            "agent-b-tool-generator",
            url="http://agent-b",
        )

        # Agent A -- knows its peer (Agent B)
        agent_a_app = build_agent_app(
            "agent-a-tool-generator",
            url="http://agent-a",
            peer_app=agent_b_app,
            peer_query=q2,
            peer_url="http://agent-b",
        )

        # Test client only talks to Agent A
        result = await run_agent(
            agent_a_app,
            q1,
            args.data,
            stream=args.stream,
            label="Test Client -> Agent A",
            base_url="http://agent-a",
        )

        own_status = result.get("status")
        data = result.get("data") or {}
        peer_info = data.get("peer_agent_b") or {}
        peer_status = peer_info.get("status")

        print(f"\n{'='*70}")
        print("Final composite result (received by test client from Agent A)")
        print(f"{'='*70}")
        own_tool = (data.get("promoted_tool") or {}).get("name", "--")
        peer_tool = ((peer_info.get("data") or {}).get("promoted_tool") or {}).get("name", "--")
        print(f"  Agent A  [{str(own_status):>14}]  {q1[:55]}  ->  {own_tool}")
        print(f"  Agent B  [{str(peer_status):>14}]  {q2[:55]}  ->  {peer_tool}")
        print(f"{'='*70}")

        all_passed = (
            own_status == TaskState.completed.value
            and peer_status == TaskState.completed.value
        )
        if all_passed:
            print("\n[A2A] Multi-agent test PASSED -- both agents generated tools.")
            sys.exit(0)
        else:
            print("\n[A2A] Multi-agent test PARTIAL/FAILED.")
            sys.exit(1)

    else:
        # ------------------------------------------------------------------
        # SINGLE-AGENT MODE
        # ------------------------------------------------------------------
        print("\n[MODE] Single-Agent A2A")
        app = build_agent_app("tool-generator", url="http://testserver")
        data_path = "" if args.missing_data else args.data

        result = await run_agent(
            app,
            args.query,
            data_path,
            stream=args.stream,
            label="Test Client",
            resume_data_path=args.data,
        )

        status = result.get("status")
        if status == TaskState.completed.value:
            print("[A2A] Test PASSED -- tool generated and promoted successfully.")
            sys.exit(0)
        elif status == TaskState.failed.value:
            print(f"[A2A] Test FAILED -- {result.get('raw', '')[:120]}")
            sys.exit(1)
        else:
            print(f"[A2A] Unexpected final status: {status}")
            sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())
