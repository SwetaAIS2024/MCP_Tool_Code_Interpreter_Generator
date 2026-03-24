"""Pipeline orchestrator module for LangGraph workflow.

This module assembles the complete tool generation pipeline:
1. Intent extraction
2. Spec generation
3. Code generation
4. Validation (with repair loop)
5. Execution
6. Two-stage feedback
7. Promotion to registry
"""

from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage

from .models import ToolGeneratorState
from .intent_extraction import intent_node, route_after_intent
from .spec_generator import spec_generator_node
from .code_generator import code_generator_node, repair_node
from .logger_config import get_logger

logger = get_logger(__name__)
from .validator import validator_node, route_after_validation
from .executor import executor_node, route_after_execution
from .promoter import promoter_node


# ============================================================================
# Projection Node
# ============================================================================

def projection_node(state: ToolGeneratorState) -> ToolGeneratorState:
    """Terminal node: package child outputs into parent-safe projected_* fields.

    Reads the completed child state and writes the 6 projected_* fields so the
    parent-graph owner can copy them 1-to-1 into AnalysisPipelineState channels
    (tool_transcript, artifact_log, capability_gap, errors, warnings,
    final_artifacts) without triggering extra='forbid' violations.
    No existing node return values are modified.
    """
    transcript: List[Dict[str, Any]] = []
    artifact_log: List[str] = []
    errors: List[str] = []
    warnings: List[str] = []

    # ---- Intent / gap -------------------------------------------------------
    extracted_intent = state.get("extracted_intent")
    has_gap = state.get("has_gap", False)

    if extracted_intent:
        transcript.append({
            "tool": "intent_extractor",
            "args": {"query": state.get("user_query", "")},
            "output": extracted_intent,
        })

    capability_gap: Optional[Dict[str, Any]] = None
    if has_gap:
        capability_gap = {
            "component": "tool_generator",
            "extracted_intent": extracted_intent,
            "reason": "capability gap detected — new tool was generated",
        }

    # ---- Generation artifacts -----------------------------------------------
    draft_path = state.get("draft_path")
    if draft_path:
        artifact_log.append(draft_path)

    tool_spec = state.get("tool_spec")
    if tool_spec:
        spec_dict = (
            tool_spec.model_dump()
            if hasattr(tool_spec, "model_dump")
            else dict(tool_spec)
        )
        transcript.append({
            "tool": "spec_generator",
            "args": {},
            "output": {
                "tool_name": spec_dict.get("tool_name"),
                "description": spec_dict.get("description"),
                "version": spec_dict.get("version"),
            },
        })

    # ---- Validation / repair ------------------------------------------------
    validation_result = state.get("validation_result")
    repair_attempts = state.get("repair_attempts", 0)
    if validation_result:
        vr_errors = getattr(validation_result, "errors", []) or []
        vr_warnings = getattr(validation_result, "warnings", []) or []
        if not getattr(validation_result, "is_valid", True):
            errors.extend(vr_errors)
        warnings.extend(vr_warnings)
        transcript.append({
            "tool": "validator",
            "args": {},
            "output": {
                "schema_ok": getattr(validation_result, "schema_ok", None),
                "tests_ok": getattr(validation_result, "tests_ok", None),
                "sandbox_ok": getattr(validation_result, "sandbox_ok", None),
                "repair_attempts": repair_attempts,
            },
        })

    # Merge any errors already accumulated in child state (e.g. from spec_generator_node)
    for e in (state.get("errors") or []):
        if e not in errors:
            errors.append(e)

    # ---- Execution ----------------------------------------------------------
    draft_output_path = state.get("draft_output_path")
    if draft_output_path:
        artifact_log.append(draft_output_path)

    execution_output = state.get("execution_output")
    if execution_output:
        # Keep transcript entry bounded — strip heavy result payload
        safe_output = {k: v for k, v in execution_output.items() if k != "result"}
        result_preview = execution_output.get("result", {})
        if isinstance(result_preview, dict):
            safe_output["result_keys"] = list(result_preview.keys())[:10]
        transcript.append({
            "tool": "executor",
            "args": {},
            "output": safe_output,
        })

    # ---- Promotion ----------------------------------------------------------
    promoted_tool = state.get("promoted_tool")
    final_artifacts: Optional[Dict[str, Any]] = None
    if promoted_tool:
        final_artifacts = {"promoted_tool": promoted_tool}
        if promoted_tool.get("path"):
            artifact_log.append(promoted_tool["path"])
        transcript.append({
            "tool": "promoter",
            "args": {},
            "output": {
                "name": promoted_tool.get("name"),
                "path": promoted_tool.get("path"),
                "status": "promoted",
            },
        })

    # ---- Build Chat response (shown in LangGraph Studio Chat UI) -----------
    if promoted_tool:
        tool_name = promoted_tool.get("name", "unknown")
        tool_path = promoted_tool.get("path", "")

        # Validation badge
        vr = state.get("validation_result")
        repair_count = state.get("repair_attempts", 0)
        if vr:
            schema_ok  = "pass" if getattr(vr, "schema_ok", False)  else "fail"
            tests_ok   = "pass" if getattr(vr, "tests_ok", False)   else "fail"
            sandbox_ok = "pass" if getattr(vr, "sandbox_ok", False) else "fail"
            val_line = f"Schema: {schema_ok} | Tests: {tests_ok} | Sandbox: {sandbox_ok} | Repairs: {repair_count}"
        else:
            val_line = ""

        # Code output — extract from RunArtifacts structure:
        # execution_output = {"result": {"result": {...}, "metadata": {...}, "plot_base64": "..."},
        #                     "summary_markdown": ..., "execution_time_ms": ..., "error": ...}
        exec_out      = state.get("execution_output") or {}
        outer_result  = exec_out.get("result") or {}
        inner_result  = outer_result.get("result", {}) if isinstance(outer_result, dict) else {}
        inner_meta    = outer_result.get("metadata", {}) if isinstance(outer_result, dict) else {}
        plot_b64      = outer_result.get("plot_base64", "") if isinstance(outer_result, dict) else ""
        summary_md    = exec_out.get("summary_markdown") or ""
        exec_time_ms  = exec_out.get("execution_time_ms")
        exec_error    = exec_out.get("error")

        # Plot — save the PNG to disk for persistence.
        # The image will be delivered as a proper image_url content block in AIMessage
        # (multimodal format), which renders inline in LangGraph Studio and any
        # OpenAI-compatible chat UI — both locally and in remote deployments.
        plot_save_note = ""
        if plot_b64:
            try:
                import base64 as _b64, os as _os
                plots_dir = _os.path.join(_os.getcwd(), "output", "plots")
                _os.makedirs(plots_dir, exist_ok=True)
                plot_filename = f"{tool_name}_plot.png"
                plot_path = _os.path.join(plots_dir, plot_filename)
                with open(plot_path, "wb") as _pf:
                    _pf.write(_b64.b64decode(plot_b64))
            except Exception:
                pass

        # Result table — show the actual computed values
        if exec_error:
            result_block = f"```\n{exec_error}\n```"
        elif plot_b64 and not inner_result:
            # Visualisation-only tool: no numeric table to show
            result_block = ""
        elif isinstance(inner_result, dict) and inner_result:
            rows = "\n".join(f"| `{k}` | {v} |" for k, v in list(inner_result.items())[:20])
            result_block = f"| Key | Value |\n|---|---|\n{rows}"
        elif isinstance(inner_result, list) and inner_result:
            # List of dicts (e.g. groupby result) — render as table
            first = inner_result[0] if inner_result else {}
            if isinstance(first, dict):
                headers = list(first.keys())
                header_row = "| " + " | ".join(f"`{h}`" for h in headers) + " |"
                sep_row   = "|" + "|".join("---" for _ in headers) + "|"
                data_rows = "\n".join(
                    "| " + " | ".join(str(row.get(h, "")) for h in headers) + " |"
                    for row in inner_result[:30]
                )
                result_block = f"{header_row}\n{sep_row}\n{data_rows}"
            else:
                result_block = f"```\n{str(inner_result)[:600]}\n```"
        elif inner_result:
            result_block = f"```\n{str(inner_result)[:600]}\n```"
        else:
            result_block = "_no result data_"

        # Execution time line
        if exec_time_ms is not None:
            exec_time_line = f"_Execution time: {exec_time_ms / 1000:.2f}s_"
        else:
            exec_time_line = ""

        # Metadata bullets (from inner metadata, not RunArtifacts metadata)
        if isinstance(inner_meta, dict) and inner_meta:
            meta_block = "\n".join(f"- **{k}:** {v}" for k, v in inner_meta.items())
        else:
            meta_block = ""

        # Generated code — read from active file
        code = state.get("generated_code") or ""
        if not code and tool_path:
            try:
                import os as _os
                abs_p = tool_path if _os.path.isabs(tool_path) else _os.path.join(
                    _os.getcwd(), tool_path.replace("\\", _os.sep)
                )
                with open(abs_p, "r", encoding="utf-8") as _f:
                    code = _f.read()
            except Exception:
                code = ""

        code_block = f"\n---\n**Generated Code**\n```python\n{code}\n```" if code else ""

        parts = [
            f"**Tool:** `{tool_name}`",
            val_line,
            f"",
            f"**Output**",
            result_block,
            exec_time_line,
            meta_block,
            f"\n{summary_md}" if summary_md else "",
            code_block,
        ]
        response_text = "\n".join(p for p in parts if p is not None)

    elif errors:
        response_text = "**Pipeline failed**\n\n" + "\n".join(f"- {e}" for e in errors)
    else:
        response_text = "Pipeline completed but no tool was promoted."

    # Build message content — use multimodal list when a chart is available so
    # the image renders inline (like ChatGPT) in LangGraph Studio and any
    # OpenAI-compatible UI, both locally and in remote deployments.
    if plot_b64:
        ai_content = [
            {"type": "text", "text": response_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{plot_b64}"}},
        ]
    else:
        ai_content = response_text

    return {
        **state,
        "projected_tool_transcript": transcript or None,
        "projected_artifact_log": artifact_log or None,
        "projected_capability_gap": capability_gap,
        "projected_errors": errors or None,
        "projected_warnings": warnings or None,
        "projected_final_artifacts": final_artifacts,
        "messages": [AIMessage(content=ai_content)],
    }


def build_graph(checkpointer: Optional[MemorySaver] = None) -> StateGraph:
    """Build and compile the LangGraph StateGraph.
    
    Args:
        checkpointer: Optional checkpointer for interrupt handling.
                     Pass a MemorySaver explicitly when running standalone.
                     When running under LangGraph API, leave as None — the
                     platform manages persistence automatically.
    
    Returns:
        Compiled graph ready for execution
    """
    # Build graph
    workflow = StateGraph(ToolGeneratorState)
    
    # Add nodes
    workflow.add_node("intent_node", intent_node)
    workflow.add_node("spec_generator_node", spec_generator_node)
    workflow.add_node("code_generator_node", code_generator_node)
    workflow.add_node("validator_node", validator_node)
    workflow.add_node("repair_node", repair_node)
    workflow.add_node("executor_node", executor_node)
    workflow.add_node("promoter_node", promoter_node)
    workflow.add_node("projection_node", projection_node)
    
    # Set entry point
    workflow.set_entry_point("intent_node")
    
    # Add edges
    workflow.add_conditional_edges("intent_node", route_after_intent)
    workflow.add_edge("spec_generator_node", "code_generator_node")
    workflow.add_edge("code_generator_node", "validator_node")
    workflow.add_conditional_edges("validator_node", route_after_validation)
    workflow.add_edge("repair_node", "validator_node")
    workflow.add_conditional_edges("executor_node", route_after_execution)
    workflow.add_edge("promoter_node", "projection_node")
    workflow.add_edge("projection_node", END)
    
    # Compile — checkpointer is None when running under LangGraph API
    graph = workflow.compile(checkpointer=checkpointer)
    
    # Generate graph visualization
    try:
        from pathlib import Path
        
        # Get graph structure
        graph_structure = graph.get_graph()
        
        # Save Mermaid diagram
        mermaid = graph_structure.draw_mermaid()
        mermaid_file = Path("pipeline_graph.mmd")
        mermaid_file.parent.mkdir(parents=True, exist_ok=True)
        mermaid_file.write_text(mermaid)
        
        # Try to generate PNG
        try:
            png_data = graph_structure.draw_mermaid_png()
            png_file = Path("pipeline_graph.png")
            png_file.write_bytes(png_data)
            logger.info(f"[OK] Graph visualization saved to: {png_file}")
        except Exception:
            logger.info(f"[OK] Graph Mermaid diagram saved to: {mermaid_file}")
            logger.info("   (Paste into https://mermaid.live for visualization)")
    except Exception as e:
        logger.warning(f"[WARN] Could not generate graph visualization: {e}")
    
    return graph


def run_pipeline(user_query: str, data_path: str, thread_id: str = None) -> Dict[str, Any]:
    """Execute the complete tool generation pipeline.
    
    Args:
        user_query: Natural language query from user
        data_path: Path to data file to analyze
        thread_id: LangGraph thread identifier for checkpointer isolation.
                   Defaults to a new UUID if not provided.
        
    Returns:
        Final state dict with all pipeline results
    """
    import uuid as _uuid
    if thread_id is None:
        thread_id = str(_uuid.uuid4())

    # Build graph — pass MemorySaver for standalone run (thread_id config support)
    graph = build_graph(checkpointer=MemorySaver())
    
    # Initialize state
    initial_state: ToolGeneratorState = {
        "user_query": user_query,
        "data_path": data_path,
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
        "messages": []
    }
    
    # Run graph — thread_id is required by the MemorySaver checkpointer
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(initial_state, config=config)
    
    return result


# ============================================================================
# Module-level compiled graph — required by LangGraph Studio (langgraph dev)
# ============================================================================
graph = build_graph()