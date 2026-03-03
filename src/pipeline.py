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

    return {
        **state,
        "projected_tool_transcript": transcript or None,
        "projected_artifact_log": artifact_log or None,
        "projected_capability_gap": capability_gap,
        "projected_errors": errors or None,
        "projected_warnings": warnings or None,
        "projected_final_artifacts": final_artifacts,
    }


def build_graph(checkpointer: Optional[MemorySaver] = None) -> StateGraph:
    """Build and compile the LangGraph StateGraph.
    
    Args:
        checkpointer: Optional checkpointer for interrupt handling.
                     If None, creates a MemorySaver for interrupt support.
    
    Returns:
        Compiled graph ready for execution
    """
    # Create checkpointer if not provided (needed for interrupts)
    if checkpointer is None:
        checkpointer = MemorySaver()
    
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
    
    # Compile without interrupts (direct execution flow)
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
            logger.info(f"📊 Graph visualization saved to: {png_file}")
        except Exception:
            logger.info(f"📊 Graph Mermaid diagram saved to: {mermaid_file}")
            logger.info("   (Paste into https://mermaid.live for visualization)")
    except Exception as e:
        logger.warning(f"⚠️  Could not generate graph visualization: {e}")
    
    return graph


def run_pipeline(user_query: str, data_path: str, thread_id: str = None) -> Dict[str, Any]:
    """Execute the complete tool generation pipeline.
    
    Args:
        user_query: Natural language query from user
        data_path: Path to data file to analyze
        thread_id: Unique identifier for this run (auto-generated UUID if not provided).
                   Pass a stable string to resume a checkpointed run.
        
    Returns:
        Final state dict with all pipeline results
    """
    import uuid
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    # Build graph
    graph = build_graph()
    
    # Initialize state
    initial_state: ToolGeneratorState = {
        "user_query": user_query,
        "data_path": data_path,
        "extracted_intent": None,
        "has_gap": False,
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
    
    # Run graph
    result = graph.invoke(initial_state, config={"configurable": {"thread_id": thread_id}})
    
    return result