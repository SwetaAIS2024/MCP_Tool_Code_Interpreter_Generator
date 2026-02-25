"""End-to-end pipeline test with interactive feedback."""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import build_graph
from src.models import ToolGeneratorState
from src.logger_config import PipelineLogger, get_logger, log_section


# Setup logger
logger = get_logger(__name__)


# ============================================================================
# Integration Verification
# ============================================================================

def verify_projection(state: dict) -> bool:
    """Verify that projection_node ran and produced parent-compatible output.

    Checks:
    - All 6 projected_* fields exist in the final state
    - Each non-None field has the correct type expected by AnalysisPipelineState
    - No unexpected keys were introduced at the top level
    - messages field contains BaseMessage instances (or is empty)

    Returns True if all checks pass, False otherwise.
    """
    import json
    from langchain_core.messages import BaseMessage

    PARENT_CHANNEL_MAP = {
        # projected field               expected Python type   parent channel
        "projected_tool_transcript":   (list,  "tool_transcript  List[Dict]"),
        "projected_artifact_log":      (list,  "artifact_log     List[str]"),
        "projected_capability_gap":    (dict,  "capability_gap   Optional[Dict]"),
        "projected_errors":            (list,  "errors           List[str]"),
        "projected_warnings":          (list,  "warnings         List[str]"),
        "projected_final_artifacts":   (dict,  "final_artifacts  Dict[str,Any]"),
    }

    all_ok = True
    print()
    print("=" * 80)
    print("INTEGRATION PROJECTION VERIFICATION")
    print("=" * 80)

    # ---- 1. projected_* field checks ----------------------------------------
    for field, (expected_type, parent_label) in PARENT_CHANNEL_MAP.items():
        value = state.get(field)
        if value is None:
            status = "None (field populated, no data produced)"
            icon = "o"
        elif isinstance(value, expected_type):
            if isinstance(value, list):
                status = f"OK  [{len(value)} items]"
            else:
                status = f"OK  [{len(value)} keys]"
            icon = "v"
        else:
            status = f"TYPE MISMATCH  got {type(value).__name__}, expected {expected_type.__name__}"
            icon = "x"
            all_ok = False
        print(f"  [{icon}] {field:<35}  ->  {parent_label:<40}  {status}")

    # ---- 2. Drill into transcript entries ------------------------------------
    transcript = state.get("projected_tool_transcript") or []
    if transcript:
        print()
        print("  projected_tool_transcript events:")
        for evt in transcript:
            tool = evt.get("tool", "?")
            out = evt.get("output", {})
            preview = ""
            if isinstance(out, dict):
                preview = str(list(out.keys()))[:80]
            print(f"    - tool={tool!r}  output_keys={preview}")

    # ---- 3. artifact_log contents -------------------------------------------
    artifact_log = state.get("projected_artifact_log") or []
    if artifact_log:
        print()
        print("  projected_artifact_log paths:")
        for p in artifact_log:
            exists = Path(p).exists() if p else False
            print(f"    - {'EXISTS' if exists else 'MISSING':7}  {p}")
            if not exists:
                all_ok = False

    # ---- 4. messages field type check ----------------------------------------
    messages = state.get("messages", [])
    if messages:
        bad = [m for m in messages if not isinstance(m, BaseMessage)]
        if bad:
            print(f"\n  [x] messages: {len(bad)} non-BaseMessage entries (type mismatch with parent)")
            all_ok = False
        else:
            print(f"\n  [v] messages: {len(messages)} BaseMessage entries (compatible with parent)")
    else:
        print(f"\n  [o] messages: empty (no entries written during this run)")

    # ---- 5. Stale top-level keys guard (should never appear in parent) -------
    CHILD_ONLY_KEYS = {
        "user_query", "data_path", "extracted_intent", "has_gap",
        "tool_spec", "generated_code", "draft_path",
        "validation_result", "repair_attempts",
        "execution_output", "draft_output_path",
        "promoted_tool", "errors", "messages",
        "projected_tool_transcript", "projected_artifact_log",
        "projected_capability_gap", "projected_errors",
        "projected_warnings", "projected_final_artifacts",
    }
    unexpected = set(state.keys()) - CHILD_ONLY_KEYS
    if unexpected:
        print(f"\n  [x] Unexpected top-level keys in child state: {unexpected}")
        all_ok = False
    else:
        print(f"\n  [v] No unexpected top-level keys in child state")

    # ---- Summary -------------------------------------------------------------
    print()
    if all_ok:
        print("  RESULT: PASS - projection is parent-compatible")
    else:
        print("  RESULT: FAIL - see issues above")
    print("=" * 80)
    print()
    return all_ok


def test_code_gen(verbosity: str = "normal", query: str = None, verify: bool = False):
    """Run the code generation pipeline end-to-end.
    
    Args:
        verbosity: Logging verbosity level (quiet, normal, verbose, debug)
        query: User query for tool generation (optional)
        verify: If True, run integration projection verification at the end
    """
    # Configure logging
    pipeline_logger = PipelineLogger()
    pipeline_logger.setup(verbosity=verbosity)
    
    # Check if test data exists
    test_data = Path("reference_files/sample_planner_output/traffic_accidents.csv")
    
    if not test_data.exists():
        logger.error(f"Test data not found at {test_data}")
        return
    
    # Use provided query or default
    if query is None:
        query = "Run ANOVA across groups, then perform a Tukey HSD post-hoc (multiple-comparisons correction required) and report adjusted p-values and effect sizes."
    
    data_path = str(test_data.resolve())  # Use absolute path for sandbox
    
    log_section(logger, "TESTING CODE GENERATION PIPELINE")
    print(f"Query: {query}")
    print(f"Data: {data_path}")
    print("=" * 80)
    print()
    
    # Build graph
    graph = build_graph()
    
    # Initialize state
    initial_state: ToolGeneratorState = {
        "user_query": query,
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
    
    # Stream execution
    config = {"configurable": {"thread_id": "test-1"}}
    current_state = initial_state
    
    print("\nExecuting pipeline...")
    print("=" * 80)
    
    for event in graph.stream(initial_state, config):
        print(f"\nEvent: {list(event.keys())}")
        
        # Update current state
        for key, value in event.items():
            if isinstance(value, dict):
                current_state.update(value)
    
    # Final results
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED")
    print("=" * 80)
    
    if current_state.get("promoted_tool"):
        print("\n[PROMOTED TOOL]")
        tool = current_state["promoted_tool"]
        print(f"  Name: {tool.get('name')}")
        print(f"  Active Path: {tool.get('path')}")
        if tool.get('output_path'):
            print(f"  Output Path: {tool.get('output_path')}")
        print(f"  Logs Path: {tool.get('logs_path')}")
        print(f"  Registry Path: {tool.get('registry_path')}")
    else:
        print("\nTool was not promoted to registry")
    
    # Show execution output if available
    if current_state.get("execution_output"):
        exec_out = current_state["execution_output"]
        print("\n[EXECUTION OUTPUT]")
        
        # Pretty print the full result with proper formatting
        result = exec_out.get('result')
        if result:
            import json
            try:
                print(f"  Result: {json.dumps(result, indent=2, default=str)}")
            except:
                # Fallback to regular string representation
                print(f"  Result: {str(result)}")
        
        print(f"  Execution Time: {exec_out.get('execution_time_ms', 0):.2f}ms")
        if exec_out.get('error'):
            print(f"  Error: {exec_out.get('error')}")
    
    print("\n" + "=" * 80)

    # Verify integration projection (only when --verify flag is passed)
    if verify:
        verify_projection(current_state)


def test_auto_approve():
    """Test pipeline with automatic execution (no feedback stages)."""
    
    test_data = Path("reference_files/sample_planner_output/traffic_accidents.csv")
    
    if not test_data.exists():
        print(f"Error: Test data not found")
        return
    
    print("=" * 80)
    print("TESTING PIPELINE - DIRECT EXECUTION MODE")
    print("=" * 80)
    
    # Build graph
    graph = build_graph()
    
    # Initialize state
    initial_state: ToolGeneratorState = {
        "user_query": "Show me the top 5 accident types by count",
        "data_path": str(test_data),
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
    
    config = {"configurable": {"thread_id": "auto-test"}}
    
    # Stream execution
    for event in graph.stream(initial_state, config):
        print(f"Event: {list(event.keys())}")
    
    logger.info("\nPipeline completed")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Test MCP tool generation pipeline"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="User query for tool generation (optional, uses default if not provided)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_const",
        const="verbose",
        dest="verbosity",
        help="Enable verbose output (show all details)"
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_const",
        const="debug",
        dest="verbosity",
        help="Enable debug output (show everything including internals)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_const",
        const="quiet",
        dest="verbosity",
        help="Quiet mode (only show warnings and errors)"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-approve all feedback stages (for testing)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run integration projection verification at the end of the pipeline"
    )
    parser.set_defaults(verbosity="normal")
    
    args = parser.parse_args()
    
    if args.auto:
        test_auto_approve()
    else:
        test_code_gen(verbosity=args.verbosity, query=args.query, verify=args.verify)
