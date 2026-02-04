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


def test_with_feedback(verbosity: str = "normal"):
    """Test pipeline with user feedback at interrupt points.
    
    Args:
        verbosity: Logging verbosity level (quiet, normal, verbose, debug)
    """
    # Configure logging
    pipeline_logger = PipelineLogger()
    pipeline_logger.setup(verbosity=verbosity)
    
    # Check if test data exists
    test_data = Path("reference_files/sample_planner_output/traffic_accidents.csv")
    
    if not test_data.exists():
        logger.error(f"Test data not found at {test_data}")
        return
    
    # Test query
    query = "Run ANOVA across groups, then perform a Tukey HSD post-hoc (multiple-comparisons correction required) and report adjusted p-values and effect sizes."
    data_path = str(test_data.resolve())  # Use absolute path for sandbox
    
    log_section(logger, "TESTING PIPELINE WITH FEEDBACK")
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
        "validation_result": None,
        "repair_attempts": 0,
        "execution_output": None,
        "stage1_approved": False,
        "stage2_approved": False,
        "promoted_tool": None,
        "messages": []
    }
    
    # Stream execution
    config = {"configurable": {"thread_id": "test-1"}}
    current_state = initial_state
    
    for event in graph.stream(initial_state, config):
        print(f"\n Event: {list(event.keys())}")
        
        # Update current state
        for key, value in event.items():
            if isinstance(value, dict):
                current_state.update(value)
    
    # Check if we hit an interrupt
    snapshot = graph.get_state(config)
    
    while snapshot.next:
        print("\n" + "=" * 80)
        print("PIPELINE PAUSED - AWAITING FEEDBACK")
        print("=" * 80)
        print(f"Next node: {snapshot.next}")
        
        # Show current state
        if "feedback_stage1" in str(snapshot.next):
            print("\nSTAGE 1 APPROVAL - Review Execution Output")
            print("-" * 80)
            
            # Show tool spec
            if current_state.get("tool_spec"):
                spec = current_state["tool_spec"]
                print(f"\n[TOOL SPEC]")
                print(f"  Name: {spec.tool_name}")
                print(f"  Description: {spec.description[:150]}...")
            
            # Show validation result
            if current_state.get("validation_result"):
                val = current_state["validation_result"]
                print(f"\n[VALIDATION]")
                print(f"  Schema OK: {val.schema_ok}")
                print(f"  Tests OK: {val.tests_ok}")
                print(f"  Sandbox OK: {val.sandbox_ok}")
            
            # Show execution output
            if current_state.get("execution_output"):
                exec_out = current_state["execution_output"]
                print(f"\n[EXECUTION OUTPUT]")
                
                # Check for failures (exec_out is now a dict, not RunArtifacts object)
                has_error = bool(exec_out.get("error"))
                empty_result = not exec_out.get("result") or exec_out.get("result") == {}
                
                if has_error:
                    print(f"  EXECUTION FAILED")
                    print(f"  Error: {exec_out.get('error')}")
                elif empty_result:
                    print(f"  EXECUTION RETURNED EMPTY RESULT")
                    print(f"  This likely indicates a problem with the code logic.")
                
                print(f"  Result: {str(exec_out.get('result'))[:200]}")
                print(f"  Execution Time: {exec_out.get('execution_time_ms', 0):.2f}ms")
                
                if has_error or empty_result:
                    print(f"\n  Recommendation: REJECT this output")
            
            # Get user input
            print("\n" + "-" * 80)
            response = input("Approve execution output? (yes/no): ").strip()
            
            # Parse response and update state with both the response and approval flag
            approved = response.strip().lower() in ["yes", "y", "approve", "approved", "accept", "ok", "okay"]
            
            # Update state and verify
            graph.update_state(config, {
                "user_response_stage1": response,
                "stage1_approved": approved
            })
            
            # DEBUG: Verify state was updated
            verify_snapshot = graph.get_state(config)
            print(f"\n[DEBUG] After update_state:")
            print(f"  user_response_stage1: '{verify_snapshot.values.get('user_response_stage1')}'")
            print(f"  stage1_approved: {verify_snapshot.values.get('stage1_approved')}")
            
        elif "feedback_stage2" in str(snapshot.next):
            print("\nSTAGE 2 APPROVAL - Promote to Registry")
            print("-" * 80)
            
            # Show generated code preview
            if current_state.get("generated_code"):
                code = current_state["generated_code"]
                print(f"\n[GENERATED CODE]")
                print(f"  Length: {len(code)} characters")
                print(f"  Preview (first 500 chars):")
                print("  " + "-" * 76)
                for line in code[:500].split('\n'):
                    print(f"  {line}")
                print("  " + "-" * 76)
            
            # Get user input
            print("\n" + "-" * 80)
            response = input("Promote to registry? (yes/no): ").strip()
            
            # Parse response and update state with both the response and approval flag
            approved = response.strip().lower() in ["yes", "y", "approve", "approved", "accept", "ok", "okay"]
            graph.update_state(config, {
                "user_response_stage2": response,
                "stage2_approved": approved
            })
        
        # Continue execution
        print("\nResuming pipeline...")
        for event in graph.stream(None, config, stream_mode="updates"):
            print(f"Event: {list(event.keys())}")
            for key, value in event.items():
                if isinstance(value, dict):
                    current_state.update(value)
                    # Debug: Check approval flags after each event
                    if "stage1_approved" in value:
                        print(f"   → stage1_approved set to: {value['stage1_approved']}")
                    if "stage2_approved" in value:
                        print(f"   → stage2_approved set to: {value['stage2_approved']}")
        
        # Get new snapshot
        snapshot = graph.get_state(config)
        print(f"   → After resume, next nodes: {snapshot.next}")
        print(f"   → Current stage1_approved: {snapshot.values.get('stage1_approved')}")
        print(f"   → Current stage2_approved: {snapshot.values.get('stage2_approved')}")
    
    # Final results
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED")
    print("=" * 80)
    
    if current_state.get("promoted_tool"):
        print("\n[PROMOTED TOOL]")
        tool = current_state["promoted_tool"]
        print(f"  Name: {tool.get('name')}")
        print(f"  Version: {tool.get('version')}")
        print(f"  Active Path: {tool.get('path')}")
        print(f"  Logs Path: {tool.get('logs_path')}")
        print(f"  Registry Path: {tool.get('registry_path')}")
    else:
        print("\n Tool was not promoted to registry")
    
    print("\n" + "=" * 80)


def test_auto_approve():
    """Test pipeline with automatic approval (for CI/CD)."""
    
    test_data = Path("reference_files/sample_planner_output/traffic_accidents.csv")
    
    if not test_data.exists():
        print(f"Error: Test data not found")
        return
    
    print("=" * 80)
    print("TESTING PIPELINE - AUTO APPROVE MODE")
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
        "validation_result": None,
        "repair_attempts": 0,
        "execution_output": None,
        "stage1_approved": False,
        "stage2_approved": False,
        "promoted_tool": None,
        "messages": []
    }
    
    config = {"configurable": {"thread_id": "auto-test"}}
    
    # Stream with auto-approval
    for event in graph.stream(initial_state, config):
        print(f"Event: {list(event.keys())}")
    
    # Auto-approve at each interrupt
    snapshot = graph.get_state(config)
    
    while snapshot.next:
        print(f"\nAuto-approving: {snapshot.next}")
        
        # Continue from interrupt - this runs the pending node
        for event in graph.stream(None, config, stream_mode="updates"):
            print(f"Event during continue: {list(event.keys())}")
            
            # Check if this is a feedback node completing
            if "feedback_stage1_node" in event:
                # Inject approval into the state for next routing
                graph.update_state(config, {"stage1_approved": True})
            elif "feedback_stage2_node" in event:
                graph.update_state(config, {"stage2_approved": True})
        
        snapshot = graph.get_state(config)
    
    logger.info("\nPipeline completed (auto-approved)")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Test MCP tool generation pipeline with human-in-the-loop feedback"
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
    parser.set_defaults(verbosity="normal")
    
    args = parser.parse_args()
    
    if args.auto:
        test_auto_approve()
    else:
        test_with_feedback(verbosity=args.verbosity)
