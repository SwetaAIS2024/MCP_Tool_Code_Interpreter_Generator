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
        "promoted_tool": None,
        "messages": []
    }
    
    # Stream execution
    config = {"configurable": {"thread_id": "test-1"}}
    current_state = initial_state
    
    print("\nExecuting pipeline (direct execution without feedback stages)...")
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
        print(f"  Logs Path: {tool.get('logs_path')}")
        print(f"  Registry Path: {tool.get('registry_path')}")
    else:
        print("\nTool was not promoted to registry")
    
    # Show execution output if available
    if current_state.get("execution_output"):
        exec_out = current_state["execution_output"]
        print("\n[EXECUTION OUTPUT]")
        print(f"  Result: {str(exec_out.get('result'))[:200]}")
        print(f"  Execution Time: {exec_out.get('execution_time_ms', 0):.2f}ms")
        if exec_out.get('error'):
            print(f"  Error: {exec_out.get('error')}")
    
    print("\n" + "=" * 80)


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
        "validation_result": None,
        "repair_attempts": 0,
        "execution_output": None,
        "promoted_tool": None,
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
