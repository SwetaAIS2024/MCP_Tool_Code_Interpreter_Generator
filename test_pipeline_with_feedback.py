"""End-to-end pipeline test with interactive feedback."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import build_graph
from src.models import ToolGeneratorState


def test_with_feedback():
    """Test pipeline with user feedback at interrupt points."""
    
    # Check if test data exists
    test_data = Path("reference_files/sample_planner_output/traffic_accidents.csv")
    
    if not test_data.exists():
        print(f"Error: Test data not found at {test_data}")
        return
    
    # Test query
    query = "Show me the top 5 accident types by count"
    data_path = str(test_data)
    
    print("=" * 80)
    print("TESTING PIPELINE WITH FEEDBACK")
    print("=" * 80)
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
        print(f"\n📍 Event: {list(event.keys())}")
        
        # Update current state
        for key, value in event.items():
            if isinstance(value, dict):
                current_state.update(value)
    
    # Check if we hit an interrupt
    snapshot = graph.get_state(config)
    
    while snapshot.next:
        print("\n" + "=" * 80)
        print("⏸️  PIPELINE PAUSED - AWAITING FEEDBACK")
        print("=" * 80)
        print(f"Next node: {snapshot.next}")
        
        # Show current state
        if "feedback_stage1" in str(snapshot.next):
            print("\n🔍 STAGE 1 APPROVAL - Review Execution Output")
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
                print(f"  Result: {str(exec_out.get('result', {}))[:200]}")
                if exec_out.get("error"):
                    print(f"  Error: {exec_out['error']}")
            
            # Get user input
            print("\n" + "-" * 80)
            response = input("Approve execution output? (yes/no): ").strip()
            
            # Update state with response as the feedback node
            graph.update_state(config, {"user_response_stage1": response}, as_node="feedback_stage1_node")
            
        elif "feedback_stage2" in str(snapshot.next):
            print("\n🔍 STAGE 2 APPROVAL - Promote to Registry")
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
            
            # Update state with response as the feedback node
            graph.update_state(config, {"user_response_stage2": response}, as_node="feedback_stage2_node")
        
        # Continue execution
        print("\n▶️  Resuming pipeline...")
        for event in graph.stream(None, config, stream_mode="updates"):
            print(f"📍 Event: {list(event.keys())}")
            for key, value in event.items():
                if isinstance(value, dict):
                    current_state.update(value)
        
        # Get new snapshot
        snapshot = graph.get_state(config)
    
    # Final results
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETED")
    print("=" * 80)
    
    if current_state.get("promoted_tool"):
        print("\n[PROMOTED TOOL]")
        tool = current_state["promoted_tool"]
        print(f"  Name: {tool.get('name')}")
        print(f"  Version: {tool.get('version')}")
        print(f"  Registry Path: {tool.get('registry_path')}")
    else:
        print("\n⚠️  Tool was not promoted to registry")
    
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
        print(f"📍 Event: {list(event.keys())}")
    
    # Auto-approve at each interrupt
    snapshot = graph.get_state(config)
    
    while snapshot.next:
        print(f"\n⏸️  Auto-approving: {snapshot.next}")
        
        # Continue from interrupt - this runs the pending node
        for event in graph.stream(None, config, stream_mode="updates"):
            print(f"📍 Event during continue: {list(event.keys())}")
            
            # Check if this is a feedback node completing
            if "feedback_stage1_node" in event:
                # Inject approval into the state for next routing
                graph.update_state(config, {"stage1_approved": True})
            elif "feedback_stage2_node" in event:
                graph.update_state(config, {"stage2_approved": True})
        
        snapshot = graph.get_state(config)
    
    print("\n✅ Pipeline completed (auto-approved)")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--auto":
        test_auto_approve()
    else:
        test_with_feedback()
