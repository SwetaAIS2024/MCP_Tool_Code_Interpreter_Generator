"""End-to-end pipeline test script."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import run_pipeline


def test_simple_query():
    """Test with a simple data analysis query."""
    
    # Check if test data exists
    test_data = Path("reference_files/sample_planner_output/traffic_accidents.csv")
    
    if not test_data.exists():
        print(f"Error: Test data not found at {test_data}")
        print("Please ensure the traffic_accidents.csv file exists.")
        return
    
    # Simple test query
    query = "Show me the top 5 accident types by count"
    data_path = str(test_data)
    
    print("=" * 80)
    print("TESTING PIPELINE")
    print("=" * 80)
    print(f"Query: {query}")
    print(f"Data: {data_path}")
    print("=" * 80)
    print()
    print("⚠️  NOTE: This test will timeout at feedback stages.")
    print("    Use test_pipeline_with_feedback.py for interactive testing.")
    print("    Or use: python test_pipeline_with_feedback.py --auto")
    print()
    
    try:
        # Run pipeline (will timeout at interrupt points)
        result = run_pipeline(query, data_path)
        
        print("\n" + "=" * 80)
        print("PIPELINE RESULT")
        print("=" * 80)
        
        # Print state keys
        print("\nState keys:")
        for key in result.keys():
            value = result[key]
            if value is not None and key not in ['messages']:
                print(f"  - {key}: {type(value).__name__}")
        
        # Print extracted intent
        if result.get("extracted_intent"):
            print("\n[INTENT EXTRACTION]")
            intent = result["extracted_intent"]
            print(f"  Operation: {intent.get('operation')}")
            print(f"  Columns: {intent.get('columns')}")
            print(f"  Metrics: {intent.get('metrics')}")
        
        # Print tool spec
        if result.get("tool_spec"):
            print("\n[TOOL SPEC]")
            spec = result["tool_spec"]
            print(f"  Name: {spec.tool_name}")
            print(f"  Description: {spec.description}")
        
        # Print generated code
        if result.get("generated_code"):
            print("\n[GENERATED CODE]")
            print(f"  Code length: {len(result['generated_code'])} characters")
            print(f"  Repair attempts: {result.get('repair_attempts', 0)}")
            print(f"  First 500 chars:")
            print("  " + "-" * 76)
            for line in result['generated_code'][:500].split('\n'):
                print(f"  {line}")
            print("  " + "-" * 76)
        
        # Print validation
        if result.get("validation_result"):
            print("\n[VALIDATION]")
            val = result["validation_result"]
            print(f"  Schema OK: {val.schema_ok}")
            print(f"  Tests OK: {val.tests_ok}")
            print(f"  Sandbox OK: {val.sandbox_ok}")
            if val.errors:
                print(f"  Errors: {val.errors}")
        
        # Print execution output
        if result.get("execution_output"):
            print("\n[EXECUTION OUTPUT]")
            output = result["execution_output"]
            print(f"  Result: {output.result}")
            if output.error:
                print(f"  Error: {output.error}")
        
        # Print promotion
        if result.get("promoted_tool"):
            print("\n[PROMOTED TOOL]")
            promoted = result["promoted_tool"]
            print(f"  Name: {promoted.get('name')}")
            print(f"  Path: {promoted.get('path')}")
            print(f"  Version: {promoted.get('version')}")
        
        print("\n" + "=" * 80)
        print("TEST COMPLETED")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_simple_query()
