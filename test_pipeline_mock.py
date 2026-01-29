"""End-to-end pipeline test with mock LLM (no server required)."""

import sys
from pathlib import Path
from typing import Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Mock LLM Client
class MockLLMClient:
    """Mock LLM client for testing without server."""
    
    def __init__(self):
        self.model = "mock-qwen"
        self.temperature = 0.3
        self.call_count = 0  # Track number of calls
    
    def generate(self, prompt: str, temperature: Optional[float] = None) -> str:
        """Return mock response based on call order."""
        self.call_count += 1
        
        # First call: Intent extraction (with schema)
        if self.call_count == 1 or "implementation_plan" in prompt.lower():
            return """```json
{
  "operation": "groupby_aggregate",
  "columns": ["AccidentType"],
  "metrics": ["count"],
  "filters": [],
  "sort_by": ["count"],
  "sort_order": "descending",
  "limit": 5,
  "output_format": "table",
  "implementation_plan": [
    {"step": 1, "action": "Load CSV", "details": "Read traffic_accidents.csv"},
    {"step": 2, "action": "Group by AccidentType", "details": "Group data by accident type"},
    {"step": 3, "action": "Count occurrences", "details": "Count accidents per type"},
    {"step": 4, "action": "Sort descending", "details": "Sort by count"},
    {"step": 5, "action": "Limit to top 5", "details": "Return top 5 results"}
  ],
  "expected_output": {
    "columns": ["AccidentType", "count"],
    "format": "markdown_table"
  },
  "edge_cases": ["empty dataset", "missing AccidentType column"],
  "validation_rules": ["AccidentType column must exist"]
}
```"""
        
        # Second call: ToolSpec generation
        elif self.call_count == 2 or "TOOL NAME:" in prompt:
            return """```json
{
  "tool_name": "top_accident_types",
  "description": "Get the top 5 accident types by count from traffic data",
  "version": "1.0.0",
  "input_schema": {
    "type": "object",
    "properties": {
      "file_path": {"type": "string", "description": "Path to traffic accidents CSV"}
    },
    "required": ["file_path"]
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "results": {"type": "array"},
      "count": {"type": "integer"}
    }
  },
  "parameters": [
    {
      "name": "file_path",
      "type": "str",
      "description": "Path to the traffic accidents CSV file",
      "required": true
    }
  ],
  "return_type": "Dict[str, Any]",
  "when_to_use": "When you need to analyze traffic accident data and find the most common accident types",
  "what_it_does": "Loads traffic accident data, groups by accident type, counts occurrences, sorts by frequency, and returns top 5",
  "returns": "Dictionary with top 5 accident types and their counts",
  "prerequisites": "CSV file with AccidentType column"
}
```"""
        
        # Third call and onwards: Code generation
        else:
            return """```python
from fastmcp import FastMCP
import pandas as pd
from typing import Dict, Any

mcp = FastMCP("TrafficAnalysis")

@mcp.tool()
def top_accident_types(file_path: str) -> Dict[str, Any]:
    \"\"\"Get the top 5 accident types by count from traffic data.
    
    Args:
        file_path: Path to the traffic accidents CSV file
        
    Returns:
        Dictionary with top 5 accident types and their counts
    \"\"\"
    try:
        df = pd.read_csv(file_path)
        
        if 'AccidentType' not in df.columns:
            return {"error": "AccidentType column not found"}
        
        result = df['AccidentType'].value_counts().head(5)
        
        return {
            "result": result.to_dict(),
            "metadata": {"total_types": len(result), "total_accidents": len(df)}
        }
    except Exception as e:
        return {"error": str(e)}
```"""
    
    def generate_structured(self, prompt: str, schema: Optional[Dict] = None) -> Dict:
        """Return mock structured response."""
        import json
        import re
        
        # Extract JSON from code block
        response_text = self.generate(prompt)
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        
        if json_match:
            return json.loads(json_match.group(1))
        
        # Fallback
        return {
            "operation": "groupby_aggregate",
            "columns": ["AccidentType"],
            "metrics": ["count"],
            "implementation_plan": [
                {"step": 1, "action": "Load CSV", "details": "Read data"}
            ]
        }


# Patch the LLM client creation
def mock_create_llm_client():
    """Create mock LLM client instead of real one."""
    return MockLLMClient()


# Monkey patch before importing pipeline
import src.llm_client
src.llm_client.create_llm_client = mock_create_llm_client
src.llm_client.QwenLLMClient = MockLLMClient

from src.pipeline import run_pipeline


def test_simple_query():
    """Test with a simple data analysis query using mock LLM."""
    
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
    print("TESTING PIPELINE (MOCK MODE - No LLM Server Required)")
    print("=" * 80)
    print(f"Query: {query}")
    print(f"Data: {data_path}")
    print("=" * 80)
    print()
    
    try:
        # Run pipeline with mock LLM
        print("Running pipeline...")
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
            print(f"  Has Gap: {intent.get('has_gap')}")
            if intent.get('gap_reason'):
                print(f"  Gap Reason: {intent.get('gap_reason')}")
            print(f"  Operation: {intent.get('operation')}")
            print(f"  Required Columns: {intent.get('required_columns')}")
            print(f"  Group By: {intent.get('group_by')}")
            print(f"  Metrics: {intent.get('metrics')}")
            print(f"  Filters: {intent.get('filters')}")
        
        # Print gap detection
        print(f"\n[GAP DETECTION]")
        print(f"  New tool needed: {result.get('has_gap')}")
        
        # Print tool spec
        if result.get("tool_spec"):
            print("\n[TOOL SPEC]")
            spec = result["tool_spec"]
            print(f"  Name: {spec.tool_name}")
            print(f"  Description: {spec.description}")
        
        # Print code generation
        if result.get("generated_code"):
            print("\n[CODE GENERATION]")
            print(f"  Code length: {len(result['generated_code'])} characters")
            print(f"  First 200 chars: {result['generated_code'][:200]}...")
        
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
        print("\nNote: This was a MOCK test. Full LLM testing requires:")
        print("  1. Start vLLM server: vllm serve Qwen/Qwen2.5-Coder-32B-Instruct")
        print("  2. Run: python test_pipeline.py")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_simple_query()
