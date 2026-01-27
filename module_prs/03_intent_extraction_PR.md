# Intent Extraction Module

**Module**: `src/intent_extraction.py`  
**Priority**: P0  
**Effort**: 2-3 days

---

## LangGraph Node

```python
def intent_node(state: ToolGeneratorState) -> ToolGeneratorState:
    """Extract intent and detect gap."""
    intent = extract_intent(state["user_query"], state["data_path"])
    gap_detected = detect_capability_gap(intent)
    
    return {
        **state,
        "extracted_intent": intent,
        "has_gap": gap_detected
    }

def route_after_intent(state: ToolGeneratorState):
    return "spec_generator_node" if state["has_gap"] else "executor_node"
```

---

## Core Logic

```python
class IntentExtractor:
    def __init__(self, llm_client: QwenLLMClient):
        self.llm = llm_client
    
    def extract(self, query: str, data_path: str) -> Dict:
        # Load dataset schema for context
        df_preview = pd.read_csv(data_path, nrows=5)
        columns = list(df_preview.columns)
        dtypes = df_preview.dtypes.to_dict()
        sample_values = {col: df_preview[col].head(3).tolist() for col in columns}
        
        # Build comprehensive analysis prompt for Qwen
        prompt = f"""Analyze this data analysis request and create a detailed implementation plan.

USER QUERY: {query}
DATASET: {data_path}
AVAILABLE COLUMNS: {columns}
COLUMN TYPES: {dtypes}
SAMPLE VALUES: {sample_values}

Provide a thorough analysis with:

1. INTENT BREAKDOWN:
   - Primary operation (groupby, filter, aggregate, join, pivot, transform, etc.)
   - Required columns and their roles
   - Metrics/calculations needed (count, sum, mean, median, std, etc.)
   - Filter conditions (if any)
   - Sorting requirements
   - Output format (table, chart, summary, json)

2. IMPLEMENTATION PLAN (Step-by-step todo list):
   - List each discrete step needed to accomplish the task
   - Include data loading, transformations, calculations, formatting
   - Specify order of operations
   - Note any edge cases or validations needed

3. EXPECTED OUTPUT:
   - Describe what the final result should look like
   - Include column names, data types, format

Return JSON with this structure:
{{
  "operation": "groupby_aggregate",
  "columns": ["state", "severity"],
  "metrics": ["count", "mean"],
  "filters": [{{"column": "date", "operator": ">", "value": "2023-01-01"}}],
  "sort_by": ["count"],
  "sort_order": "descending",
  "limit": 10,
  "output_format": "table",
  "implementation_plan": [
    {{"step": 1, "action": "Load CSV file", "details": "Read traffic_accidents.csv"}},
    {{"step": 2, "action": "Validate columns", "details": "Check state, severity columns exist"}},
    {{"step": 3, "action": "Apply filters", "details": "Filter date > 2023-01-01"}},
    {{"step": 4, "action": "Group data", "details": "Group by state and severity"}},
    {{"step": 5, "action": "Calculate metrics", "details": "Count rows, calculate mean"}},
    {{"step": 6, "action": "Sort results", "details": "Sort by count descending"}},
    {{"step": 7, "action": "Limit output", "details": "Take top 10 rows"}},
    {{"step": 8, "action": "Format output", "details": "Convert to markdown table"}}
  ],
  "expected_output": {{
    "columns": ["state", "severity", "count", "mean_value"],
    "format": "markdown_table",
    "sample": "| state | severity | count | mean_value |\\n|-------|----------|-------|------------|"
  }},
  "edge_cases": ["empty dataset", "missing columns", "null values in groupby columns"],
  "validation_rules": ["columns must exist", "date must be valid format", "numeric columns for mean"]
}}
"""
        
        # Use Qwen LLM for detailed extraction
        return self.llm.generate_structured(prompt, {
            "type": "object",
            "properties": {
                "operation": {"type": "string"},
                "columns": {"type": "array", "items": {"type": "string"}},
                "metrics": {"type": "array", "items": {"type": "string"}},
                "filters": {"type": "array"},
                "sort_by": {"type": "array"},
                "sort_order": {"type": "string"},
                "limit": {"type": "integer"},
                "output_format": {"type": "string"},
                "implementation_plan": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {"type": "integer"},
                            "action": {"type": "string"},
                            "details": {"type": "string"}
                        }
                    }
                },
                "expected_output": {"type": "object"},
                "edge_cases": {"type": "array"},
                "validation_rules": {"type": "array"}
            },
            "required": ["operation", "columns", "implementation_plan"]
        })

class GapDetector:
    def detect(self, intent: Dict) -> bool:
        # Check if existing tool can handle this
        existing_tools = self._load_registry()
        overlap_scores = [self._calculate_overlap(intent, tool) for tool in existing_tools]
        return max(overlap_scores, default=0) < 0.85
```

---

---

## Example Output

```json
{
  "operation": "groupby_aggregate",
  "columns": ["state", "severity"],
  "metrics": ["count"],
  "implementation_plan": [
    {"step": 1, "action": "Load dataset", "details": "Read CSV with pandas"},
    {"step": 2, "action": "Validate columns", "details": "Ensure state, severity exist"},
    {"step": 3, "action": "Handle missing values", "details": "Drop rows with null in groupby columns"},
    {"step": 4, "action": "Group data", "details": "df.groupby(['state', 'severity'])"},
    {"step": 5, "action": "Aggregate", "details": "Count rows in each group"},
    {"step": 6, "action": "Sort results", "details": "Sort by count descending"},
    {"step": 7, "action": "Format output", "details": "Convert to markdown table"}
  ],
  "expected_output": {
    "columns": ["state", "severity", "count"],
    "format": "markdown_table"
  },
  "edge_cases": ["empty dataset", "missing columns", "all nulls"],
  "validation_rules": ["state and severity must be strings", "result must have > 0 rows"]
}
```

---

## Implementation Checklist

- [ ] Implement LLM-based intent extraction with implementation plan
- [ ] Add dataset schema inspection
- [ ] Generate step-by-step todo list
- [ ] Implement gap detection
- [ ] Write tests
