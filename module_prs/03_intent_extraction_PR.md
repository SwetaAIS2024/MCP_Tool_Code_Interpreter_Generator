# Intent Extraction & Gap Detection Module - Project Requirements

**Module**: `src/intent_extraction.py`  
**Version**: 1.0.0  
**Dependencies**: `src/llm_client.py`, `src/models.py`, `pandas`  
**Parent PR**: Main ProjectRequirements.instructions.md - Section 4, Step 1

---

## 1. Module Purpose

Extract structured intent from natural language requests and determine if a new tool is needed:
- Parse user requests into structured intents
- Query active tool registry for existing capabilities
- Calculate capability overlap scores
- Decide: reuse existing tool vs. generate new tool

---

## 2. Core Components

### 2.1 Intent Data Structure

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class UserIntent(BaseModel):
    """Structured representation of user request"""
    
    # Primary operation
    operation: str = Field(
        description="Core data operation: groupby, filter, aggregate, join, pivot, etc."
    )
    
    # Data sources
    input_files: List[str] = Field(description="CSV/data file paths")
    
    # Column requirements
    required_columns: List[str] = Field(
        description="Columns needed for operation"
    )
    optional_columns: List[str] = Field(default_factory=list)
    
    # Metrics/aggregations
    metrics: List[str] = Field(
        default_factory=list,
        description="count, sum, mean, etc."
    )
    
    # Filters/conditions
    filters: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Filter conditions"
    )
    
    # Output requirements
    output_format: str = Field(
        default="markdown_table",
        description="markdown_table, json, csv, chart"
    )
    
    # Sorting/limiting
    sort_by: Optional[List[str]] = None
    limit: Optional[int] = None
    
    # Original user query
    original_query: str = Field(description="Raw user input")
    
    # Extracted dataset info
    dataset_schema: Optional[Dict[str, Any]] = None
```

---

## 3. Implementation

### 3.1 Intent Extractor

```python
from src.llm_client import BaseLLMClient, LLMTask
from src.models import ToolCandidate
import re
import json

class IntentExtractor:
    """Extract structured intent from natural language"""
    
    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client
        self.operation_keywords = self._load_operation_mappings()
    
    def extract(
        self,
        user_query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> UserIntent:
        """
        Extract intent from user query
        
        Args:
            user_query: Natural language request
            context: Optional context (file paths, previous results, etc.)
        
        Returns:
            UserIntent with structured information
        """
        
        # Step 1: Quick pattern matching for simple cases
        if simple_intent := self._try_pattern_match(user_query):
            return simple_intent
        
        # Step 2: LLM-based extraction for complex queries
        intent_dict = self._llm_extract(user_query, context)
        
        # Step 3: Enrich with dataset schema if file provided
        if intent_dict.get("input_files"):
            intent_dict["dataset_schema"] = self._get_dataset_schema(
                intent_dict["input_files"][0]
            )
        
        return UserIntent(**intent_dict)
    
    def _try_pattern_match(self, query: str) -> Optional[UserIntent]:
        """Fast extraction for common patterns"""
        
        query_lower = query.lower()
        
        # Pattern: "group by X and count"
        if match := re.search(r"group by (\\w+)(?: and (\\w+))? and count", query_lower):
            columns = [match.group(1)]
            if match.group(2):
                columns.append(match.group(2))
            
            return UserIntent(
                operation="groupby_count",
                input_files=[self._extract_file_path(query)],
                required_columns=columns,
                metrics=["count"],
                original_query=query
            )
        
        # Pattern: "filter X where Y > Z"
        if match := re.search(r"filter (\\w+) where (\\w+) ([><=]+) ([\\w\\.]+)", query_lower):
            return UserIntent(
                operation="filter",
                input_files=[self._extract_file_path(query)],
                required_columns=[match.group(1), match.group(2)],
                filters=[{
                    "column": match.group(2),
                    "operator": match.group(3),
                    "value": match.group(4)
                }],
                original_query=query
            )
        
        return None
    
    def _llm_extract(
        self,
        query: str,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Use LLM to extract complex intent"""
        
        prompt = f\"\"\"Extract structured intent from this data analysis request.

USER QUERY:
{query}

{f"CONTEXT:\\n{json.dumps(context, indent=2)}" if context else ""}

Extract and return JSON with:
{{
  "operation": "groupby | filter | aggregate | join | pivot | transform",
  "input_files": ["file1.csv", ...],
  "required_columns": ["col1", "col2", ...],
  "optional_columns": ["col3", ...],
  "metrics": ["count", "sum", "mean", ...],
  "filters": [{{"column": "...", "operator": ">", "value": "..."}}],
  "output_format": "markdown_table | json | csv",
  "sort_by": ["col1", ...],
  "limit": 100
}}

Respond only with valid JSON.\"\"\"
        
        intent_schema = {
            "type": "object",
            "properties": {
                "operation": {"type": "string"},
                "input_files": {"type": "array", "items": {"type": "string"}},
                "required_columns": {"type": "array", "items": {"type": "string"}},
                "metrics": {"type": "array"},
                "filters": {"type": "array"},
                "output_format": {"type": "string"}
            },
            "required": ["operation", "input_files", "required_columns"]
        }
        
        result = self.llm_client.generate_structured(
            prompt,
            intent_schema,
            LLMTask.INTENT_EXTRACTION
        )
        
        result["original_query"] = query
        return result
    
    def _extract_file_path(self, query: str) -> str:
        """Extract file path from query"""
        # Look for .csv, .parquet, etc.
        if match := re.search(r'([\\w/_\\-\\.]+\\.(?:csv|parquet|json))', query):
            return match.group(1)
        return "data.csv"  # default
    
    def _get_dataset_schema(self, file_path: str) -> Dict[str, Any]:
        """Get dataset schema from file"""
        try:
            import pandas as pd
            df = pd.read_csv(file_path, nrows=5)
            
            return {
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "row_count": len(df),
                "sample_values": {
                    col: df[col].head(3).tolist()
                    for col in df.columns
                }
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _load_operation_mappings(self) -> Dict[str, List[str]]:
        """Map keywords to operations"""
        return {
            "groupby": ["group by", "group", "aggregate by", "count by"],
            "filter": ["filter", "where", "select where", "subset"],
            "join": ["join", "merge", "combine"],
            "pivot": ["pivot", "crosstab", "pivot table"],
            "sort": ["sort", "order by", "rank"],
            "aggregate": ["sum", "average", "mean", "total"]
        }
```

---

### 3.2 Gap Detector

```python
from pathlib import Path
from typing import Tuple, Optional

class GapDetector:
    """Determine if existing tool can be reused"""
    
    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self.active_tools = self._load_active_tools()
    
    def check_gap(
        self,
        intent: UserIntent
    ) -> Tuple[bool, Optional[str], float]:
        """
        Check if a new tool is needed
        
        Returns:
            (needs_new_tool, existing_tool_id, max_overlap_score)
        """
        
        best_tool = None
        max_overlap = 0.0
        
        for tool_id, tool_info in self.active_tools.items():
            overlap = self._calculate_overlap(intent, tool_info)
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_tool = tool_id
        
        # Threshold: 0.85 (85% overlap)
        if max_overlap >= 0.85:
            return False, best_tool, max_overlap  # Reuse existing
        else:
            return True, best_tool, max_overlap   # Generate new
    
    def _calculate_overlap(
        self,
        intent: UserIntent,
        tool_info: Dict[str, Any]
    ) -> float:
        """
        Calculate capability overlap score (0.0 to 1.0)
        
        Weighted components:
        - Operation match: 30%
        - Input compatibility: 25%
        - Output format match: 20%
        - Constraint match: 15%
        - Semantic similarity: 10%
        """
        
        # Load tool spec
        spec = self._load_tool_spec(tool_info["path"])
        
        # 1. Operation match (0 or 1)
        operation_match = 1.0 if self._operations_match(
            intent.operation,
            spec.get("operation", "")
        ) else 0.0
        
        # 2. Input compatibility (0 to 1)
        input_compat = self._check_input_compatibility(
            intent.required_columns,
            spec.get("input_schema", {})
        )
        
        # 3. Output format match (0 or 1)
        output_match = 1.0 if intent.output_format in spec.get("returns", "") else 0.0
        
        # 4. Constraint match (0 to 1)
        constraint_match = self._check_constraints(
            intent.filters,
            spec.get("supports_filters", False)
        )
        
        # 5. Semantic similarity (0 to 1)
        semantic_sim = self._semantic_similarity(
            intent.original_query,
            spec.get("description", "")
        )
        
        # Weighted sum
        overlap_score = (
            0.30 * operation_match +
            0.25 * input_compat +
            0.20 * output_match +
            0.15 * constraint_match +
            0.10 * semantic_sim
        )
        
        return overlap_score
    
    def _operations_match(self, op1: str, op2: str) -> bool:
        """Check if operations are equivalent"""
        # Normalize operation names
        normalize = lambda x: x.lower().replace("_", "").replace("-", "")
        return normalize(op1) == normalize(op2)
    
    def _check_input_compatibility(
        self,
        required_cols: List[str],
        input_schema: Dict
    ) -> float:
        """Check if tool can handle required inputs"""
        
        if not input_schema or "properties" not in input_schema:
            return 0.0
        
        schema_params = set(input_schema["properties"].keys())
        
        # Check if all required columns can be provided
        # (tool might take generic column list parameter)
        if "columns" in schema_params or "group_by_columns" in schema_params:
            return 1.0  # Generic column support
        
        # Check direct column matches
        required_set = set(required_cols)
        if required_set.issubset(schema_params):
            return 1.0
        
        # Partial match
        overlap = len(required_set & schema_params)
        return overlap / len(required_set) if required_set else 0.0
    
    def _check_constraints(
        self,
        filters: List[Dict],
        supports_filters: bool
    ) -> float:
        """Check if tool supports required constraints"""
        if not filters:
            return 1.0  # No constraints needed
        
        return 1.0 if supports_filters else 0.0
    
    def _semantic_similarity(self, query: str, description: str) -> float:
        """Simple semantic similarity (can enhance with embeddings)"""
        # For MVP: keyword overlap
        query_words = set(query.lower().split())
        desc_words = set(description.lower().split())
        
        if not query_words or not desc_words:
            return 0.0
        
        overlap = len(query_words & desc_words)
        union = len(query_words | desc_words)
        
        return overlap / union if union > 0 else 0.0
    
    def _load_active_tools(self) -> Dict[str, Dict]:
        """Load active tool registry"""
        metadata_file = self.registry_path / "active" / "metadata.json"
        
        if not metadata_file.exists():
            return {}
        
        import json
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        return metadata.get("tools", {})
    
    def _load_tool_spec(self, tool_path: str) -> Dict[str, Any]:
        """Load tool specification"""
        spec_file = Path(tool_path) / "spec.json"
        
        if not spec_file.exists():
            return {}
        
        import json
        with open(spec_file) as f:
            return json.load(f)
```

---

## 4. Main Pipeline Function

```python
from dataclasses import dataclass

@dataclass
class IntentAnalysisResult:
    """Result of intent extraction and gap detection"""
    intent: UserIntent
    needs_new_tool: bool
    existing_tool_id: Optional[str]
    overlap_score: float
    recommendation: str

def analyze_user_request(
    user_query: str,
    llm_client: BaseLLMClient,
    registry_path: Path,
    context: Optional[Dict] = None
) -> IntentAnalysisResult:
    """
    Main entry point: extract intent and check for gaps
    
    Args:
        user_query: Natural language request
        llm_client: LLM client for extraction
        registry_path: Path to tool registry
        context: Optional context
    
    Returns:
        IntentAnalysisResult with decision and reasoning
    """
    
    # Step 1: Extract intent
    extractor = IntentExtractor(llm_client)
    intent = extractor.extract(user_query, context)
    
    # Step 2: Check for existing tools
    detector = GapDetector(registry_path)
    needs_new, existing_tool, overlap = detector.check_gap(intent)
    
    # Step 3: Build recommendation
    if needs_new:
        recommendation = (
            f"Generate new tool (best match: {existing_tool or 'none'} "
            f"with {overlap*100:.1f}% overlap)"
        )
    else:
        recommendation = (
            f"Use existing tool '{existing_tool}' "
            f"({overlap*100:.1f}% match)"
        )
    
    return IntentAnalysisResult(
        intent=intent,
        needs_new_tool=needs_new,
        existing_tool_id=existing_tool,
        overlap_score=overlap,
        recommendation=recommendation
    )
```

---

## 5. Testing Requirements

### 5.1 Unit Tests

```python
# tests/test_intent_extraction.py

def test_simple_groupby_pattern():
    query = "Group by region and count rows in sales.csv"
    extractor = IntentExtractor(mock_llm_client)
    
    intent = extractor._try_pattern_match(query)
    
    assert intent is not None
    assert intent.operation == "groupby_count"
    assert "region" in intent.required_columns
    assert "count" in intent.metrics

def test_llm_extraction():
    query = "Show me the top 10 products by total sales value"
    extractor = IntentExtractor(llm_client)
    
    intent = extractor.extract(query)
    
    assert intent.operation in ["groupby", "aggregate"]
    assert intent.limit == 10
    assert "sales" in intent.metrics or "sum" in intent.metrics

def test_overlap_calculation_exact_match():
    detector = GapDetector(registry_path)
    
    intent = UserIntent(
        operation="groupby_count",
        required_columns=["region"],
        metrics=["count"],
        output_format="markdown_table"
    )
    
    tool_info = {
        "path": "path/to/groupby_tool",
        "operation": "groupby_count"
    }
    
    # Mock spec loading
    overlap = detector._calculate_overlap(intent, tool_info)
    assert overlap >= 0.85  # High overlap

def test_gap_detection_reuse_existing():
    detector = GapDetector(registry_path)
    
    intent = UserIntent(...)  # Matches existing tool
    
    needs_new, tool_id, overlap = detector.check_gap(intent)
    
    assert not needs_new
    assert tool_id is not None
    assert overlap >= 0.85

def test_gap_detection_create_new():
    detector = GapDetector(registry_path)
    
    intent = UserIntent(
        operation="complex_pivot_with_custom_agg",
        ...
    )
    
    needs_new, tool_id, overlap = detector.check_gap(intent)
    
    assert needs_new
    assert overlap < 0.85
```

---

## 6. Implementation Checklist

- [ ] Implement UserIntent model with validation
- [ ] Implement IntentExtractor with pattern matching
- [ ] Add LLM-based extraction for complex queries
- [ ] Implement dataset schema detection
- [ ] Implement GapDetector with overlap scoring
- [ ] Add operation matching logic
- [ ] Add input compatibility checking
- [ ] Add semantic similarity (keyword-based MVP)
- [ ] Create main analyze_user_request function
- [ ] Write unit tests (>90% coverage)
- [ ] Write integration tests with sample queries
- [ ] Add logging and debugging output
- [ ] Document common query patterns

---

## 7. Configuration

```yaml
# config.yaml addition
intent_extraction:
  overlap_threshold: 0.85
  pattern_matching_enabled: true
  use_llm_fallback: true
  cache_intents: true
  
gap_detection:
  weights:
    operation_match: 0.30
    input_compatibility: 0.25
    output_format_match: 0.20
    constraint_match: 0.15
    semantic_similarity: 0.10
```

---

**Status**: Ready for Implementation  
**Priority**: P0 (Entry point of pipeline)  
**Estimated Effort**: 2-3 days  
**Prerequisites**: LLM Client, Data Models
