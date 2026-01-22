# Module PR 04: Spec Generator

**Module**: `src/spec_generator.py`  
**Priority**: P0 (Core - Code generation pipeline)  
**Estimated Effort**: 2-3 days  
**Dependencies**: `01_data_models`, `02_llm_client`, `03_intent_extraction`

---

## 1. Module Purpose

The Spec Generator translates a `UserIntent` (extracted from natural language) into a formal `ToolSpec` that serves as the blueprint for code generation. It determines:
- Function name and description
- Input parameter schemas (with types and constraints)
- Output schema (structured result format)
- Constraints and validation rules
- Reference tools for similar functionality

**Key Principle**: The ToolSpec must be complete and unambiguous. The code generator should NOT need to make semantic decisions—all logic is defined in the spec.

---

## 2. Core Components

### 2.1 SpecGenerator Class

```python
class SpecGenerator:
    """Generate formal ToolSpec from UserIntent."""
    
    def __init__(self, llm_client: BaseLLMClient, reference_tools_dir: str):
        self.llm = llm_client
        self.reference_tools_dir = Path(reference_tools_dir)
        self.reference_tools = self._load_reference_tools()
    
    def generate(
        self,
        intent: UserIntent,
        reference_only: bool = False
    ) -> ToolSpec:
        """
        Generate ToolSpec from UserIntent.
        
        Args:
            intent: Extracted user intent
            reference_only: If True, only use reference tools (no LLM)
        
        Returns:
            Complete ToolSpec ready for code generation
        """
        pass
    
    def _load_reference_tools(self) -> List[Dict[str, Any]]:
        """Load reference tool specs from disk."""
        pass
    
    def _select_relevant_references(self, intent: UserIntent) -> List[Dict]:
        """Find reference tools similar to the intent."""
        pass
    
    def _generate_with_llm(
        self,
        intent: UserIntent,
        references: List[Dict]
    ) -> ToolSpec:
        """Use LLM to generate spec with reference examples."""
        pass
    
    def _infer_input_schema(self, intent: UserIntent) -> Dict[str, Any]:
        """Generate JSON Schema for input parameters."""
        pass
    
    def _infer_output_schema(self, intent: UserIntent) -> Dict[str, Any]:
        """Generate JSON Schema for output structure."""
        pass
    
    def _generate_function_name(self, intent: UserIntent) -> str:
        """Create Python function name from intent."""
        pass
    
    def _validate_spec_completeness(self, spec: ToolSpec) -> None:
        """Ensure spec has all required fields."""
        pass
```

---

## 3. Data Structures

### 3.1 Input: UserIntent (from module 03)

```python
class UserIntent(BaseModel):
    operation: str  # "group_by_and_aggregate", "filter_rows", etc.
    columns: List[str]  # ["state", "severity"]
    metrics: Optional[List[str]] = None  # ["count", "mean"]
    filters: Optional[List[Dict[str, Any]]] = None
    join_specs: Optional[List[Dict[str, Any]]] = None
    sort_by: Optional[List[str]] = None
    limit: Optional[int] = None
```

### 3.2 Output: ToolSpec (from module 01)

```python
class ToolSpec(BaseModel):
    tool_name: str  # "group_by_state_and_severity"
    description: str
    input_schema: Dict[str, Any]  # JSON Schema
    output_schema: Dict[str, Any]  # JSON Schema
    constraints: List[str]
    examples: List[Dict[str, Any]]
```

### 3.3 Reference Tool Format

```json
{
  "name": "group_by_column_with_count",
  "description": "Group DataFrame by column and count occurrences",
  "input_schema": {
    "type": "object",
    "properties": {
      "df": {
        "type": "object",
        "description": "pandas DataFrame"
      },
      "column": {
        "type": "string",
        "description": "Column name to group by"
      }
    },
    "required": ["df", "column"]
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "result": {
        "type": "object",
        "description": "DataFrame with grouped counts"
      }
    }
  },
  "code_snippet": "...",
  "tags": ["groupby", "aggregate", "count"]
}
```

---

## 4. Implementation

### 4.1 Main Generation Flow

```python
def generate(self, intent: UserIntent, reference_only: bool = False) -> ToolSpec:
    # Step 1: Find similar reference tools
    references = self._select_relevant_references(intent)
    
    if reference_only and not references:
        raise ValueError("No matching reference tools found")
    
    # Step 2: Generate with LLM (or adapt reference)
    if references and reference_only:
        # Use reference template directly
        spec = self._adapt_reference_spec(references[0], intent)
    else:
        # Generate with LLM using references as examples
        spec = self._generate_with_llm(intent, references)
    
    # Step 3: Validate completeness
    self._validate_spec_completeness(spec)
    
    return spec
```

### 4.2 Schema Generation

```python
def _infer_input_schema(self, intent: UserIntent) -> Dict[str, Any]:
    """
    Generate JSON Schema for tool inputs.
    
    Example for "group by state and severity":
    {
      "type": "object",
      "properties": {
        "df": {
          "type": "object",
          "description": "Input DataFrame with traffic accident data"
        },
        "state_column": {
          "type": "string",
          "enum": ["state"],
          "description": "Column name for state grouping"
        },
        "severity_column": {
          "type": "string",
          "enum": ["severity"],
          "description": "Column name for severity grouping"
        }
      },
      "required": ["df", "state_column", "severity_column"]
    }
    """
    schema = {
        "type": "object",
        "properties": {
            "df": {
                "type": "object",
                "description": "Input pandas DataFrame"
            }
        },
        "required": ["df"]
    }
    
    # Add column parameters
    for col in intent.columns:
        param_name = f"{col}_column"
        schema["properties"][param_name] = {
            "type": "string",
            "enum": [col],
            "description": f"Column name for {col}"
        }
        schema["required"].append(param_name)
    
    # Add metric parameters if aggregation
    if intent.metrics:
        schema["properties"]["metrics"] = {
            "type": "array",
            "items": {"type": "string"},
            "enum": intent.metrics,
            "description": "Aggregation functions to apply"
        }
    
    return schema
```

```python
def _infer_output_schema(self, intent: UserIntent) -> Dict[str, Any]:
    """
    Generate JSON Schema for tool outputs.
    
    Example:
    {
      "type": "object",
      "properties": {
        "result": {
          "type": "object",
          "description": "DataFrame with grouped results"
        },
        "summary": {
          "type": "string",
          "description": "Human-readable summary"
        },
        "metadata": {
          "type": "object",
          "properties": {
            "row_count": {"type": "integer"},
            "columns": {"type": "array", "items": {"type": "string"}}
          }
        }
      },
      "required": ["result"]
    }
    """
    return {
        "type": "object",
        "properties": {
            "result": {
                "type": "object",
                "description": "Transformed DataFrame"
            },
            "summary": {
                "type": "string",
                "description": "Human-readable summary of results"
            },
            "metadata": {
                "type": "object",
                "properties": {
                    "row_count": {"type": "integer"},
                    "columns": {"type": "array", "items": {"type": "string"}}
                }
            }
        },
        "required": ["result"]
    }
```

### 4.3 LLM-Based Generation

```python
def _generate_with_llm(
    self,
    intent: UserIntent,
    references: List[Dict]
) -> ToolSpec:
    """Generate spec using LLM with reference examples."""
    
    prompt = self._build_spec_generation_prompt(intent, references)
    
    response = self.llm.generate_json(
        prompt=prompt,
        response_model=ToolSpec,
        temperature=0.3  # Low temp for consistency
    )
    
    # Post-process and validate
    spec = ToolSpec(**response)
    
    # Ensure function name is valid Python identifier
    spec.tool_name = self._sanitize_function_name(spec.tool_name)
    
    return spec
```

### 4.4 Prompt Template

```python
def _build_spec_generation_prompt(
    self,
    intent: UserIntent,
    references: List[Dict]
) -> str:
    """Build prompt for LLM spec generation."""
    
    reference_examples = "\n\n".join([
        f"### Reference {i+1}: {ref['name']}\n"
        f"Description: {ref['description']}\n"
        f"Input Schema:\n{json.dumps(ref['input_schema'], indent=2)}\n"
        f"Output Schema:\n{json.dumps(ref['output_schema'], indent=2)}"
        for i, ref in enumerate(references[:3])  # Top 3 references
    ])
    
    prompt = f"""Generate a formal tool specification for a pandas data analysis function.

## User Intent
Operation: {intent.operation}
Columns: {', '.join(intent.columns)}
Metrics: {', '.join(intent.metrics or [])}
Filters: {json.dumps(intent.filters, indent=2) if intent.filters else 'None'}

## Reference Tools (Similar Functions)
{reference_examples}

## Requirements
1. Create a JSON Schema for input parameters
2. Create a JSON Schema for output structure
3. Generate a descriptive function name (snake_case)
4. Write a clear description
5. List constraints and validation rules
6. Provide at least one usage example

## Output Format
Return a JSON object matching this structure:
{{
  "tool_name": "function_name_in_snake_case",
  "description": "Clear description of what the tool does",
  "input_schema": {{...JSON Schema...}},
  "output_schema": {{...JSON Schema...}},
  "constraints": ["constraint 1", "constraint 2"],
  "examples": [
    {{
      "input": {{...}},
      "output": {{...}},
      "description": "..."
    }}
  ]
}}

Generate the specification:"""
    
    return prompt
```

### 4.5 Function Name Generation

```python
def _generate_function_name(self, intent: UserIntent) -> str:
    """
    Create valid Python function name from intent.
    
    Examples:
    - operation="group_by", columns=["state", "severity"] 
      → "group_by_state_and_severity"
    - operation="filter", filters=[{"column": "severity", "op": ">", "value": 3}]
      → "filter_severity_greater_than_3"
    """
    parts = [intent.operation]
    
    # Add primary columns
    if intent.columns:
        parts.extend(intent.columns[:2])  # Max 2 columns for brevity
    
    # Add metric if single aggregation
    if intent.metrics and len(intent.metrics) == 1:
        parts.append(intent.metrics[0])
    
    # Sanitize and join
    name = "_".join(parts)
    name = re.sub(r'[^a-z0-9_]', '_', name.lower())
    name = re.sub(r'__+', '_', name)  # Remove double underscores
    
    # Ensure valid Python identifier
    if not name[0].isalpha():
        name = 'tool_' + name
    
    return name
```

### 4.6 Reference Tool Selection

```python
def _select_relevant_references(self, intent: UserIntent) -> List[Dict]:
    """Find reference tools similar to the intent."""
    
    scored_refs = []
    
    for ref in self.reference_tools:
        score = self._calculate_similarity(intent, ref)
        scored_refs.append((score, ref))
    
    # Sort by score descending
    scored_refs.sort(key=lambda x: x[0], reverse=True)
    
    # Return top 3
    return [ref for score, ref in scored_refs[:3] if score > 0.3]


def _calculate_similarity(self, intent: UserIntent, ref: Dict) -> float:
    """Calculate similarity score between intent and reference tool."""
    
    score = 0.0
    
    # Operation match (40% weight)
    if intent.operation in ref.get('tags', []):
        score += 0.4
    
    # Column match (30% weight)
    ref_params = ref.get('input_schema', {}).get('properties', {}).keys()
    column_overlap = len(set(intent.columns) & set(ref_params))
    if column_overlap > 0:
        score += 0.3 * (column_overlap / len(intent.columns))
    
    # Metric match (30% weight)
    if intent.metrics:
        ref_tags = ref.get('tags', [])
        metric_overlap = len(set(intent.metrics) & set(ref_tags))
        if metric_overlap > 0:
            score += 0.3 * (metric_overlap / len(intent.metrics))
    
    return score
```

---

## 5. Testing Requirements

### 5.1 Unit Tests

```python
import pytest
from src.spec_generator import SpecGenerator
from src.models import UserIntent, ToolSpec

def test_generate_spec_for_groupby_intent():
    """Test spec generation for group-by operation."""
    intent = UserIntent(
        operation="group_by_and_aggregate",
        columns=["state", "severity"],
        metrics=["count"]
    )
    
    llm_client = MockLLMClient()
    generator = SpecGenerator(llm_client, "reference_files/sample_mcp_tools")
    
    spec = generator.generate(intent)
    
    assert spec.tool_name == "group_by_state_and_severity_count"
    assert "DataFrame" in spec.description
    assert "df" in spec.input_schema["properties"]
    assert "state_column" in spec.input_schema["properties"]
    assert spec.input_schema["required"] == ["df", "state_column", "severity_column"]


def test_input_schema_generation():
    """Test JSON Schema generation for inputs."""
    intent = UserIntent(
        operation="filter",
        columns=["severity"],
        filters=[{"column": "severity", "op": ">", "value": 3}]
    )
    
    generator = SpecGenerator(MockLLMClient(), "refs")
    schema = generator._infer_input_schema(intent)
    
    assert schema["type"] == "object"
    assert "df" in schema["properties"]
    assert "severity_column" in schema["properties"]
    assert schema["properties"]["severity_column"]["enum"] == ["severity"]


def test_function_name_sanitization():
    """Test function name generation and sanitization."""
    intent = UserIntent(
        operation="group by",
        columns=["State Name", "Severity Level"]
    )
    
    generator = SpecGenerator(MockLLMClient(), "refs")
    name = generator._generate_function_name(intent)
    
    assert name.isidentifier()  # Valid Python identifier
    assert "group_by" in name
    assert "state_name" in name


def test_reference_tool_selection():
    """Test selection of relevant reference tools."""
    intent = UserIntent(
        operation="group_by",
        columns=["category"],
        metrics=["count"]
    )
    
    generator = SpecGenerator(MockLLMClient(), "reference_files/sample_mcp_tools")
    refs = generator._select_relevant_references(intent)
    
    assert len(refs) <= 3
    assert all("group" in ref['name'].lower() for ref in refs)


def test_spec_completeness_validation():
    """Test validation of spec completeness."""
    incomplete_spec = ToolSpec(
        tool_name="test_tool",
        description="Test",
        input_schema={},  # Missing properties
        output_schema={},
        constraints=[]
    )
    
    generator = SpecGenerator(MockLLMClient(), "refs")
    
    with pytest.raises(ValueError, match="input_schema must have 'properties'"):
        generator._validate_spec_completeness(incomplete_spec)
```

### 5.2 Integration Tests

```python
def test_end_to_end_spec_generation():
    """Test complete flow from intent to spec."""
    intent = UserIntent(
        operation="group_by_and_aggregate",
        columns=["state", "severity"],
        metrics=["count", "mean"]
    )
    
    llm_client = QwenLLMClient(base_url="http://localhost:8000/v1")
    generator = SpecGenerator(llm_client, "reference_files/sample_mcp_tools")
    
    spec = generator.generate(intent)
    
    # Validate structure
    assert spec.tool_name
    assert spec.description
    assert "type" in spec.input_schema
    assert "type" in spec.output_schema
    assert len(spec.constraints) > 0
    
    # Validate semantics
    assert "state" in str(spec.input_schema)
    assert "severity" in str(spec.input_schema)
    assert "count" in spec.description or "mean" in spec.description
```

---

## 6. Configuration

### 6.1 YAML Configuration

```yaml
spec_generation:
  reference_tools_dir: "reference_files/sample_mcp_tools"
  llm_temperature: 0.3  # Low for consistency
  max_references: 3
  similarity_threshold: 0.3
  
  # Schema defaults
  input_schema_defaults:
    df:
      type: "object"
      description: "Input pandas DataFrame"
  
  output_schema_defaults:
    result:
      type: "object"
      description: "Transformed DataFrame"
    summary:
      type: "string"
      description: "Human-readable summary"
```

---

## 7. Dependencies

### 7.1 Internal Modules
- `src/models.py` - UserIntent, ToolSpec
- `src/llm_client.py` - BaseLLMClient, QwenLLMClient

### 7.2 External Packages
```txt
pydantic>=2.5.0
pyyaml>=6.0
```

---

## 8. Implementation Checklist

- [ ] Create `SpecGenerator` class
- [ ] Implement `_load_reference_tools()` from JSON files
- [ ] Implement `_select_relevant_references()` with scoring
- [ ] Implement `_infer_input_schema()` with JSON Schema generation
- [ ] Implement `_infer_output_schema()` with standardized structure
- [ ] Implement `_generate_function_name()` with sanitization
- [ ] Implement `_build_spec_generation_prompt()` with references
- [ ] Implement `_generate_with_llm()` with structured output
- [ ] Implement `_validate_spec_completeness()` checks
- [ ] Add unit tests for each function (>90% coverage)
- [ ] Add integration test with real LLM
- [ ] Create reference tools JSON files (at least 5)
- [ ] Test with traffic accidents dataset
- [ ] Document all public methods
- [ ] Code review and refactor

---

## 9. Example Usage

```python
from src.spec_generator import SpecGenerator
from src.llm_client import QwenLLMClient
from src.models import UserIntent

# Initialize
llm_client = QwenLLMClient(base_url="http://localhost:8000/v1")
generator = SpecGenerator(
    llm_client=llm_client,
    reference_tools_dir="reference_files/sample_mcp_tools"
)

# Create intent
intent = UserIntent(
    operation="group_by_and_aggregate",
    columns=["state", "severity"],
    metrics=["count", "mean"]
)

# Generate spec
spec = generator.generate(intent)

print(f"Tool Name: {spec.tool_name}")
print(f"Description: {spec.description}")
print(f"Input Schema: {json.dumps(spec.input_schema, indent=2)}")
print(f"Output Schema: {json.dumps(spec.output_schema, indent=2)}")
```

**Expected Output**:
```
Tool Name: group_by_state_and_severity
Description: Group traffic accident data by state and severity, calculating count and mean
Input Schema: {
  "type": "object",
  "properties": {
    "df": {"type": "object", "description": "Input DataFrame"},
    "state_column": {"type": "string", "enum": ["state"]},
    "severity_column": {"type": "string", "enum": ["severity"]},
    "metrics": {"type": "array", "items": {"type": "string"}, "enum": ["count", "mean"]}
  },
  "required": ["df", "state_column", "severity_column"]
}
Output Schema: {
  "type": "object",
  "properties": {
    "result": {"type": "object", "description": "Grouped DataFrame"},
    "summary": {"type": "string"},
    "metadata": {"type": "object", "properties": {"row_count": {"type": "integer"}}}
  }
}
```

---

## 10. Edge Cases & Error Handling

### 10.1 Missing Reference Tools
```python
if not self.reference_tools:
    logger.warning("No reference tools found, relying on LLM only")
```

### 10.2 Ambiguous Intent
```python
if not intent.operation or not intent.columns:
    raise ValueError("Intent must specify operation and columns")
```

### 10.3 LLM Generation Failure
```python
try:
    spec = self._generate_with_llm(intent, references)
except Exception as e:
    logger.error(f"LLM generation failed: {e}")
    # Fallback to template-based generation
    spec = self._generate_from_template(intent)
```

### 10.4 Invalid Function Name
```python
if not spec.tool_name.isidentifier():
    spec.tool_name = self._sanitize_function_name(spec.tool_name)
```

---

## 11. Performance Considerations

- **LLM Calls**: ~2-5 seconds per spec generation
- **Reference Lookup**: <100ms for 100 reference tools
- **Schema Generation**: <50ms

**Optimization**:
- Cache LLM responses for identical intents
- Pre-compute reference tool embeddings for semantic search
- Batch process multiple intents if needed

---

## 12. Future Enhancements

1. **Semantic Reference Matching**: Use embeddings instead of keyword matching
2. **Multi-Step Specs**: Support tool composition (e.g., filter → group → join)
3. **Schema Auto-Refinement**: Learn from user corrections
4. **Type Inference**: Auto-detect pandas dtypes from data samples
5. **Constraint Learning**: Infer validation rules from data distribution

---

**Estimated Lines of Code**: 600-800  
**Test Coverage Target**: >90%  
**Ready for Implementation**: ✅
