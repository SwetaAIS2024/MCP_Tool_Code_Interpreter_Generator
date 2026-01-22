# Data Models Module - Project Requirements

**Module**: `src/models.py`  
**Version**: 1.0.0  
**Dependencies**: `pydantic`, `enum`, `typing`, `datetime`  
**Parent PR**: Main ProjectRequirements.instructions.md - Section 3

---

## 1. Module Purpose

Define all data models and type definitions used across the MCP Tool Code Generator pipeline. Provides type safety, validation, and serialization for:
- Tool specifications and candidates
- Validation reports
- Code quality metrics
- User feedback
- Registry metadata

---

## 2. Core Data Models

### 2.1 ToolStatus Enum

```python
from enum import Enum

class ToolStatus(str, Enum):
    """Tool lifecycle states"""
    DRAFT = "DRAFT"           # Initial generation, not validated
    STAGED = "STAGED"         # Validated, ready for user review
    APPROVED = "APPROVED"     # User approved, ready for promotion
    REJECTED = "REJECTED"     # User rejected
    PROMOTED = "PROMOTED"     # Registered in active registry
```

**Usage**: Track tool progression through pipeline stages.

---

### 2.2 ToolSpec Model

```python
from pydantic import BaseModel, Field
from typing import Dict, List, Any

class ToolSpec(BaseModel):
    """Contract-first tool specification"""
    
    # Identity
    tool_name: str = Field(description="Snake_case tool identifier")
    description: str = Field(description="One-sentence summary")
    
    # Formal schemas (machine-readable)
    input_schema: Dict[str, Any] = Field(
        description="JSON Schema for input validation"
    )
    output_schema: Dict[str, Any] = Field(
        description="JSON Schema for output structure"
    )
    
    # Legacy parameter info (backward compatibility)
    parameters: List[Dict[str, Any]] = Field(
        description="Annotated parameter definitions"
    )
    return_type: str = Field(default="str")
    
    # Documentation (human-readable)
    when_to_use: str = Field(description="Trigger conditions")
    what_it_does: str = Field(description="Step-by-step logic")
    returns: str = Field(description="Output format specification")
    prerequisites: str = Field(description="Required prior steps/state")
    
    class Config:
        json_schema_extra = {
            "example": {
                "tool_name": "group_and_count_by_columns",
                "description": "Group data by columns and count occurrences",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "group_by_columns": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["file_path", "group_by_columns"]
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "result": {"type": "string"},
                        "metadata": {"type": "object"}
                    }
                },
                "parameters": [...],
                "when_to_use": "When counting occurrences across dimensions",
                "what_it_does": "Loads CSV, groups by columns, counts rows",
                "returns": "Markdown table with counts",
                "prerequisites": "CSV file must exist"
            }
        }
```

**Validation Rules**:
- `tool_name`: Must be lowercase, snake_case, no spaces
- `input_schema`: Must be valid JSON Schema
- `output_schema`: Must be valid JSON Schema
- All text fields: Non-empty strings

---

### 2.3 Code Metrics Models

#### 2.3.1 FunctionalCorrectnessMetrics

```python
class FunctionalCorrectnessMetrics(BaseModel):
    """Functional correctness validation results"""
    
    # Reference solution comparison
    reference_solution_similarity: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Output similarity to reference (0.0-1.0)"
    )
    
    # Test case results
    test_cases_passed: int = Field(ge=0)
    test_cases_total: int = Field(ge=1)
    test_case_pass_rate: float = Field(ge=0.0, le=1.0)
    
    # Combined score
    correctness_score: float = Field(
        ge=0.0, le=1.0,
        description="Weighted: 60% tests + 40% reference"
    )
    
    # Detailed results
    test_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-test execution details"
    )
```

#### 2.3.2 SemanticClosenessMetrics

```python
class SemanticClosenessMetrics(BaseModel):
    """Code BLEU: Multi-level semantic similarity"""
    
    # 1. N-gram Match (standard BLEU)
    ngram_match_score: float = Field(ge=0.0, le=1.0)
    ngram_precision: Dict[int, float] = Field(
        description="Precision for 1-4 grams"
    )
    
    # 2. Weighted N-gram Match
    weighted_ngram_score: float = Field(ge=0.0, le=1.0)
    keyword_weights: Dict[str, float] = Field(
        default_factory=lambda: {"__default__": 1.0}
    )
    
    # 3. AST Match
    ast_match_score: float = Field(ge=0.0, le=1.0)
    ast_node_overlap: float = Field(ge=0.0, le=1.0)
    ast_depth_similarity: float = Field(ge=0.0, le=1.0)
    
    # 4. Dataflow Match
    dataflow_match_score: float = Field(ge=0.0, le=1.0)
    variable_flow_similarity: float = Field(ge=0.0, le=1.0)
    dependency_graph_similarity: float = Field(ge=0.0, le=1.0)
    
    # Combined Code BLEU
    code_bleu_score: float = Field(ge=0.0, le=1.0)
    weights: Dict[str, float] = Field(
        default={
            "ngram": 0.25,
            "weighted_ngram": 0.25,
            "ast": 0.25,
            "dataflow": 0.25
        }
    )
```

#### 2.3.3 CodeMetrics (Container)

```python
class CodeMetrics(BaseModel):
    """Comprehensive code quality metrics"""
    
    # 1. Functional Correctness
    functional_correctness: FunctionalCorrectnessMetrics
    
    # 2. Pass@k
    pass_at_k: Dict[int, float] = Field(
        description="Pass@1, Pass@5, Pass@10"
    )
    
    # 3. Test Pass Rate
    test_pass_rate: float = Field(ge=0.0, le=1.0)
    
    # 4. Semantic Closeness
    semantic_closeness: Optional[SemanticClosenessMetrics] = Field(
        None,
        description="Optional - for offline evaluation"
    )
    
    # Overall quality score
    overall_score: float = Field(
        ge=0.0, le=1.0,
        description="Weighted combination of all metrics"
    )
```

---

### 2.4 ValidationReport Model

```python
class ValidationReport(BaseModel):
    """Tool validation results"""
    
    # Basic gates
    schema_ok: bool = Field(description="Parameter/return types valid")
    tests_ok: bool = Field(description="All tests passed")
    sandbox_ok: bool = Field(description="Executed without errors")
    
    # Logs and errors
    logs: List[str] = Field(default_factory=list)
    errors: Optional[List[str]] = None
    
    # Code quality metrics (optional)
    code_metrics: Optional[CodeMetrics] = None
    
    @property
    def is_valid(self) -> bool:
        """Overall validation status"""
        return self.schema_ok and self.tests_ok and self.sandbox_ok
```

---

### 2.5 RunArtifacts Model

```python
class RunArtifacts(BaseModel):
    """Tool execution results"""
    
    # Structured output (preferred)
    result: Dict[str, Any] = Field(
        description="Main result as JSON-serializable dict"
    )
    
    # Display output
    summary_markdown: Optional[str] = Field(
        None,
        description="Human-readable summary"
    )
    
    # Legacy format (backward compatibility)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    
    # Additional data
    files: List[str] = Field(
        default_factory=list,
        description="Generated file paths"
    )
    sample_rows: Optional[List[Dict]] = None
    artifacts: List[str] = Field(default_factory=list)
    
    # Execution metadata
    execution_time_ms: float = Field(ge=0.0)
    rows_processed: Optional[int] = Field(None, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

---

### 2.6 UserFeedback Model

```python
from datetime import datetime

class UserFeedback(BaseModel):
    """User approval/rejection decision"""
    
    decision: str = Field(
        description="OUTPUT_ACCEPTED | OUTPUT_REJECTED | APPROVED | REJECTED"
    )
    notes: Optional[str] = Field(None, description="User comments")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )
    
    @property
    def is_approved(self) -> bool:
        return self.decision == "APPROVED"
    
    @property
    def is_rejected(self) -> bool:
        return self.decision == "REJECTED"
```

---

### 2.7 ToolCandidate Model (Master)

```python
import hashlib

class ToolCandidate(BaseModel):
    """Complete tool candidate with all metadata"""
    
    # Identity
    tool_id: str = Field(description="Unique tool identifier")
    version: str = Field(default="1.0.0", pattern=r"^\d+\.\d+\.\d+$")
    
    # Specification
    spec: ToolSpec
    
    # Implementation
    package_path: str = Field(description="Path to tool package")
    code_hash: str = Field(description="SHA256 of implementation")
    spec_hash: str = Field(description="SHA256 of spec JSON")
    
    # Lifecycle
    status: ToolStatus = Field(default=ToolStatus.DRAFT)
    
    # Validation
    validation_report: Optional[ValidationReport] = None
    
    # Execution
    run_artifacts: Optional[RunArtifacts] = None
    
    # User feedback
    user_feedback: Optional[UserFeedback] = None
    
    # Provenance
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )
    created_by: str = Field(description="Model/prompt hash")
    dependencies: List[str] = Field(default_factory=list)
    
    @staticmethod
    def compute_hash(content: str) -> str:
        """Compute SHA256 hash"""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def update_hashes(self, code: str, spec_json: str):
        """Update code and spec hashes"""
        self.code_hash = self.compute_hash(code)
        self.spec_hash = self.compute_hash(spec_json)
```

---

### 2.8 RegistryMetadata Model

```python
class ToolRegistryEntry(BaseModel):
    """Single tool entry in registry"""
    version: str
    path: str
    spec_hash: str
    code_hash: str
    promoted_at: str
    created_by: str

class RegistryMetadata(BaseModel):
    """Registry metadata file structure"""
    
    tools: Dict[str, ToolRegistryEntry] = Field(
        default_factory=dict,
        description="tool_id -> entry mapping"
    )
    last_updated: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )
    schema_version: str = Field(default="1.0.0")
```

---

## 3. Helper Types

### 3.1 TaskComplexity Enum

```python
class TaskComplexity(Enum):
    """Tool generation complexity levels"""
    SIMPLE = "simple"        # Basic groupby, filter
    MODERATE = "moderate"    # Multi-step transformations
    COMPLEX = "complex"      # Custom logic, edge cases
```

### 3.2 Quality Tier Type

```python
from typing import Literal

QualityTier = Literal[
    "EXCELLENT",      # ≥0.90 - Auto-stage high priority
    "GOOD",           # ≥0.75 - Auto-stage normal priority
    "ACCEPTABLE",     # ≥0.60 - Auto-stage with warnings
    "POOR"            # <0.60 - Reject or repair
]
```

---

## 4. Model Serialization

### 4.1 Save/Load Functions

```python
import json
from pathlib import Path

def save_tool_candidate(candidate: ToolCandidate, path: Path):
    """Save candidate to JSON"""
    with open(path / "candidate.json", "w") as f:
        json.dump(candidate.model_dump(), f, indent=2)

def load_tool_candidate(path: Path) -> ToolCandidate:
    """Load candidate from JSON"""
    with open(path / "candidate.json") as f:
        data = json.load(f)
    return ToolCandidate(**data)

def save_registry_metadata(metadata: RegistryMetadata, path: Path):
    """Save registry metadata"""
    with open(path / "metadata.json", "w") as f:
        json.dump(metadata.model_dump(), f, indent=2)
```

---

## 5. Validation Rules

### 5.1 Custom Validators

```python
from pydantic import field_validator

class ToolSpec(BaseModel):
    # ... fields ...
    
    @field_validator('tool_name')
    @classmethod
    def validate_tool_name(cls, v: str) -> str:
        """Ensure tool name is snake_case"""
        if not v.islower() or ' ' in v:
            raise ValueError("Tool name must be lowercase snake_case")
        if not v.replace('_', '').isalnum():
            raise ValueError("Tool name must be alphanumeric + underscores")
        return v
    
    @field_validator('input_schema')
    @classmethod
    def validate_json_schema(cls, v: Dict) -> Dict:
        """Validate JSON Schema structure"""
        if 'type' not in v:
            raise ValueError("Schema must have 'type' field")
        return v
```

---

## 6. Testing Requirements

### 6.1 Unit Tests

```python
# tests/test_models.py

def test_tool_status_enum():
    assert ToolStatus.DRAFT.value == "DRAFT"
    assert ToolStatus.PROMOTED.value == "PROMOTED"

def test_tool_spec_validation():
    # Valid spec
    spec = ToolSpec(
        tool_name="test_tool",
        description="Test",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        parameters=[],
        when_to_use="Test",
        what_it_does="Test",
        returns="Test",
        prerequisites="None"
    )
    assert spec.tool_name == "test_tool"
    
    # Invalid tool name (uppercase)
    with pytest.raises(ValidationError):
        ToolSpec(tool_name="TestTool", ...)

def test_tool_candidate_hashes():
    code = "def foo(): pass"
    spec_json = '{"tool_name": "foo"}'
    
    candidate = ToolCandidate(...)
    candidate.update_hashes(code, spec_json)
    
    assert len(candidate.code_hash) == 64  # SHA256
    assert candidate.code_hash != candidate.spec_hash

def test_validation_report_is_valid():
    report = ValidationReport(
        schema_ok=True,
        tests_ok=True,
        sandbox_ok=True
    )
    assert report.is_valid
    
    report.tests_ok = False
    assert not report.is_valid

def test_code_metrics_bounds():
    metrics = CodeMetrics(
        functional_correctness=...,
        pass_at_k={1: 1.5},  # Invalid: >1.0
        ...
    )
    # Should raise ValidationError
```

---

## 7. Implementation Checklist

- [ ] Define all enums (ToolStatus, TaskComplexity)
- [ ] Implement ToolSpec with validators
- [ ] Implement code metrics models (FunctionalCorrectnessMetrics, SemanticClosenessMetrics, CodeMetrics)
- [ ] Implement ValidationReport
- [ ] Implement RunArtifacts with structured output
- [ ] Implement UserFeedback
- [ ] Implement ToolCandidate master model
- [ ] Implement RegistryMetadata
- [ ] Add custom validators for tool_name, schemas
- [ ] Add hash computation methods
- [ ] Add serialization helpers (save/load)
- [ ] Write comprehensive unit tests (>95% coverage)
- [ ] Add Pydantic examples in docstrings
- [ ] Generate JSON schemas for all models

---

## 8. Dependencies

```python
# requirements.txt (for this module)
pydantic>=2.5.0
typing-extensions>=4.8.0
```

---

## 9. Module Output

**Files Created**:
- `src/models.py` - All data models
- `tests/test_models.py` - Unit tests
- `schemas/` - Generated JSON schemas for external validation

**Exports**:
```python
# src/models.py
__all__ = [
    "ToolStatus",
    "ToolSpec",
    "FunctionalCorrectnessMetrics",
    "SemanticClosenessMetrics",
    "CodeMetrics",
    "ValidationReport",
    "RunArtifacts",
    "UserFeedback",
    "ToolCandidate",
    "RegistryMetadata",
    "TaskComplexity",
    "QualityTier",
]
```

---

**Status**: Ready for Implementation  
**Priority**: P0 (Required by all other modules)  
**Estimated Effort**: 2-3 days
