# MCP Tool Code Interpreter Generator - Project Requirements

## 1. Project Overview

### 1.1 Purpose
Build an agentic code-generation pipeline that transforms natural-language analysis intents into executable MCP (Model Context Protocol) tools for data analysis workflows. The system ensures tool quality through staged validation and promotion based on explicit user feedback.

### 1.2 Core Principles
- **Contract-First Design**: Generate tool specifications with schemas before implementation
- **Staging Before Registration**: Never auto-register tools; promote only after user approval
- **Quality Gates**: Multi-stage validation (schema, sandbox, execution) with iterative repair
- **Prevent Tool Pollution**: Strict acceptance criteria to avoid low-quality tools in the registry

### 1.3 Repository Description
MCP_Tool_Code_Interpreter_Generator is a modular, agentic code-generation pipeline that turns natural-language analysis intents into executable MCP tools for data analysis workflows. It performs intent extraction, tool retrieval and gap detection, contract-first tool specification (schemas), code/package scaffolding, and sandbox validation with iterative repair. Newly generated tools are staged by default and promoted to the active MCP registry only after explicit positive user feedback on the tool's output, ensuring tool quality and preventing registry pollution.

---

## 2. System Architecture

### 2.1 High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                    User Request / Intent                     │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  1. Intent Extraction & Gap Detection                       │
│     - Parse user request                                     │
│     - Query active tool registry                             │
│     - Identify missing capabilities                          │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Tool Specification Generation (Contract-First)          │
│     - Generate ToolSpec with schemas                         │
│     - Define inputs (with Annotated types)                   │
│     - Define outputs and return types                        │
│     - Document WHEN/WHAT/RETURNS/PREREQUISITES               │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Code Generation & Scaffolding                           │
│     - Generate implementation from spec                      │
│     - Create package structure                               │
│     - Add helper functions and dependencies                  │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Validation & Repair Loop                                │
│     - Schema validation                                      │
│     - Sandbox testing (isolated environment)                 │
│     - Iterative repair if validation fails                   │
│     - Status: DRAFT → STAGED                                 │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  5. Staged Execution (Test Run)                             │
│     - Execute tool with user's current input/data            │
│     - Capture output artifacts                               │
│     - Record execution metadata                              │
│     - Keep tool isolated in staging registry                 │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  6. Present Output to User                                  │
│     - Show primary output (tables/metrics/artifacts)         │
│     - Display tool summary (assumptions, limitations)        │
│     - Request explicit approval/rejection                    │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  7. Feedback Capture & Decision Gate                        │
│     - Parse user feedback                                    │
│     - Accept: "yes", "approved", "looks good", etc.          │
│     - Reject: anything else (including silence/ambiguity)    │
└─────────────────┬───────────────────────────────────────────┘
                  │
           ┌──────┴──────┐
           │             │
      APPROVED      REJECTED
           │             │
           ▼             ▼
┌──────────────┐  ┌──────────────────┐
│  8. Promote  │  │  8. Discard or   │
│  & Register  │  │  Repair Loop     │
│              │  │                  │
│  STAGED →    │  │  - Capture issues│
│  PROMOTED    │  │  - Refine spec   │
│              │  │  - Regenerate    │
└──────────────┘  └──────────────────┘
```

### 2.2 Tool Lifecycle States

```
DRAFT → STAGED → APPROVED → PROMOTED (registered in active MCP)
                     ↓
                 REJECTED (stored for learning, not registered)
```

---

## 3. Data Models

### 3.1 ToolCandidate

```python
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class ToolStatus(str, Enum):
    DRAFT = "DRAFT"
    STAGED = "STAGED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    PROMOTED = "PROMOTED"

class ValidationReport(BaseModel):
    schema_ok: bool
    tests_ok: bool
    sandbox_ok: bool
    logs: List[str]
    errors: Optional[List[str]] = None
    
    # Code Quality Metrics
    code_metrics: Optional["CodeMetrics"] = None

class CodeMetrics(BaseModel):
    """Comprehensive code quality and correctness metrics"""
    
    # 1. Functional Correctness
    functional_correctness: "FunctionalCorrectnessMetrics"
    
    # 2. Pass@k Metrics
    pass_at_k: Dict[int, float]  # {1: 0.85, 5: 0.92, 10: 0.95}
    
    # 3. Test Pass Rate
    test_pass_rate: float  # 0.0 to 1.0
    
    # 4. Semantic Closeness (Code BLEU)
    semantic_closeness: "SemanticClosenessMetrics"
    
    # Overall quality score (weighted combination)
    overall_score: float  # 0.0 to 1.0

class FunctionalCorrectnessMetrics(BaseModel):
    """Measures how well the generated code performs the intended task"""
    
    # Reference solution comparison
    reference_solution_similarity: Optional[float] = None  # 0.0 to 1.0
    
    # Test case results
    test_cases_passed: int
    test_cases_total: int
    test_case_pass_rate: float  # test_cases_passed / test_cases_total
    
    # Combined correctness score
    correctness_score: float  # Weighted combination of reference + test cases
    
    # Detailed test results
    test_results: List[Dict[str, Any]]  # [{"test_id": "...", "passed": True, ...}]

class SemanticClosenessMetrics(BaseModel):
    """Code BLEU: Multi-level code similarity measurement"""
    
    # 1. N-gram Match (standard BLEU)
    ngram_match_score: float  # BLEU score (0.0 to 1.0)
    ngram_precision: Dict[int, float]  # {1: 0.8, 2: 0.7, 3: 0.6, 4: 0.5}
    
    # 2. Weighted N-gram Match
    weighted_ngram_score: float  # Weighted by keyword importance (0.0 to 1.0)
    keyword_weights: Dict[str, float]  # {"def": 2.0, "return": 1.5, ...}
    
    # 3. AST (Abstract Syntax Tree) Match
    ast_match_score: float  # Structural similarity (0.0 to 1.0)
    ast_node_overlap: float  # Percentage of matching AST nodes
    ast_depth_similarity: float  # Tree depth similarity
    
    # 4. Dataflow Match
    dataflow_match_score: float  # Variable usage patterns (0.0 to 1.0)
    variable_flow_similarity: float  # How variables flow through code
    dependency_graph_similarity: float  # Variable dependency matching
    
    # Combined Code BLEU Score
    code_bleu_score: float  # Weighted average of all components
    weights: Dict[str, float] = {  # Component weights
        "ngram": 0.25,
        "weighted_ngram": 0.25,
        "ast": 0.25,
        "dataflow": 0.25
    }

class RunArtifacts(BaseModel):
    # Structured output (preferred)
    result: Dict[str, Any]  # Main result as JSON-serializable dict
    
    # Legacy/display output
    outputs: Dict[str, Any]  # Can include markdown, tables, metrics
    summary_markdown: Optional[str] = None  # Human-readable summary
    
    # Additional data
    files: List[str]  # Generated file paths
    sample_rows: Optional[List[Dict]] = None
    artifacts: List[str] = []  # Paths to output artifacts
    
    # Execution metadata
    execution_time_ms: float
    rows_processed: Optional[int] = None
    metadata: Dict[str, Any] = {}

# Example usage:
# artifacts = RunArtifacts(
#     result={"grouped_data": [...], "total_groups": 5},
#     summary_markdown="## Results\n\n| Group | Count |\n...",
#     outputs={"markdown_table": "..."},  # backward compat
#     execution_time_ms=45.2,
#     metadata={"assumptions": [...], "limitations": [...]}
# )

class UserFeedback(BaseModel):
    decision: str  # "APPROVED" | "REJECTED" | "PENDING"
    notes: Optional[str] = None
    timestamp: str

class ToolSpec(BaseModel):
    tool_name: str
    description: str
    
    # Formal schemas (machine-readable)
    input_schema: Dict[str, Any]  # JSON Schema for input validation
    output_schema: Dict[str, Any]  # JSON Schema for output structure
    
    # Legacy/detailed parameter info
    parameters: List[Dict[str, Any]]  # Annotated parameter definitions
    return_type: str
    
    # Documentation (human-readable)
    when_to_use: str
    what_it_does: str
    returns: str
    prerequisites: str

# Example with schemas:
# input_schema = {
#     "type": "object",
#     "properties": {
#         "file_path": {"type": "string", "description": "Path to CSV file"},
#         "group_by_columns": {"type": "array", "items": {"type": "string"}}
#     },
#     "required": ["file_path", "group_by_columns"]
# }
#
# output_schema = {
#     "type": "object",
#     "properties": {
#         "result": {"type": "string"},  # markdown table
#         "metadata": {"type": "object"},
#         "artifacts": {"type": "array"}
#     },
#     "required": ["result"]
# }

class ToolCandidate(BaseModel):
    tool_id: str
    version: str = "1.0.0"
    spec: ToolSpec
    package_path: str
    code_hash: str  # SHA256 of implementation
    spec_hash: str  # SHA256 of spec
    status: ToolStatus = ToolStatus.DRAFT
    validation_report: Optional[ValidationReport] = None
    run_artifacts: Optional[RunArtifacts] = None
    user_feedback: Optional[UserFeedback] = None
    created_at: str
    created_by: str  # Model/prompt hash
    dependencies: List[str] = []
```

### 3.2 Registry Structure

```
registry/
├── active/           # Promoted, production-ready tools (PROMOTED)
│   ├── metadata.json
│   └── tools/
│       ├── load_and_analyze_csv/
│       │   ├── tool.py
│       │   ├── spec.json
│       │   ├── tests.py
│       │   └── VERSION
│       └── ...
├── staging/          # Validated but not yet approved (STAGED)
│   ├── metadata.json
│   └── candidates/
│       └── <tool_id>_<version>/
│           ├── tool.py
│           ├── spec.json
│           ├── validation_report.json
│           └── run_artifacts.json
└── archive/          # Rejected or superseded tools
    └── ...
```

---

## 4. Detailed Workflow Steps

### Step 1: Intent Extraction & Gap Detection

**Input**: User's natural language request
**Output**: Decision (use existing tool vs. generate new tool)

**Process**:
1. Parse user intent using LLM
2. Extract:
   - Required data operations (filter, aggregate, visualize, etc.)
   - Input data source (CSV path, columns)
   - Expected output format
3. Query active tool registry
4. Perform semantic similarity search
5. Calculate capability overlap for each existing tool:
   ```python
   overlap_score = (
       0.30 * operation_match +      # 1.0 if exact match, 0.0 otherwise
       0.25 * input_compatibility +  # 1.0 if all inputs available
       0.20 * output_format_match +  # 1.0 if format matches
       0.15 * constraint_match +     # 1.0 if constraints satisfied
       0.10 * semantic_similarity    # 0.0-1.0 from embedding similarity
   )
   ```
6. Decision:
   - If overlap_score ≥ 0.85 for any existing tool → use it
   - Otherwise → proceed to tool generation

**Example**:
```
User: "Analyze traffic accidents by severity and location"
Intent: {
  operation: "group_by_and_aggregate",
  columns: ["severity", "location"],
  metrics: ["count"],
  input: "traffic_accidents.csv"
}
Gap: No existing tool for this specific grouping
Decision: GENERATE_NEW_TOOL
```

---

### Step 2: Tool Specification Generation

**Input**: Extracted intent, dataset schema
**Output**: ToolSpec (contract)

**Process**:
1. Generate tool name (descriptive, snake_case)
2. Define input parameters with Annotated types:
   ```python
   file_path: Annotated[str, Field(description="...")]
   group_by_columns: Annotated[List[str], Field(description="...")]
   ```
3. Define return type and structure
4. Write documentation sections:
   - **WHEN TO USE THIS TOOL**: Trigger conditions
   - **WHAT THIS TOOL DOES**: Step-by-step logic
   - **RETURNS**: Output format specification
   - **PREREQUISITES**: Required prior steps or data state

**Example ToolSpec**:
```json
{
  "tool_name": "group_and_count_by_columns",
  "description": "Group data by specified columns and count occurrences",
  "parameters": [
    {
      "name": "file_path",
      "type": "str",
      "annotation": "Annotated[str, Field(description='Path to CSV file')]",
      "required": true
    },
    {
      "name": "group_by_columns",
      "type": "List[str]",
      "annotation": "Annotated[List[str], Field(description='Columns to group by')]",
      "required": true
    }
  ],
  "return_type": "str",
  "when_to_use": "When you need to count occurrences across multiple grouping dimensions",
  "what_it_does": "Loads CSV, groups by specified columns, counts rows, returns markdown table",
  "returns": "Markdown table with grouped counts",
  "prerequisites": "CSV file must exist and contain specified columns"
}
```

---

### Step 3: Code Generation & Scaffolding

**Input**: ToolSpec
**Output**: Python package with implementation

**Process**:
1. Generate MCP tool decorator and function signature:
   ```python
   @mcp.tool()
   def tool_name(param1: Annotated[Type, Field(...)], ...) -> ReturnType:
       """Docstring from spec"""
   ```
2. Implement logic based on `what_it_does`:
   - Data loading (use shared utilities like `load_csv_data_with_types`)
   - Validation (use `validate_file_and_columns`)
   - Core transformation
   - Output formatting
3. Add error handling with try/except
4. Generate machine-readable JSON footer:
   ```python
   footer = {"path": file_path, "numeric_columns": [...], ...}
   report.append(f"\n<!--output_json:{json.dumps(footer)}-->")
   ```
5. Create package structure:
   ```
   staging/candidates/tool_name_v1.0.0/
   ├── tool.py          # Main implementation
   ├── spec.json        # ToolSpec
   ├── __init__.py
   ├── helpers.py       # Extracted helper functions
   ├── requirements.txt
   └── tests.py         # Auto-generated tests
   ```

**Code Quality Standards**:
- Follow example tool pattern (load_and_analyze_csv)
- Use pandas for data manipulation
- Return markdown-formatted strings
- Include JSON footer for downstream parsing
- Handle errors gracefully

---

### Step 4: Validation & Repair Loop

**Input**: Generated tool package
**Output**: ValidationReport + tool with status STAGED

**Validation Gates**:

#### 4.1 Schema Validation
- Verify parameter annotations match ToolSpec
- Check return type consistency
- Validate docstring completeness

#### 4.2 Static Analysis
- Run `mypy` for type checking
- Run `pylint` for code quality
- Check for security issues (no `eval`, `exec`, unsafe file operations)

#### 4.3 Sandbox Testing
- Create isolated Python environment
- Install dependencies
- Import tool module
- Test with synthetic data:
  ```python
  # Generate test CSV matching expected schema
  # Call tool with test data
  # Verify output format
  # Check for runtime errors
  ```

#### 4.4 Functional Correctness Validation

**Purpose**: Verify the generated tool produces correct outputs for given inputs

**Methods**:

##### 4.4.1 Reference Solution Comparison
If a reference implementation exists:
```python
def validate_against_reference(
    generated_tool: Callable,
    reference_tool: Callable,
    test_inputs: List[Dict]
) -> float:
    """Compare generated tool output with reference solution"""
    matches = 0
    for test_input in test_inputs:
        generated_output = generated_tool(**test_input)
        reference_output = reference_tool(**test_input)
        
        # Normalize outputs (remove whitespace, sort tables, etc.)
        if outputs_match(generated_output, reference_output):
            matches += 1
    
    return matches / len(test_inputs)
```

##### 4.4.2 Test Case Execution
Generate diverse test cases covering:
- **Edge cases**: Empty datasets, single row, missing columns
- **Normal cases**: Typical data patterns
- **Stress cases**: Large datasets, high cardinality

```python
test_cases = [
    {
        "name": "empty_dataset",
        "input": {"file_path": "empty.csv", "group_by": ["col1"]},
        "expected": "No data to analyze",
        "validation": lambda output: "No data" in output
    },
    {
        "name": "single_group",
        "input": {"file_path": "single.csv", "group_by": ["region"]},
        "expected_rows": 1,
        "validation": lambda output: count_table_rows(output) == 1
    },
    # ... more test cases
]

# Execute and score
test_results = []
for test in test_cases:
    try:
        output = generated_tool(**test["input"])
        passed = test["validation"](output)
        test_results.append({"test_id": test["name"], "passed": passed})
    except Exception as e:
        test_results.append({"test_id": test["name"], "passed": False, "error": str(e)})

test_pass_rate = sum(r["passed"] for r in test_results) / len(test_results)
```

##### 4.4.3 Combined Correctness Score
```python
def calculate_correctness_score(
    reference_similarity: Optional[float],
    test_pass_rate: float
) -> float:
    """Weighted combination of reference and test-based validation"""
    if reference_similarity is not None:
        # 60% test cases + 40% reference similarity
        return 0.6 * test_pass_rate + 0.4 * reference_similarity
    else:
        # Only test cases available
        return test_pass_rate
```

#### 4.5 Pass@k Metrics

**Purpose**: Measure probability that at least one of k generated solutions is correct

**Formula**:
```
pass@k = 1 - (C(n-c, k) / C(n, k))

where:
  n = total samples generated
  c = number of correct samples
  k = number of samples to consider
  C(n, k) = binomial coefficient
```

**Implementation**:
```python
import math
from typing import List

def calculate_pass_at_k(
    n: int,  # total generated candidates
    c: int,  # correct candidates
    k_values: List[int] = [1, 5, 10]
) -> Dict[int, float]:
    """Calculate pass@k for multiple k values"""
    
    def comb(n: int, k: int) -> int:
        """Binomial coefficient"""
        return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
    
    pass_at_k = {}
    for k in k_values:
        if n - c < k:
            pass_at_k[k] = 1.0
        else:
            pass_at_k[k] = 1.0 - (comb(n - c, k) / comb(n, k))
    
    return pass_at_k

# Example: Generate 10 candidate implementations
candidates = [generate_tool_variant(spec) for _ in range(10)]
correct_count = sum(1 for c in candidates if validate_tool(c))

# Calculate pass@1, pass@5, pass@10
pass_at_k_scores = calculate_pass_at_k(
    n=10,
    c=correct_count,
    k_values=[1, 5, 10]
)
# Result: {1: 0.85, 5: 0.92, 10: 0.95}
```

**Use Case**: When generating multiple candidate tools, pass@k tells you:
- **pass@1**: Probability the best candidate is correct
- **pass@5**: Probability at least one of top 5 is correct
- **pass@10**: Probability at least one of top 10 is correct

**Decision Threshold**: Promote tool if `pass@5 > 0.9`

#### 4.6 Test Pass Rate

**Purpose**: Overall percentage of test cases that pass

**Calculation**:
```python
def calculate_test_pass_rate(
    test_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Calculate comprehensive test pass rate metrics"""
    
    total_tests = len(test_results)
    passed_tests = sum(1 for r in test_results if r["passed"])
    
    # Overall pass rate
    overall_rate = passed_tests / total_tests if total_tests > 0 else 0.0
    
    # Pass rate by category
    categories = {}
    for result in test_results:
        category = result.get("category", "general")
        if category not in categories:
            categories[category] = {"total": 0, "passed": 0}
        categories[category]["total"] += 1
        if result["passed"]:
            categories[category]["passed"] += 1
    
    category_rates = {
        cat: stats["passed"] / stats["total"]
        for cat, stats in categories.items()
    }
    
    return {
        "overall_pass_rate": overall_rate,
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": total_tests - passed_tests,
        "category_pass_rates": category_rates
    }
```

**Example Test Categories**:
- **edge_cases**: 95% pass rate required
- **normal_cases**: 98% pass rate required
- **stress_cases**: 85% pass rate required

**Acceptance Criteria**:
```python
if test_pass_rate_metrics["overall_pass_rate"] >= 0.95:
    print("Tool passes test validation")
else:
    print("Tool requires repair")
```

#### 4.7 Semantic Closeness Metrics (Code BLEU)

**Purpose**: Measure how semantically similar generated code is to reference implementations beyond surface-level text matching

**⚠️ Implementation Note**: Code BLEU metrics are **optional for MVP**. They provide valuable quality signals but are computationally expensive. For MVP, prioritize:
- Schema validation (required)
- Test execution success (required)
- Test pass rate (required)
- Functional correctness (required if reference solution exists)

Code BLEU components (n-gram, AST, dataflow) are recommended for:
- **Offline evaluation** and benchmarking
- **Research/quality analysis** of the code generator
- **Advanced validation** when reference solutions are available

For production use, implement as **async background scoring** that doesn't block staging.

##### 4.7.1 N-gram Match (Standard BLEU)

**Tokenize code and compute BLEU score**:
```python
import re
from collections import Counter
from typing import List, Dict

def tokenize_code(code: str) -> List[str]:
    """Tokenize Python code into meaningful units"""
    # Remove comments and docstrings
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r'#.*', '', code)
    
    # Split on whitespace and operators
    tokens = re.findall(r'\b\w+\b|[+\-*/=<>!]+|[()\[\]{}:,.]', code)
    return [t for t in tokens if t.strip()]

def calculate_ngram_precision(
    candidate_tokens: List[str],
    reference_tokens: List[str],
    n: int
) -> float:
    """Calculate n-gram precision"""
    def get_ngrams(tokens: List[str], n: int) -> Counter:
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    
    candidate_ngrams = get_ngrams(candidate_tokens, n)
    reference_ngrams = get_ngrams(reference_tokens, n)
    
    matches = sum(
        min(candidate_ngrams[ng], reference_ngrams[ng])
        for ng in candidate_ngrams
    )
    
    total = sum(candidate_ngrams.values())
    return matches / total if total > 0 else 0.0

def calculate_bleu(
    candidate_code: str,
    reference_code: str,
    max_n: int = 4
) -> Dict[str, float]:
    """Calculate BLEU score with multiple n-gram orders"""
    cand_tokens = tokenize_code(candidate_code)
    ref_tokens = tokenize_code(reference_code)
    
    precisions = {}
    for n in range(1, max_n + 1):
        precisions[n] = calculate_ngram_precision(cand_tokens, ref_tokens, n)
    
    # Geometric mean of precisions
    import math
    bleu_score = math.exp(
        sum(math.log(p) if p > 0 else -float('inf') for p in precisions.values()) / max_n
    )
    
    return {
        "ngram_precision": precisions,
        "bleu_score": bleu_score
    }
```

##### 4.7.2 Weighted N-gram Match

**Give higher weight to important keywords**:
```python
def calculate_weighted_ngram_score(
    candidate_code: str,
    reference_code: str,
    keyword_weights: Dict[str, float]
) -> float:
    """BLEU with keyword importance weighting"""
    
    # Default weights for Python keywords
    default_weights = {
        # Control flow
        "def": 2.0, "class": 2.0, "return": 1.8,
        "if": 1.5, "else": 1.5, "elif": 1.5,
        "for": 1.7, "while": 1.7,
        
        # Data operations
        "import": 1.6, "from": 1.6,
        "pd": 2.0, "DataFrame": 1.8,  # pandas-specific
        "groupby": 2.0, "agg": 1.8, "merge": 1.8,
        
        # MCP-specific
        "@mcp": 2.5, "tool": 2.5, "Annotated": 2.0,
        
        # Error handling
        "try": 1.5, "except": 1.5, "raise": 1.4,
        
        # Default weight for other tokens
        "__default__": 1.0
    }
    
    weights = {**default_weights, **keyword_weights}
    
    cand_tokens = tokenize_code(candidate_code)
    ref_tokens = tokenize_code(reference_code)
    
    # Calculate weighted matches
    weighted_matches = 0.0
    total_weight = 0.0
    
    for token in cand_tokens:
        weight = weights.get(token, weights["__default__"])
        total_weight += weight
        if token in ref_tokens:
            weighted_matches += weight
    
    return weighted_matches / total_weight if total_weight > 0 else 0.0
```

##### 4.7.3 AST (Abstract Syntax Tree) Match

**Compare code structure, not just surface tokens**:
```python
import ast
from typing import Set, Tuple

def extract_ast_features(code: str) -> Dict[str, Any]:
    """Extract structural features from AST"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"valid": False}
    
    node_types = []
    node_depths = []
    
    def visit(node, depth=0):
        node_types.append(type(node).__name__)
        node_depths.append(depth)
        for child in ast.iter_child_nodes(node):
            visit(child, depth + 1)
    
    visit(tree)
    
    return {
        "valid": True,
        "node_types": node_types,
        "max_depth": max(node_depths) if node_depths else 0,
        "node_count": len(node_types),
        "unique_nodes": set(node_types)
    }

def calculate_ast_similarity(
    candidate_code: str,
    reference_code: str
) -> Dict[str, float]:
    """Calculate AST-based structural similarity"""
    
    cand_ast = extract_ast_features(candidate_code)
    ref_ast = extract_ast_features(reference_code)
    
    if not (cand_ast["valid"] and ref_ast["valid"]):
        return {
            "ast_match_score": 0.0,
            "ast_node_overlap": 0.0,
            "ast_depth_similarity": 0.0
        }
    
    # Node type overlap (Jaccard similarity)
    cand_nodes = set(cand_ast["node_types"])
    ref_nodes = set(ref_ast["node_types"])
    node_overlap = len(cand_nodes & ref_nodes) / len(cand_nodes | ref_nodes)
    
    # Depth similarity
    depth_diff = abs(cand_ast["max_depth"] - ref_ast["max_depth"])
    max_depth = max(cand_ast["max_depth"], ref_ast["max_depth"])
    depth_similarity = 1.0 - (depth_diff / max_depth) if max_depth > 0 else 1.0
    
    # Node sequence similarity (ordered)
    def sequence_similarity(seq1: List, seq2: List) -> float:
        # Longest common subsequence ratio
        from difflib import SequenceMatcher
        return SequenceMatcher(None, seq1, seq2).ratio()
    
    node_sequence_sim = sequence_similarity(
        cand_ast["node_types"],
        ref_ast["node_types"]
    )
    
    # Combined AST score
    ast_match_score = (
        0.4 * node_overlap +
        0.3 * depth_similarity +
        0.3 * node_sequence_sim
    )
    
    return {
        "ast_match_score": ast_match_score,
        "ast_node_overlap": node_overlap,
        "ast_depth_similarity": depth_similarity,
        "ast_node_sequence_similarity": node_sequence_sim
    }
```

##### 4.7.4 Dataflow Match

**Analyze variable usage patterns and dependencies**:
```python
import ast
from typing import Dict, Set, List, Tuple
from collections import defaultdict

class DataflowAnalyzer(ast.NodeVisitor):
    """Extract dataflow graph from Python code"""
    
    def __init__(self):
        self.variables: Dict[str, Set[str]] = defaultdict(set)  # var -> dependencies
        self.assignments: List[Tuple[str, int]] = []  # (var, line_no)
        self.usages: List[Tuple[str, int]] = []  # (var, line_no)
        self.current_dependencies: Set[str] = set()
    
    def visit_Assign(self, node):
        # Extract dependencies from RHS
        deps = set()
        for child in ast.walk(node.value):
            if isinstance(child, ast.Name):
                deps.add(child.id)
        
        # Record assignment
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                self.variables[var_name].update(deps)
                self.assignments.append((var_name, node.lineno))
        
        self.generic_visit(node)
    
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.usages.append((node.id, node.lineno))
        self.generic_visit(node)

def extract_dataflow(code: str) -> Dict[str, Any]:
    """Extract dataflow information from code"""
    try:
        tree = ast.parse(code)
        analyzer = DataflowAnalyzer()
        analyzer.visit(tree)
        
        return {
            "valid": True,
            "variable_dependencies": dict(analyzer.variables),
            "assignments": analyzer.assignments,
            "usages": analyzer.usages,
            "variable_count": len(analyzer.variables)
        }
    except SyntaxError:
        return {"valid": False}

def calculate_dataflow_similarity(
    candidate_code: str,
    reference_code: str
) -> Dict[str, float]:
    """Calculate dataflow-based similarity"""
    
    cand_flow = extract_dataflow(candidate_code)
    ref_flow = extract_dataflow(reference_code)
    
    if not (cand_flow["valid"] and ref_flow["valid"]):
        return {
            "dataflow_match_score": 0.0,
            "variable_flow_similarity": 0.0,
            "dependency_graph_similarity": 0.0
        }
    
    # Variable usage pattern similarity
    cand_vars = set(cand_flow["variable_dependencies"].keys())
    ref_vars = set(ref_flow["variable_dependencies"].keys())
    var_overlap = len(cand_vars & ref_vars) / len(cand_vars | ref_vars) if (cand_vars | ref_vars) else 0.0
    
    # Dependency graph similarity (for common variables)
    common_vars = cand_vars & ref_vars
    if common_vars:
        dep_similarities = []
        for var in common_vars:
            cand_deps = cand_flow["variable_dependencies"][var]
            ref_deps = ref_flow["variable_dependencies"][var]
            if cand_deps or ref_deps:
                dep_sim = len(cand_deps & ref_deps) / len(cand_deps | ref_deps)
                dep_similarities.append(dep_sim)
        
        dependency_graph_sim = sum(dep_similarities) / len(dep_similarities) if dep_similarities else 0.0
    else:
        dependency_graph_sim = 0.0
    
    # Assignment/usage order similarity
    def flow_order_similarity(flow1, flow2, key):
        seq1 = [v for v, _ in flow1[key]]
        seq2 = [v for v, _ in flow2[key]]
        from difflib import SequenceMatcher
        return SequenceMatcher(None, seq1, seq2).ratio()
    
    assignment_order_sim = flow_order_similarity(cand_flow, ref_flow, "assignments")
    
    # Combined dataflow score
    dataflow_match_score = (
        0.4 * var_overlap +
        0.4 * dependency_graph_sim +
        0.2 * assignment_order_sim
    )
    
    return {
        "dataflow_match_score": dataflow_match_score,
        "variable_flow_similarity": var_overlap,
        "dependency_graph_similarity": dependency_graph_sim,
        "assignment_order_similarity": assignment_order_sim
    }
```

##### 4.7.5 Combined Code BLEU Score

**Weighted combination of all semantic metrics**:
```python
def calculate_code_bleu(
    candidate_code: str,
    reference_code: str,
    weights: Dict[str, float] = None
) -> "SemanticClosenessMetrics":
    """Calculate comprehensive Code BLEU score"""
    
    if weights is None:
        weights = {
            "ngram": 0.25,
            "weighted_ngram": 0.25,
            "ast": 0.25,
            "dataflow": 0.25
        }
    
    # 1. N-gram match
    bleu_results = calculate_bleu(candidate_code, reference_code)
    ngram_score = bleu_results["bleu_score"]
    ngram_precision = bleu_results["ngram_precision"]
    
    # 2. Weighted n-gram
    weighted_ngram_score = calculate_weighted_ngram_score(
        candidate_code,
        reference_code,
        keyword_weights={}
    )
    
    # 3. AST match
    ast_results = calculate_ast_similarity(candidate_code, reference_code)
    ast_score = ast_results["ast_match_score"]
    
    # 4. Dataflow match
    dataflow_results = calculate_dataflow_similarity(candidate_code, reference_code)
    dataflow_score = dataflow_results["dataflow_match_score"]
    
    # Combined Code BLEU
    code_bleu_score = (
        weights["ngram"] * ngram_score +
        weights["weighted_ngram"] * weighted_ngram_score +
        weights["ast"] * ast_score +
        weights["dataflow"] * dataflow_score
    )
    
    return SemanticClosenessMetrics(
        ngram_match_score=ngram_score,
        ngram_precision=ngram_precision,
        weighted_ngram_score=weighted_ngram_score,
        keyword_weights={"__default__": 1.0},
        ast_match_score=ast_score,
        ast_node_overlap=ast_results["ast_node_overlap"],
        ast_depth_similarity=ast_results["ast_depth_similarity"],
        dataflow_match_score=dataflow_score,
        variable_flow_similarity=dataflow_results["variable_flow_similarity"],
        dependency_graph_similarity=dataflow_results["dependency_graph_similarity"],
        code_bleu_score=code_bleu_score,
        weights=weights
    )
```

#### 4.8 Overall Code Quality Scoring

**Combine all metrics into a single quality score**:
```python
def calculate_overall_quality_score(
    functional_correctness: float,
    pass_at_k: Dict[int, float],
    test_pass_rate: float,
    code_bleu: float,
    weights: Dict[str, float] = None
) -> float:
    """Calculate weighted overall quality score"""
    
    if weights is None:
        weights = {
            "functional_correctness": 0.40,  # Most important
            "test_pass_rate": 0.30,
            "code_bleu": 0.20,
            "pass_at_1": 0.10
        }
    
    score = (
        weights["functional_correctness"] * functional_correctness +
        weights["test_pass_rate"] * test_pass_rate +
        weights["code_bleu"] * code_bleu +
        weights["pass_at_1"] * pass_at_k.get(1, 0.0)
    )
    
    return score

# Quality thresholds
QUALITY_THRESHOLDS = {
    "excellent": 0.90,  # Promote immediately
    "good": 0.75,       # Promote with user approval
    "acceptable": 0.60, # Requires user review
    "poor": 0.0         # Reject or repair
}

def evaluate_tool_quality(overall_score: float) -> str:
    """Classify tool quality based on score"""
    if overall_score >= QUALITY_THRESHOLDS["excellent"]:
        return "EXCELLENT - Auto-stage (ready for user approval)"
    elif overall_score >= QUALITY_THRESHOLDS["good"]:
        return "GOOD - Stage and request approval"
    elif overall_score >= QUALITY_THRESHOLDS["acceptable"]:
        return "ACCEPTABLE - Requires manual review"
    else:
        return "POOR - Reject or enter repair loop"

# IMPORTANT: Quality metrics determine STAGING readiness only.
# Only explicit user approval can promote STAGED → APPROVED → PROMOTED.
```

#### 4.9 Repair Loop (with Code Metrics)
If any validation fails:
1. Collect comprehensive diagnostic information:
   ```python
   diagnostics = {
       "schema_errors": [...],
       "runtime_errors": [...],
       "code_metrics": {
           "functional_correctness": 0.65,
           "test_pass_rate": 0.70,
           "code_bleu": 0.55,
           "failed_tests": ["edge_case_empty_data", "stress_large_dataset"]
       }
   }
   ```

2. Feed errors and metrics back to LLM with enhanced prompt:
   ```
   The tool failed validation with the following issues:
   
   SCHEMA ERRORS:
   - Parameter 'x' type mismatch: expected str, got int
   
   RUNTIME ERRORS:
   - KeyError on column 'y' (line 45)
   
   CODE QUALITY METRICS:
   - Functional Correctness: 65% (target: >90%)
   - Test Pass Rate: 70% (14/20 tests passed)
   - Code BLEU: 0.55 (AST match low: 0.40)
   
   FAILED TESTS:
   1. edge_case_empty_data: Expected "No data" message, got KeyError
   2. stress_large_dataset: Timeout after 30s
   
   IMPROVEMENT SUGGESTIONS:
   - Add validation for empty dataframes before grouping
   - Optimize groupby operation for large datasets (use chunking)
   - Match reference solution's AST structure more closely
   
   Fix the implementation while preserving the ToolSpec contract.
   Target: >90% correctness, >95% test pass rate, >0.75 Code BLEU
   ```

3. Regenerate code with targeted improvements
4. Re-validate with full metrics (max 3 iterations)
5. Track improvement delta:
   ```python
   improvement = {
       "iteration": 2,
       "correctness": {"before": 0.65, "after": 0.88, "delta": +0.23},
       "test_pass_rate": {"before": 0.70, "after": 0.95, "delta": +0.25},
       "code_bleu": {"before": 0.55, "after": 0.78, "delta": +0.23}
   }
   ```
6. If still failing after 3 attempts → mark as REJECTED with detailed report

**Success Criteria** (Enhanced):
- ✅ All validation gates pass
- ✅ Functional correctness score ≥ 0.90
- ✅ Test pass rate ≥ 0.95
- ✅ Code BLEU ≥ 0.75 (if reference solution available)
- ✅ pass@5 ≥ 0.90 (if multiple candidates generated)
- ✅ Tool runs without errors on synthetic data
- ✅ Output matches expected format
- ✅ Status updated to STAGED

**Quality-Based Routing** (determines staging readiness, NOT auto-promotion):
```python
if overall_quality_score >= 0.90:
    status = ToolStatus.STAGED  # Excellent - ready for user review
    staging_priority = "HIGH"   # Fast-track approval request
elif overall_quality_score >= 0.75:
    status = ToolStatus.STAGED  # Good - ready for user review
    staging_priority = "NORMAL"
elif overall_quality_score >= 0.60:
    status = ToolStatus.STAGED  # Acceptable - flag for careful review
    staging_priority = "LOW"
    add_quality_warnings()      # Warn user about marginal quality
elif iteration < 3:
    enter_repair_loop()         # Attempt improvement
else:
    status = ToolStatus.REJECTED  # Failed after max attempts

# CRITICAL: STAGED ≠ PROMOTED
# Only user approval can transition STAGED → APPROVED → PROMOTED
# Quality score affects presentation priority, not promotion decision
```

---

### Step 5: Staged Execution (Test Run)

**Input**: STAGED tool + user's actual data/request
**Output**: RunArtifacts

**Process**:
1. Load tool from staging registry (isolated from active registry)
2. Execute tool with user's original input:
   ```python
   result = staged_tool.run(
       file_path="traffic_accidents.csv",
       group_by_columns=["severity", "location"]
   )
   ```
3. Capture execution metadata:
   ```python
   artifacts = RunArtifacts(
       outputs={"markdown_table": result},
       files=[],
       execution_time_ms=elapsed_ms,
       rows_processed=df_row_count,
       metadata={
           "assumptions": ["Assumes columns exist", ...],
           "limitations": ["Does not handle missing values in group columns"]
       }
   )
   ```
4. Store artifacts in `staging/candidates/<tool_id>/run_artifacts.json`

**Isolation**:
- Tool is NOT visible in normal tool retrieval
- Callable only by:
  - Current session
  - Validation harness
- Does NOT affect active registry

---

### Step 6: Present Output to User

**Input**: RunArtifacts
**Output**: User-facing presentation

**Format**:
```markdown
## Analysis Results

[Primary output - table/chart/metrics]

---

### Tool Summary
**Tool Name**: group_and_count_by_columns
**What It Did**: Grouped traffic_accidents.csv by severity and location, counted occurrences
**Rows Processed**: 1,234
**Execution Time**: 45ms

**Assumptions**:
- Columns 'severity' and 'location' exist in the dataset
- No null values in grouping columns

**Limitations**:
- Does not handle missing values (rows with nulls are excluded)
- Output limited to top 100 groups

---

### ⚠️ Approval Required
This tool is **staged** and not yet registered in your MCP tool registry.

**Two-Step Approval Process**:

1. **Is this output correct for your current analysis?**
   Reply: `Yes` or `No`

2. **Should I save this as a reusable tool for future analyses?**
   Reply: `Approve` (register tool) or `Reject` (don't register)

**Note**: You can accept the output without registering the tool.
If you reject the tool, please specify what needs improvement (incorrect columns, wrong metric, missing filter, etc.)
```

**Key Elements**:
1. Primary output first (user's requested result)
2. Transparent tool summary
3. Clear assumptions and limitations
4. Explicit, unambiguous approval request
5. Guidance on rejection feedback

---

### Step 7: Feedback Capture & Decision Gate

**Input**: User's response
**Output**: UserFeedback + decision (APPROVED/REJECTED)

**Acceptance Criteria (Strict)**:

**Two-Stage Approval**:

**Stage 1: Output Validation**
- User answers: "Yes" or "No" to "Is this output correct?"
- This validates the current result only

**Stage 2: Tool Registration** (only if output accepted)
- Require exact token: `Approve` or `Reject`
- Case-insensitive, but must be standalone or first word
- Any other response → default to `Reject`

**APPROVE if**:
- Response contains standalone "approve" or "approved"
- First word is "approve" (case-insensitive)
- Examples: "Approve", "approve it", "Approved!"

**REJECT for**:
- Response contains "reject" or "no"
- Ambiguous: "ok", "fine", "maybe", "sure"
- Mixed signals: "approve but...", "yes however..."
- Silence (no response after timeout)
- Any response not starting with "approve"

**Implementation**:
```python
def parse_feedback(user_response: str, stage: str = "registration") -> UserFeedback:
    """Parse user feedback with strict token matching"""
    response_clean = user_response.lower().strip()
    first_word = response_clean.split()[0] if response_clean else ""
    
    # Stage 1: Output validation (lenient)
    if stage == "output":
        if response_clean in ["yes", "y", "correct", "good"]:
            decision = "OUTPUT_ACCEPTED"
        else:
            decision = "OUTPUT_REJECTED"
        return UserFeedback(decision=decision, notes=user_response, timestamp=datetime.now().isoformat())
    
    # Stage 2: Tool registration (strict)
    # Check for explicit approval token
    if first_word in ["approve", "approved"]:
        # Check for disqualifying words
        disqualifiers = ["but", "however", "except", "although", "though"]
        if any(word in response_clean for word in disqualifiers):
            decision = "REJECTED"  # Mixed signals
        else:
            decision = "APPROVED"
    elif first_word in ["reject", "rejected", "no"]:
        decision = "REJECTED"
    else:
        # Anything else (including "ok", "sure", "maybe") → reject
        decision = "REJECTED"
    
    return UserFeedback(
        decision=decision,
        notes=user_response,
        timestamp=datetime.now().isoformat()
    )

# Usage:
# Step 1: output_feedback = parse_feedback(response, stage="output")
# Step 2: if output_feedback.decision == "OUTPUT_ACCEPTED":
#             tool_feedback = parse_feedback(response, stage="registration")
```

**Store Feedback**:
```python
candidate.user_feedback = feedback
if feedback.decision == "APPROVED":
    candidate.status = ToolStatus.APPROVED
else:
    candidate.status = ToolStatus.REJECTED
save_candidate(candidate)
```

---

### Step 8A: Promotion & Registration (If APPROVED)

**Input**: APPROVED ToolCandidate
**Output**: Tool registered in active MCP registry

**Process**:

#### 8.1 Pre-Promotion Checks
```python
def can_promote(candidate: ToolCandidate) -> Tuple[bool, str]:
    # Check if tool already exists
    existing = get_active_tool(candidate.tool_id)
    if existing:
        if existing.spec_hash == candidate.spec_hash:
            return False, "Identical tool already registered"
        else:
            # Version conflict - require explicit version bump
            return False, f"Tool exists as v{existing.version}. Increment version."
    
    # Verify candidate is approved
    if candidate.status != ToolStatus.APPROVED:
        return False, f"Tool must be APPROVED (current: {candidate.status})"
    
    return True, "OK"
```

#### 8.2 Promotion Steps
1. **Copy package**:
   ```
   staging/candidates/tool_v1.0.0/ 
     → active/tools/tool_v1.0.0/
   ```
2. **Update registry metadata**:
   ```json
   {
     "tools": {
       "group_and_count_by_columns": {
         "version": "1.0.0",
         "path": "active/tools/group_and_count_by_columns",
         "spec_hash": "abc123...",
         "code_hash": "def456...",
         "promoted_at": "2026-01-22T10:30:00Z",
         "created_by": "gpt-4-model-hash"
       }
     }
   }
   ```
3. **Update MCP server** (reload tools or restart server)
4. **Log promotion**:
   ```
   promotion_log.append({
       "tool_id": candidate.tool_id,
       "version": candidate.version,
       "promoted_at": datetime.now(),
       "approved_by": "user_session_id",
       "spec_hash": candidate.spec_hash
   })
   ```
5. **Update status**:
   ```python
   candidate.status = ToolStatus.PROMOTED
   archive_staged_candidate(candidate)
   ```

#### 8.3 Idempotency & Versioning
- If identical tool exists → skip promotion
- If spec changed → require version bump (v1.0.0 → v1.1.0 or v2.0.0)
- Keep promotion log to prevent duplicates

**Post-Promotion**:
- Tool now visible in normal tool retrieval
- Can be invoked in future analysis requests
- Added to MCP tool documentation/help

---

### Step 8B: Rejection & Repair (If REJECTED)

**Input**: REJECTED ToolCandidate + user feedback notes
**Output**: Refined ToolSpec → re-enter pipeline OR discard

**Process**:

#### 8.1 Parse Rejection Reasons
```python
def extract_issues(feedback_notes: str) -> List[str]:
    # Common patterns:
    # "wrong column: should be 'region' not 'location'"
    # "missing filter for year > 2020"
    # "output format should be JSON not markdown"
    
    issues = []
    if "wrong column" in feedback_notes:
        issues.append("COLUMN_MISMATCH")
    if "missing filter" in feedback_notes:
        issues.append("MISSING_FILTER")
    if "format" in feedback_notes:
        issues.append("OUTPUT_FORMAT")
    # ... more patterns
    return issues
```

#### 8.2 Decision: Repair or Discard
```python
if user_provides_specific_feedback:
    # Re-enter pipeline at Step 2 with refined intent
    refined_intent = apply_user_corrections(original_intent, feedback_notes)
    generate_tool_spec(refined_intent)  # Back to Step 2
else:
    # Generic rejection, store for learning
    archive_rejected_candidate(candidate)
```

#### 8.3 Archive Rejected Tools
```
archive/rejected/
└── <tool_id>_<timestamp>/
    ├── tool.py
    ├── spec.json
    ├── user_feedback.json
    └── metadata.json
```

**Learning from Rejections**:
- Collect common rejection patterns
- Use to refine prompt engineering
- Improve gap detection accuracy
- Build better example tools

---

## 5. Implementation Guidelines

### 5.1 Technology Stack

**Core**:
- **Python 3.11+**
- **FastMCP** or **MCP Python SDK** for tool server
- **Pydantic** for data models and validation
- **Pandas** for data manipulation

**LLM Integration**:
- **Cloud-based**: OpenAI API / Anthropic API for tool generation
- **On-Premises (Recommended for 48GB VRAM)**: Qwen2.5-Coder models (see Section 5.1.1)
- **Inference Engines**: vLLM, TGI (Text Generation Inference), or llama.cpp
- **Orchestration**: LangChain/LangGraph (optional)

**Validation**:
- `mypy` for type checking
- `pylint` for code quality
- `pytest` for automated testing

**Code Metrics & Evaluation**:
- `ast` (built-in) for Abstract Syntax Tree analysis
- `difflib` (built-in) for sequence matching
- Custom implementations for:
  - Code BLEU calculation
  - N-gram matching
  - Dataflow analysis
  - Pass@k metrics
- Optional: `tree-sitter` for advanced AST parsing

**Storage**:
- Filesystem-based registry (JSON metadata)
- SQLite for promotion logs (optional)

### 5.1.1 Recommended LLM Models for On-Premises Deployment (48GB VRAM)

#### **Qwen2.5-Coder Series (Highly Recommended)** ⭐

Alibaba's Qwen2.5-Coder models are specifically optimized for code generation and are excellent for this project.

**Model Options for 48GB VRAM:**

| Model | Parameters | Quantization | VRAM Usage | Quality | Speed | Recommended Use |
|-------|------------|--------------|------------|---------|-------|-----------------|
| **Qwen2.5-Coder-32B-Instruct** | 32B | 4-bit (GPTQ/AWQ) | ~20-24GB | ⭐⭐⭐⭐⭐ | Medium | **Primary choice** - Best quality/size ratio |
| **Qwen2.5-Coder-14B-Instruct** | 14B | FP16 | ~28GB | ⭐⭐⭐⭐ | Fast | Good balance, full precision |
| **Qwen2.5-Coder-14B-Instruct** | 14B | 8-bit | ~14GB | ⭐⭐⭐⭐ | Fast | Resource efficient |
| **Qwen2.5-Coder-7B-Instruct** | 7B | FP16 | ~14GB | ⭐⭐⭐ | Very Fast | Simple tools, fast iteration |
| **Qwen2.5-32B-Instruct** | 32B | 4-bit | ~20-24GB | ⭐⭐⭐⭐⭐ | Medium | Alternative for general reasoning |

**Recommended Configuration for 48GB:**

```yaml
# config.yaml - Qwen On-Premises Setup

llm:
  provider: "local"
  inference_engine: "vllm"  # or "tgi" or "llamacpp"
  base_url: "http://localhost:8000/v1"
  
  # Primary model for code generation
  code_generation_model:
    model_name: "Qwen/Qwen2.5-Coder-32B-Instruct"
    quantization: "awq"  # or "gptq" for 4-bit
    gpu_memory_utilization: 0.85
    temperature: 0.2
    max_tokens: 4096
    top_p: 0.95
    
  # Fallback for simple tasks (runs simultaneously)
  fallback_model:
    model_name: "Qwen/Qwen2.5-Coder-7B-Instruct"
    quantization: "fp16"
    gpu_memory_utilization: 0.15  # Uses remaining VRAM
    temperature: 0.2
    max_tokens: 2048

# Multi-GPU setup (if available)
multi_gpu:
  enabled: false  # Set to true if you have 2x 24GB GPUs
  tensor_parallel_size: 2  # Split model across GPUs
```

**Installation & Setup:**

```bash
# 1. Install vLLM (recommended inference engine)
pip install vllm

# 2. Download Qwen2.5-Coder-32B-Instruct (4-bit quantized)
# Option A: Using Hugging Face Hub
huggingface-cli download Qwen/Qwen2.5-Coder-32B-Instruct-AWQ

# Option B: Auto-download on first run
# vLLM will download automatically

# 3. Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-32B-Instruct-AWQ \
    --quantization awq \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --port 8000

# Alternative: Using llama.cpp (CPU/GPU hybrid, lower memory)
# Download GGUF version
huggingface-cli download Qwen/Qwen2.5-Coder-32B-Instruct-GGUF

# Run with llama.cpp
./llama-server \
    -m qwen2.5-coder-32b-instruct-q4_k_m.gguf \
    -ngl 60 \
    --port 8000
```

**LLM Client Implementation:**

```python
# src/llm_client.py
from openai import OpenAI
from typing import Dict, Any, Optional
import yaml

class QwenLLMClient:
    """Client for Qwen models running locally via vLLM or llama.cpp"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Connect to local vLLM/llama.cpp server (OpenAI-compatible API)
        self.client = OpenAI(
            base_url=self.config["llm"]["base_url"],
            api_key="not-needed"  # Local server doesn't need auth
        )
        
        self.primary_model = self.config["llm"]["code_generation_model"]["model_name"]
        self.fallback_model = self.config["llm"]["fallback_model"]["model_name"]
    
    def generate_tool_spec(self, intent: Dict[str, Any]) -> str:
        """Generate ToolSpec from user intent"""
        prompt = self._build_spec_prompt(intent)
        return self._call_model(prompt, task="spec_generation")
    
    def generate_code(self, spec: Dict[str, Any]) -> str:
        """Generate code from ToolSpec"""
        prompt = self._build_code_prompt(spec)
        return self._call_model(prompt, task="code_generation")
    
    def repair_code(self, code: str, errors: Dict[str, Any]) -> str:
        """Fix code based on validation errors"""
        prompt = self._build_repair_prompt(code, errors)
        return self._call_model(
            prompt, 
            task="repair",
            temperature=0.3  # Higher for creative fixes
        )
    
    def _call_model(
        self,
        prompt: str,
        task: str = "code_generation",
        temperature: Optional[float] = None,
        use_fallback: bool = False
    ) -> str:
        """Call Qwen model via OpenAI-compatible API"""
        
        model_config = self.config["llm"]["code_generation_model"]
        if use_fallback:
            model_config = self.config["llm"]["fallback_model"]
            model = self.fallback_model
        else:
            model = self.primary_model
        
        temp = temperature or model_config.get("temperature", 0.2)
        max_tokens = model_config.get("max_tokens", 4096)
        
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Python developer specializing in data analysis tool generation. Generate clean, idiomatic, type-safe code."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=temp,
            max_tokens=max_tokens,
            top_p=model_config.get("top_p", 0.95),
        )
        
        return response.choices[0].message.content
    
    def _build_spec_prompt(self, intent: Dict) -> str:
        # Load prompt template from prompts/spec_generation.txt
        with open("prompts/spec_generation.txt") as f:
            template = f.read()
        return template.format(**intent)
    
    def _build_code_prompt(self, spec: Dict) -> str:
        with open("prompts/code_generation.txt") as f:
            template = f.read()
        return template.format(**spec)
    
    def _build_repair_prompt(self, code: str, errors: Dict) -> str:
        with open("prompts/repair.txt") as f:
            template = f.read()
        return template.format(code=code, errors=errors)
```

**Why Qwen2.5-Coder is Ideal for This Project:**

1. **Specialized for Code**: Trained on 5.5T tokens of code and text
2. **Long Context**: Supports up to 128K context (great for large reference files)
3. **Strong Python Performance**: Excellent at pandas, type hints, decorators
4. **Instruction Following**: Fine-tuned to follow complex specifications
5. **Repository-Level Understanding**: Can understand project structure
6. **Artifact Generation**: Good at generating structured outputs (JSON, markdown)
7. **Cost**: Free for on-prem deployment

**Performance Benchmarks (Qwen2.5-Coder-32B):**

- HumanEval: ~80% pass@1 (comparable to GPT-4)
- MBPP: ~75% pass@1
- LiveCodeBench: Strong performance on recent problems
- Python-specific tasks: Often exceeds GPT-4

**Memory Usage Breakdown (48GB VRAM):**

```
Qwen2.5-Coder-32B-Instruct (4-bit AWQ):
├── Model weights: ~20GB
├── KV cache (8K context): ~4GB
├── Activation memory: ~2GB
├── Reserved/fragmentation: ~2GB
└── Total: ~28GB

Remaining VRAM: ~20GB
├── Can run Qwen2.5-Coder-7B simultaneously: ~14GB
└── Buffer: ~6GB
```

**Alternative: Dual-Model Setup (Recommended)**

Run two models simultaneously for different tasks:

```python
# Primary: Qwen2.5-Coder-32B-Instruct (4-bit) - Complex generation
# Port 8000, 28GB VRAM

# Secondary: Qwen2.5-Coder-7B-Instruct (FP16) - Simple tasks, validation
# Port 8001, 14GB VRAM

# Total: 42GB (fits comfortably in 48GB)
```

**Docker Deployment:**

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN pip install vllm transformers

# Download models
RUN huggingface-cli download Qwen/Qwen2.5-Coder-32B-Instruct-AWQ
RUN huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct

# Start both models
CMD python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-32B-Instruct-AWQ \
    --quantization awq \
    --port 8000 \
    --gpu-memory-utilization 0.6 & \
    python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --port 8001 \
    --gpu-memory-utilization 0.3
```

**Fallback Options (If 32B is Too Large):**

If you need more headroom or faster inference:

```yaml
# Lightweight config for 48GB
primary_model: "Qwen/Qwen2.5-Coder-14B-Instruct"  # FP16, ~28GB
fallback_model: "Qwen/Qwen2.5-Coder-7B-Instruct"  # FP16, ~14GB
# Total: ~42GB, leaves 6GB buffer
```

**Comparison with Other Local Models:**

| Model | Size | Quality (Code) | Speed | Memory (4-bit) | Notes |
|-------|------|----------------|-------|----------------|-------|
| **Qwen2.5-Coder-32B** | 32B | ⭐⭐⭐⭐⭐ | Medium | ~20GB | **Best choice** |
| DeepSeek-Coder-33B | 33B | ⭐⭐⭐⭐ | Medium | ~21GB | Good alternative |
| CodeLlama-70B | 70B | ⭐⭐⭐⭐ | Slow | ~42GB | Too large for dual-model |
| Qwen2.5-32B | 32B | ⭐⭐⭐⭐ | Medium | ~20GB | General model (also good) |
| StarCoder2-15B | 15B | ⭐⭐⭐ | Fast | ~10GB | Weaker instruction following |

**Recommendation for Your Setup:**

```yaml
# Optimal 48GB Configuration
Primary: Qwen2.5-Coder-32B-Instruct-AWQ (4-bit)
  - Code generation, spec creation, repair
  - ~20-24GB VRAM
  
Secondary: Qwen2.5-Coder-7B-Instruct (FP16)
  - Simple tools, validation, quick drafts
  - ~14GB VRAM

Total: ~38-42GB (safe margin for 48GB)
```

### 5.2 Project Structure

```
MCP_Tool_Code_Interpreter_Generator/
├── src/
│   ├── __init__.py
│   ├── intent_extraction.py      # Step 1
│   ├── spec_generator.py          # Step 2
│   ├── code_generator.py          # Step 3
│   ├── validator.py               # Step 4
│   ├── executor.py                # Step 5
│   ├── presenter.py               # Step 6
│   ├── feedback_handler.py        # Step 7
│   ├── promoter.py                # Step 8A
│   ├── models.py                  # Data models
│   ├── metrics/                   # Code quality metrics
│   │   ├── __init__.py
│   │   ├── functional_correctness.py  # Reference solution & test cases
│   │   ├── pass_at_k.py           # Pass@k calculation
│   │   ├── test_pass_rate.py      # Test execution & scoring
│   │   ├── code_bleu.py           # Semantic closeness metrics
│   │   ├── ngram_matcher.py       # N-gram & weighted n-gram
│   │   ├── ast_matcher.py         # AST-based similarity
│   │   └── dataflow_analyzer.py   # Dataflow graph analysis
│   └── utils/
│       ├── csv_helpers.py         # Shared utilities
│       ├── type_detection.py
│       └── validation_helpers.py
├── registry/
│   ├── active/
│   ├── staging/
│   └── archive/
├── prompts/
│   ├── intent_extraction.txt
│   ├── spec_generation.txt
│   ├── code_generation.txt
│   └── repair.txt
├── tests/
│   ├── test_intent_extraction.py
│   ├── test_spec_generator.py
│   └── ...
├── reference_files/            # Example tools and outputs
├── mcp_server.py               # MCP tool server entry point
├── requirements.txt
├── README.md
└── ProjectRequirements.instructions.md  # This file
```

### 5.3 Shared Utilities (Must Implement)

Based on the example tool, create these reusable functions:

```python
# csv_helpers.py
def load_csv_data_with_types(
    file_path: str, 
    auto_expand_json: bool = True
) -> pd.DataFrame:
    """Load CSV with automatic type detection and JSON expansion"""
    pass

def detect_column_types(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect numeric, categorical, datetime, and high-cardinality columns"""
    pass

def validate_file_and_columns(
    file_path: str, 
    required_columns: Optional[List[str]] = None
) -> None:
    """Validate file exists and contains required columns"""
    pass
```

### 5.4 LLM Prompts (Templates)

#### Spec Generation Prompt
```
You are an expert tool specification designer for MCP data analysis tools.

Given the user's intent and dataset schema, generate a ToolSpec.

User Intent: {intent}
Dataset Schema: {schema}

Generate a JSON ToolSpec with:
1. tool_name: descriptive snake_case name
2. description: one-sentence summary
3. parameters: list of Annotated parameters with Field descriptions
4. return_type: always "str" (markdown-formatted)
5. when_to_use: trigger conditions
6. what_it_does: step-by-step logic
7. returns: output format specification
8. prerequisites: required prior steps

Follow the pattern of this reference tool:
{reference_tool_example}

Output JSON only.
```

#### Code Generation Prompt
```
You are an expert Python developer generating MCP data analysis tools.

Generate a complete Python tool implementation from this ToolSpec:
{tool_spec}

Requirements:
1. Use @mcp.tool() decorator
2. Use Annotated types with Field descriptions
3. Follow this structure:
   - Load data with load_csv_data_with_types()
   - Validate with validate_file_and_columns()
   - Perform transformation
   - Format output as markdown
   - Add JSON footer with column metadata
4. Include comprehensive error handling
5. Match the style of this reference tool:
   {reference_tool_example}

Output the complete tool.py file.
```

### 5.5 Configuration

Create `config.yaml`:
```yaml
llm:
  model: "gpt-4"
  temperature: 0.2
  max_tokens: 2000

validation:
  max_repair_iterations: 3
  sandbox_timeout_seconds: 30
  
  # Code quality thresholds
  quality_thresholds:
    functional_correctness_min: 0.90
    test_pass_rate_min: 0.95
    code_bleu_min: 0.75
    pass_at_5_min: 0.90
    overall_quality_min: 0.85
  
  # Metric weights for overall score
  metric_weights:
    functional_correctness: 0.40
    test_pass_rate: 0.30
    code_bleu: 0.20
    pass_at_1: 0.10
  
  # Code BLEU component weights
  code_bleu_weights:
    ngram: 0.25
    weighted_ngram: 0.25
    ast: 0.25
    dataflow: 0.25

registry:
  active_path: "registry/active"
  staging_path: "registry/staging"
  archive_path: "registry/archive"

feedback:
  approve_keywords: ["yes", "approve", "looks good", "works", "ship it"]
  reject_keywords: ["no", "reject", "wrong", "incorrect"]
  default_decision: "REJECTED"

promotion:
  require_version_bump: true
  keep_promotion_log: true
```

---

## 6. Testing Strategy

### 6.1 Unit Tests
- Test each component independently
- Mock LLM responses for deterministic tests
- Validate data models with Pydantic

### 6.2 Integration Tests
- End-to-end pipeline: intent → tool → execution → feedback
- Test with sample datasets (traffic_accidents.csv)
- Verify registry promotion/rollback

### 6.3 Validation Tests
- Schema validation correctness
- Sandbox isolation (no access to host filesystem outside allowed paths)
- Error recovery in repair loop

### 6.4 Code Metrics Tests
- **N-gram matching accuracy**: Verify BLEU scores against known code pairs
- **AST extraction correctness**: Test AST node extraction and depth calculation
- **Dataflow analysis accuracy**: Validate variable dependency graph extraction
- **Pass@k calculation**: Test with known sample distributions
- **Test case generation**: Verify edge case coverage
- **Functional correctness scoring**: Test with reference solutions

Example metric tests:
```python
def test_ngram_matching():
    code1 = "def foo(): return x + y"
    code2 = "def foo(): return x + y"
    score = calculate_bleu(code1, code2)
    assert score["bleu_score"] == 1.0  # Identical code
    
def test_ast_similarity():
    code1 = "x = a + b"
    code2 = "y = c + d"  # Same structure, different names
    result = calculate_ast_similarity(code1, code2)
    assert result["ast_match_score"] > 0.8  # High structural similarity
    
def test_dataflow_extraction():
    code = """
    a = 10
    b = a + 5
    c = b * 2
    """
    flow = extract_dataflow(code)
    assert "b" in flow["variable_dependencies"]
    assert "a" in flow["variable_dependencies"]["b"]
    assert "b" in flow["variable_dependencies"]["c"]

def test_pass_at_k():
    # 10 candidates, 8 correct
    pass_at_k = calculate_pass_at_k(n=10, c=8, k_values=[1, 5])
    assert pass_at_k[1] == 0.8  # 80% chance first is correct
    assert pass_at_k[5] > 0.95  # >95% chance one of top 5 is correct

def test_functional_correctness():
    def reference_tool(data):
        return data.groupby("col").size()
    
    def generated_tool(data):
        return data.groupby("col").count()
    
    test_data = pd.DataFrame({"col": ["A", "A", "B"]})
    correctness = validate_against_reference(
        generated_tool, reference_tool, [{"data": test_data}]
    )
    # Should detect difference between size() and count()
    assert correctness < 1.0

def test_code_bleu_combined():
    reference = load_reference_tool("load_and_analyze_csv")
    candidate = generate_tool_from_spec(spec)
    
    metrics = calculate_code_bleu(candidate, reference)
    assert 0.0 <= metrics.code_bleu_score <= 1.0
    assert all(0.0 <= v <= 1.0 for v in metrics.ngram_precision.values())
    assert metrics.ast_match_score >= 0.0
    assert metrics.dataflow_match_score >= 0.0

def test_overall_quality_scoring():
    score = calculate_overall_quality_score(
        functional_correctness=0.92,
        pass_at_k={1: 0.85, 5: 0.93},
        test_pass_rate=0.96,
        code_bleu=0.78
    )
    assert 0.85 <= score <= 0.95
    assert evaluate_tool_quality(score) == "EXCELLENT - Auto-promote candidate"
```

### 6.5 Integration Tests for Metrics Pipeline
```python
def test_full_validation_with_metrics():
    """Test complete validation pipeline with all code metrics"""
    
    # Generate candidate tool
    spec = generate_tool_spec(user_intent)
    candidate_code = generate_tool_code(spec)
    reference_code = load_reference_solution(spec.tool_name)
    
    # Run full validation
    validation_report = validate_tool_with_metrics(
        candidate_code=candidate_code,
        reference_code=reference_code,
        test_cases=generate_test_cases(spec)
    )
    
    # Verify all metrics are computed
    assert validation_report.code_metrics is not None
    metrics = validation_report.code_metrics
    
    assert metrics.functional_correctness.correctness_score >= 0.0
    assert metrics.test_pass_rate >= 0.0
    assert metrics.pass_at_k[1] >= 0.0
    assert metrics.semantic_closeness.code_bleu_score >= 0.0
    
    # Verify thresholds are applied
    if metrics.overall_score >= 0.90:
        assert validation_report.schema_ok == True
        assert validation_report.sandbox_ok == True
```

### 6.6 Example Test Cases

```python
def test_intent_extraction():
    request = "Analyze traffic accidents by severity"
    intent = extract_intent(request)
    assert intent["operation"] == "group_by_and_aggregate"
    assert "severity" in intent["columns"]

def test_spec_generation():
    intent = {...}
    spec = generate_tool_spec(intent)
    assert spec.tool_name.islower()
    assert "_" in spec.tool_name
    assert len(spec.parameters) > 0

def test_code_generation():
    spec = ToolSpec(...)
    code = generate_tool_code(spec)
    assert "@mcp.tool()" in code
    assert "def " + spec.tool_name in code

def test_validation_pass():
    candidate = create_test_candidate()
    report = validate_tool(candidate)
    assert report.schema_ok == True
    assert report.sandbox_ok == True
    
    # Verify code metrics are computed
    assert report.code_metrics is not None
    assert report.code_metrics.overall_score >= 0.0

def test_feedback_approval():
    feedback = parse_feedback("yes, looks good!")
    assert feedback.decision == "APPROVED"

def test_feedback_rejection():
    feedback = parse_feedback("no, wrong column")
    assert feedback.decision == "REJECTED"

def test_promotion():
    candidate = create_approved_candidate()
    promote_tool(candidate)
    assert get_active_tool(candidate.tool_id) is not None
    assert candidate.status == ToolStatus.PROMOTED
```

---

## 7. Deployment & Operations

### 7.1 MCP Server Setup

```python
# mcp_server.py
from mcp.server.fastmcp import FastMCP
import importlib.util
import json
from pathlib import Path

mcp = FastMCP("Data Analysis Tools")

def load_active_tools():
    """Dynamically load all tools from active registry"""
    metadata_path = Path("registry/active/metadata.json")
    with open(metadata_path) as f:
        registry = json.load(f)
    
    for tool_name, tool_info in registry["tools"].items():
        # Convert filesystem path to actual file path
        tool_file = Path(tool_info["path"]) / "tool.py"
        
        if not tool_file.exists():
            print(f"Warning: Tool file not found: {tool_file}")
            continue
        
        # Load module from file path (not import path)
        spec = importlib.util.spec_from_file_location(
            f"registry.active.tools.{tool_name}",
            tool_file
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # Tool is auto-registered via @mcp.tool() decorator
            print(f"✓ Loaded tool: {tool_name} v{tool_info['version']}")
        else:
            print(f"✗ Failed to load: {tool_name}")

# Load all promoted tools on server start
load_active_tools()

if __name__ == "__main__":
    mcp.run()
```

### 7.2 Tool Discovery API

Provide endpoint for querying available tools:
```python
@mcp.tool()
def list_available_tools() -> str:
    """List all registered MCP tools with descriptions"""
    metadata_path = "registry/active/metadata.json"
    with open(metadata_path) as f:
        registry = json.load(f)
    
    tools_list = []
    for tool_name, info in registry["tools"].items():
        spec_path = f"{info['path']}/spec.json"
        with open(spec_path) as spec_file:
            spec = json.load(spec_file)
        tools_list.append(f"- **{tool_name}** (v{info['version']}): {spec['description']}")
    
    return "\n".join(tools_list)
```

### 7.3 Monitoring

Track:
- Tool generation success rate
- Validation failure reasons
- Approval vs. rejection rate
- Tool execution performance (runtime, errors)

Create `metrics.json`:
```json
{
  "total_generated": 42,
  "approved": 35,
  "rejected": 7,
  "avg_generation_time_ms": 1200,
  "validation_failure_rate": 0.12,
  "common_rejection_reasons": [
    "wrong_column_names",
    "missing_filter",
    "output_format_mismatch"
  ]
}
```

---

## 8. Security & Safety

### 8.1 Code Generation Safety

**Capability-Based Security Policy**:

Instead of pattern-based blocking, enforce runtime capabilities:

```python
# Forbidden patterns (no exceptions)
FORBIDDEN_PATTERNS = [
    "eval(",
    "exec(",
    "compile(",
    "__import__",  # except via safe imports
    "subprocess",
    "os.system",
    "pickle.loads",
    "requests.",   # network access
    "urllib",
    "socket",
    "http.",
]

# Restricted patterns (allowed with validation)
RESTRICTED_PATTERNS = {
    "open(": "file_operations",
    "pd.read_": "file_operations",
    "Path(": "file_operations",
}

def check_code_safety(code: str) -> Tuple[bool, List[str]]:
    """Pattern-based static checks + runtime policy enforcement"""
    violations = []
    
    # Check forbidden patterns
    for pattern in FORBIDDEN_PATTERNS:
        if pattern in code:
            violations.append(f"Forbidden pattern: {pattern}")
    
    # Check restricted patterns (warn, don't block)
    warnings = []
    for pattern, capability in RESTRICTED_PATTERNS.items():
        if pattern in code:
            warnings.append(f"Restricted pattern '{pattern}' requires {capability} validation")
    
    return len(violations) == 0, violations, warnings

# Runtime enforcement (primary security layer)
ALLOWED_READ_PATHS = [
    "/workspace/data",
    "/tmp/analysis/inputs",
    # User-provided paths validated at runtime
]

ALLOWED_WRITE_PATHS = [
    "/tmp/analysis/outputs",
]

MAX_DATASET_ROWS = 10_000_000
MAX_DATASET_COLS = 10_000
MAX_MEMORY_MB = 4096

def validate_file_path(path: str, mode: str = "read") -> bool:
    """Runtime file access validation"""
    import os
    resolved = os.path.realpath(path)
    
    allowed_paths = ALLOWED_READ_PATHS if mode == "read" else ALLOWED_WRITE_PATHS
    
    # Check if path is within allowed directories
    if not any(resolved.startswith(os.path.realpath(allowed)) for allowed in allowed_paths):
        raise SecurityError(f"File access denied: {resolved} not in allowed paths")
    
    return True

def enforce_resource_limits(df: pd.DataFrame) -> None:
    """Enforce dataset size limits"""
    if len(df) > MAX_DATASET_ROWS:
        raise ResourceError(f"Dataset too large: {len(df)} rows (max: {MAX_DATASET_ROWS})")
    if len(df.columns) > MAX_DATASET_COLS:
        raise ResourceError(f"Too many columns: {len(df.columns)} (max: {MAX_DATASET_COLS})")
    
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    if memory_mb > MAX_MEMORY_MB:
        raise ResourceError(f"Dataset too large: {memory_mb:.1f}MB (max: {MAX_MEMORY_MB}MB)")
```

**Import Allowlist**:
```python
ALLOWED_IMPORTS = [
    "pandas", "numpy", "json", "csv", "datetime", "re",
    "collections", "itertools", "functools", "math",
    "typing", "dataclasses", "pydantic"
]

def validate_imports(code: str) -> bool:
    """Ensure only safe imports are used"""
    import ast
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split('.')[0] not in ALLOWED_IMPORTS:
                    raise SecurityError(f"Import not allowed: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split('.')[0] not in ALLOWED_IMPORTS:
                raise SecurityError(f"Import not allowed: {node.module}")
    return True
```

### 8.2 Sandbox Isolation

Run validation in Docker container or isolated virtual environment:
```python
def run_in_sandbox(tool_code: str, test_data: str) -> Dict:
    # Create temp venv
    # Install dependencies
    # Run tool with test_data
    # Capture output and errors
    # Destroy venv
    pass
```

### 8.3 File Access Restrictions

Generated tools should only access:
- Files explicitly provided by user
- Paths within allowed workspace directories

```python
ALLOWED_PATHS = [
    "/workspace/data",
    "/tmp/analysis"
]

def validate_file_path(path: str) -> bool:
    resolved = os.path.realpath(path)
    return any(resolved.startswith(allowed) for allowed in ALLOWED_PATHS)
```

---

## 9. Future Enhancements

### 9.0 Non-Goals (Out of Scope)

To maintain focus, the following are **explicitly not in scope** for this project:

- ❌ **General IDE/code editor agent** - We generate MCP tools only, not arbitrary code
- ❌ **Production deployment of generated code** - Tools run in controlled MCP server environment
- ❌ **Multi-language support** - Python only for MVP
- ❌ **Visual UI builder** - Command-line/API driven; UI is future enhancement
- ❌ **Real-time collaboration** - Single-user workflow for MVP
- ❌ **Automatic dependency installation** - Dependencies must be pre-installed in environment
- ❌ **Cloud deployment** - Local/self-hosted only
- ❌ **Tool marketplace/sharing** - Private registry only

### 9.1 Version 1.0 (MVP) - Planned Scope
- [ ] Intent extraction and gap detection
- [ ] Contract-first spec generation with JSON schemas
- [ ] Code generation with validation
- [ ] Schema + test-based validation (Code BLEU optional)
- [ ] Staged execution in isolated environment
- [ ] Two-stage feedback capture (output + tool approval)
- [ ] User-gated promotion/rejection with audit log

### 9.2 Version 1.1
- [ ] Auto-repair with LLM feedback loop (implemented but can be improved)
- [ ] Support for non-CSV data sources (JSON, Parquet, SQL)
- [ ] Tool versioning and upgrade paths
- [ ] Performance metrics and caching

### 9.3 Version 2.0
- [ ] Multi-tool composition (chaining tools)
- [ ] Visual tool builder UI
- [ ] A/B testing for tool variants
- [ ] Automatic tool deprecation based on usage stats

### 9.4 Version 2.1
- [ ] Support for visualization tools (matplotlib, plotly)
- [ ] Export tools as standalone packages
- [ ] Collaborative tool library (team registry)

---

## 10. Success Criteria

### 10.1 Functional Requirements
✅ Generate syntactically correct MCP tools from natural language
✅ Validate tools before execution
✅ Never auto-register; require explicit approval
✅ Support iterative refinement based on feedback
✅ Maintain separation between staging and active registry

### 10.2 Quality Metrics
- **Generation Success Rate**: >90% of tools pass initial validation
- **Approval Rate**: >70% of presented tools approved by users
- **False Positive Rate**: <5% of approved tools fail in production
- **Average Time to Tool**: <60 seconds from request to staged execution

### 10.3 User Experience
- Clear, unambiguous approval prompts
- Transparent about tool assumptions and limitations
- Fast feedback loop (request → result in <1 minute)
- No tool pollution (only high-quality tools in registry)

---

## 11. References & Examples

### 11.1 Example Tool (load_and_analyze_csv)
See reference implementation in `reference_files/sample_mcp_tools/`

Key patterns to follow:
- Use `@mcp.tool()` decorator
- Annotated parameter types
- Structured docstring (WHEN/WHAT/RETURNS/PREREQUISITES)
- Markdown output format
- JSON footer for downstream parsing
- Comprehensive error handling

### 11.2 Sample Outputs
See `reference_files/sample_planner_output/` and `sample_response_to_no_2/`

### 11.3 Related Documentation
- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [Pydantic Models](https://docs.pydantic.dev/)

---

## 12. Getting Started

### 12.1 Quick Start

1. **Clone repository**:
   ```bash
   git clone <repo_url>
   cd MCP_Tool_Code_Interpreter_Generator
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment**:
   ```bash
   export OPENAI_API_KEY="your-key"
   # or create .env file
   ```

4. **Initialize registry**:
   ```bash
   python -m src.init_registry
   ```

5. **Run example**:
   ```bash
   python examples/generate_first_tool.py
   ```

### 12.2 Development Workflow

1. Create feature branch
2. Implement component (e.g., `intent_extraction.py`)
3. Write unit tests
4. Run validation: `pytest tests/`
5. Test end-to-end with sample data
6. Submit PR with test coverage report

---

## Appendix A: Glossary

- **MCP**: Model Context Protocol - standard for AI tool integration
- **ToolSpec**: Contract defining tool interface (inputs, outputs, behavior)
- **Staging Registry**: Temporary storage for validated but unapproved tools
- **Active Registry**: Production registry of promoted, approved tools
- **Gap Detection**: Process of identifying missing tool capabilities
- **Promotion**: Moving a tool from staging to active registry
- **Tool Pollution**: Accumulation of low-quality tools in registry
- **Code BLEU**: Composite metric measuring semantic code similarity through n-gram, AST, and dataflow matching
- **Pass@k**: Probability that at least one of k generated code samples is functionally correct
- **Functional Correctness**: Measure of how well generated code performs the intended task against reference solutions or test cases
- **AST**: Abstract Syntax Tree - hierarchical tree representation of code structure
- **Dataflow Analysis**: Tracking how data (variables) flows through code and their dependencies
- **N-gram Match**: Text similarity metric based on contiguous sequences of n tokens

---

## Appendix B: Code Metrics Reference

### B.1 Metrics Summary Table

| Metric Category | Metric Name | Range | Target Threshold | Purpose |
|----------------|-------------|-------|------------------|---------|
| **Functional Correctness** | Reference Similarity | 0.0-1.0 | ≥0.90 | Match reference solution output |
| | Test Pass Rate | 0.0-1.0 | ≥0.95 | Pass generated test cases |
| | Correctness Score | 0.0-1.0 | ≥0.90 | Combined reference + test score |
| **Pass@k** | pass@1 | 0.0-1.0 | ≥0.85 | Best candidate correctness |
| | pass@5 | 0.0-1.0 | ≥0.90 | Top 5 candidates correctness |
| | pass@10 | 0.0-1.0 | ≥0.95 | Top 10 candidates correctness |
| **Test Pass Rate** | Overall Pass Rate | 0.0-1.0 | ≥0.95 | All test categories combined |
| | Edge Case Pass Rate | 0.0-1.0 | ≥0.95 | Edge case handling |
| | Stress Test Pass Rate | 0.0-1.0 | ≥0.85 | Large data handling |
| **Code BLEU** | N-gram Match | 0.0-1.0 | ≥0.70 | Token sequence similarity |
| | Weighted N-gram | 0.0-1.0 | ≥0.75 | Keyword-weighted similarity |
| | AST Match | 0.0-1.0 | ≥0.70 | Structural similarity |
| | Dataflow Match | 0.0-1.0 | ≥0.70 | Variable flow similarity |
| | Combined Code BLEU | 0.0-1.0 | ≥0.75 | Weighted average of all |
| **Overall** | Quality Score | 0.0-1.0 | ≥0.85 | Weighted combination of all metrics |

### B.2 Metric Weights (Default Configuration)

```yaml
# Overall quality score weights
overall_quality_weights:
  functional_correctness: 0.40  # Most critical
  test_pass_rate: 0.30          # High importance
  code_bleu: 0.20               # Semantic similarity
  pass_at_1: 0.10               # Generation reliability

# Code BLEU component weights
code_bleu_weights:
  ngram: 0.25                   # Token-level matching
  weighted_ngram: 0.25          # Keyword importance
  ast: 0.25                     # Structural matching
  dataflow: 0.25                # Logic flow matching

# Quality tier thresholds
quality_tiers:
  excellent: 0.90               # Auto-promote candidate
  good: 0.75                    # Promote with user approval
  acceptable: 0.60              # Requires detailed review
  poor: 0.00                    # Reject or repair
```

### B.3 Typical Metric Profiles

**High-Quality Tool** (Promotable):
```json
{
  "functional_correctness": 0.95,
  "test_pass_rate": 0.98,
  "pass_at_k": {"1": 0.90, "5": 0.96},
  "code_bleu": 0.82,
  "overall_score": 0.92,
  "quality_tier": "EXCELLENT"
}
```

**Borderline Tool** (Needs Review):
```json
{
  "functional_correctness": 0.78,
  "test_pass_rate": 0.85,
  "pass_at_k": {"1": 0.70, "5": 0.85},
  "code_bleu": 0.68,
  "overall_score": 0.77,
  "quality_tier": "GOOD"
}
```

**Poor Tool** (Requires Repair):
```json
{
  "functional_correctness": 0.55,
  "test_pass_rate": 0.65,
  "pass_at_k": {"1": 0.45, "5": 0.70},
  "code_bleu": 0.48,
  "overall_score": 0.58,
  "quality_tier": "POOR"
}
```

### B.4 Metrics Computation Workflow

```
┌─────────────────────────────────────────────────────────┐
│         Generated Tool Code + Reference Solution         │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
┌──────────────────┐    ┌──────────────────────┐
│ Functional       │    │ Semantic Closeness   │
│ Correctness      │    │ (Code BLEU)          │
│                  │    │                      │
│ • Reference comp │    │ • N-gram match       │
│ • Test execution │    │ • Weighted n-gram    │
│ • Pass rate      │    │ • AST match          │
└────────┬─────────┘    │ • Dataflow match     │
         │              └──────────┬───────────┘
         │                         │
         └────────────┬────────────┘
                      ▼
         ┌─────────────────────────┐
         │  Overall Quality Score   │
         │  (weighted combination)  │
         └────────────┬─────────────┘
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
┌─────────────────┐      ┌─────────────────┐
│ Score ≥ 0.85?   │ YES  │ PROMOTE         │
│                 ├─────→│ (with approval) │
└────────┬────────┘      └─────────────────┘
         │ NO
         ▼
┌─────────────────┐
│ Enter Repair    │
│ Loop (max 3x)   │
└─────────────────┘
```

---

## Appendix C: Decision Log

| Date       | Decision                                      | Rationale                                           |
|------------|-----------------------------------------------|-----------------------------------------------------|
| 2026-01-22 | Use staging-first approach                    | Prevent auto-registration of untested tools         |
| 2026-01-22 | Require explicit approval                     | Avoid tool pollution from ambiguous feedback        |
| 2026-01-22 | Default to REJECTED on ambiguous feedback     | Safety-first: only register high-confidence tools   |
| 2026-01-22 | Max 3 repair iterations                       | Balance quality improvement with generation speed   |
| 2026-01-22 | Markdown output format                        | Human-readable, easy to embed in agent responses    |
| 2026-01-22 | JSON footer in tool outputs                   | Enable downstream argument inference                |
| 2026-01-22 | Add comprehensive code quality metrics        | Ensure only high-quality tools promoted to registry |
| 2026-01-22 | Implement Code BLEU with 4 components         | Multi-faceted semantic similarity beyond text match |
| 2026-01-22 | Use pass@k for candidate ranking              | Statistical measure of generation reliability       |
| 2026-01-22 | Weight functional correctness 40%             | Correctness more important than style similarity    |
| 2026-01-22 | Set overall quality threshold at 0.85         | Balance quality bar with practical generation speed |

---

**Document Version**: 1.0.0  
**Last Updated**: 2026-01-22  
**Maintained By**: MCP Tool Generator Team
