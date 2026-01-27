# MCP Tool Code Interpreter Generator - Module Documentation

This directory contains Pull Request specifications for all modules in the **MCP Tool Code Interpreter Generator** project. This system automatically generates, validates, and deploys MCP tools from natural language queries using LangGraph orchestration and Qwen LLM.

---

## 🎯 Project Overview

**Goal**: Build an autonomous system that:
1. Takes a natural language data analysis query
2. Generates a complete MCP tool with validation
3. Executes the tool and presents results
4. Promotes approved tools to an active registry

**Tech Stack**:
- **FastMCP**: MCP server framework (decorator-based, no class inheritance)
- **LangGraph**: StateGraph workflow orchestration with conditional routing
- **Qwen 2.5-Coder 32B**: On-premises LLM via vLLM for all code generation
- **Pydantic v2.5.0+**: Data validation and serialization
- **Python 3.10+**: Core runtime

---

## 📊 System Architecture

### Execution Flow

```
User Query + Dataset
        ↓
   @mcp.tool() analyze_data()
        ↓
   LangGraph StateGraph Pipeline
        ↓
   ┌─────────────────────────────────────────┐
   │  1. intent_node                         │  Extract operation, columns, metrics
   │     ↓                                    │  Generate implementation plan
   │  2. spec_generator_node                 │  Create ToolSpec (schema, params)
   │     ↓                                    │
   │  3. code_generator_node                 │  Generate Python function with MCP decorator
   │     ↓                                    │
   │  4. validator_node ←─┐                  │  Syntax, schema, sandbox checks
   │     ↓                │                   │
   │  5. repair_node ─────┘ (if errors)      │  Auto-fix validation errors (max 3 attempts)
   │     ↓                                    │
   │  6. executor_node                       │  Execute tool, capture results
   │     ↓                                    │
   │  7. feedback_stage1_node (interrupt)    │  User validates output quality
   │     ↓                                    │
   │  8. feedback_stage2_node (interrupt)    │  User approves for promotion
   │     ↓                                    │
   │  9. promoter_node                       │  Copy to active registry
   └─────────────────────────────────────────┘
        ↓
   Return promoted tool + execution results
```

### Tool Lifecycle States

```
DRAFT → STAGED → APPROVED → PROMOTED
  ↓                 ↓
REJECTED        ARCHIVED
```

---

## 📦 Module Specifications

### Foundation Modules (P0)

#### [01_data_models_PR.md](01_data_models_PR.md)
**Priority**: P0 | **Effort**: 2-3 days | **Module**: `src/models.py`

Defines all core data models used across the system:

**Key Models**:
- `ToolStatus`: Enum for lifecycle states (DRAFT, STAGED, APPROVED, REJECTED, PROMOTED)
- `ToolSpec`: Complete specification (name, description, schemas, parameters, metadata)
- `ValidationReport`: Validation results (schema_ok, tests_ok, sandbox_ok, errors, warnings)
- `RunArtifacts`: Execution results (result dict, markdown summary, timing, errors)
- `UserFeedback`: User decisions (APPROVED/REJECTED, notes, timestamp)
- `ToolCandidate`: Complete tool bundle (spec, code, status, validation, feedback)
- `ToolGeneratorState`: LangGraph state with all pipeline data

**Responsibilities**:
- Provide type-safe data structures
- Enable state persistence across nodes
- Support JSON serialization for LLM interactions

---

#### [02_llm_client_PR.md](02_llm_client_PR.md)
**Priority**: P0 | **Effort**: 2-3 days | **Module**: `src/llm_client.py`

Provides abstraction layer for LLM interactions:

**Classes**:
- `BaseLLMClient`: Abstract interface for LLM operations
- `QwenLLMClient`: Implementation for Qwen 2.5-Coder via vLLM

**Features**:
- `generate()`: Free-form text generation
- `generate_structured()`: JSON-schema constrained generation
- Config-driven endpoint management
- Temperature control for deterministic outputs (0.2-0.3 for code gen)

**Usage**:
```python
llm = QwenLLMClient()
spec = llm.generate_structured(prompt, TOOLSPEC_SCHEMA)
```

---

#### [03_intent_extraction_PR.md](03_intent_extraction_PR.md)
**Priority**: P0 | **Effort**: 2-3 days | **Module**: `src/intent_extraction.py`

Extracts structured intent from natural language queries:

**Node**: `intent_node()`
- Analyzes user query + dataset schema
- Generates detailed implementation plan (step-by-step todos)
- Detects capability gaps (new tool needed vs. existing tool)

**Core Logic**:
```python
class IntentExtractor:
    def extract(self, query: str, data_path: str) -> Dict
```

**Output Structure**:
```json
{
  "operation": "groupby_aggregate",
  "columns": ["state", "severity"],
  "metrics": ["count", "mean"],
  "filters": [{"column": "date", "operator": ">", "value": "2023-01-01"}],
  "sort_by": ["count"],
  "sort_order": "descending",
  "limit": 10,
  "output_format": "table",
  "implementation_plan": [
    {"step": 1, "action": "Load CSV", "details": "..."},
    {"step": 2, "action": "Validate columns", "details": "..."}
  ],
  "expected_output": {...},
  "edge_cases": [...],
  "validation_rules": [...]
}
```

**Routing**:
- If `has_gap = True` → `spec_generator_node`
- If `has_gap = False` → `executor_node` (use existing tool)

---

#### [04_spec_generator_PR.md](04_spec_generator_PR.md)
**Priority**: P0 | **Effort**: 2-3 days | **Module**: `src/spec_generator.py`

Generates formal ToolSpec from extracted intent:

**Node**: `spec_generator_node()`
- Consumes intent from step 1
- Generates complete ToolSpec with schemas
- Validates against Pydantic model

**Core Logic**:
```python
class SpecGenerator:
    def generate(self, intent: Dict) -> ToolSpec
```

**Output**: Complete `ToolSpec` object with:
- `tool_name`, `description`
- `input_schema`, `output_schema` (JSON Schema format)
- `parameters` list
- Metadata: `when_to_use`, `what_it_does`, `returns`, `prerequisites`

---

#### [05_code_generator_PR.md](05_code_generator_PR.md)
**Priority**: P0 | **Effort**: 3-4 days | **Module**: `src/code_generator.py`

Generates Python code from ToolSpec:

**Nodes**:
- `code_generator_node()`: Initial code generation
- `repair_node()`: Auto-fix validation errors (max 3 attempts)

**Core Logic**:
```python
class CodeGenerator:
    def generate(self, spec: ToolSpec) -> str
```

**Features**:
- Generates complete Python function
- Adds `@mcp.tool()` decorator
- Includes type hints, error handling
- Formats with `black`
- Returns `Dict[str, Any]` with `result` and `metadata`

**Repair Logic**:
- Takes validation errors as input
- Prompts Qwen to fix specific issues
- Increments `repair_attempts` counter
- Stops after 3 failed attempts

---

#### [06_validator_PR.md](06_validator_PR.md)
**Priority**: P0 | **Effort**: 3-4 days | **Module**: `src/validator.py`

Validates generated code before execution:

**Node**: `validator_node()`
- Runs 3-stage validation
- Routes to repair or execution

**Validation Stages**:
1. **Syntax Check**: AST parsing
2. **Schema Compliance**: Verify input/output schemas match ToolSpec
3. **Sandbox Execution**: Test run with sample data

**Output**: `ValidationReport` with:
- `schema_ok`, `tests_ok`, `sandbox_ok` (bool)
- `errors`, `warnings` (lists)
- `is_valid` property (all checks pass)

**Routing**:
- Valid → `executor_node`
- Invalid + attempts < 3 → `repair_node`
- Invalid + attempts ≥ 3 → `END` (failure)

---

### Execution & Feedback Modules (P1)

#### [07_executor_PR.md](07_executor_PR.md)
**Priority**: P1 | **Effort**: 2 days | **Module**: `src/executor.py`

Executes validated tools and captures results:

**Node**: `executor_node()`
- Loads generated function
- Executes with 300s timeout
- Captures results and timing

**Core Logic**:
```python
class ToolExecutor:
    def execute(self, code_path: str, data_path: str) -> RunArtifacts
```

**Output**: `RunArtifacts` with:
- `result`: Dict with analysis results
- `summary_markdown`: Formatted output
- `execution_time_ms`: Performance metric
- `error`: Exception details (if failed)

---

#### [08_feedback_handler_PR.md](08_feedback_handler_PR.md)
**Priority**: P0 | **Effort**: 1 day | **Module**: `src/feedback_handler.py`

Handles two-stage human-in-the-loop approval:

**Nodes**:
- `feedback_stage1_node()`: Validate execution output quality
- `feedback_stage2_node()`: Approve for promotion to registry

**Interrupts**: Graph pauses at both stages for user input

**Routing**:
- Stage 1 approved → Stage 2
- Stage 1 rejected → END
- Stage 2 approved → `promoter_node`
- Stage 2 rejected → END

**Input**: `user_response_stage1`, `user_response_stage2` (yes/approve/approved)

---

#### [09_promoter_PR.md](09_promoter_PR.md)
**Priority**: P0 | **Effort**: 2 days | **Module**: `src/promoter.py`

Promotes approved tools to active registry:

**Node**: `promoter_node()`
- Handles version conflicts (auto-increment)
- Copies code to active directory
- Updates registry metadata

**Core Logic**:
```python
class ToolPromoter:
    def promote(self, spec: ToolSpec, code: str) -> Dict
```

**Features**:
- Version conflict resolution: `tool_name` → `tool_name_v2`
- Registry update with metadata
- File system management

**Output**:
```json
{
  "name": "analyze_accidents_by_state",
  "path": "/active_tools/analyze_accidents_by_state.py",
  "version": "1.0.0"
}
```

---

### Orchestration Module (P0)

#### [10_pipeline_orchestrator_PR.md](10_pipeline_orchestrator_PR.md)
**Priority**: P0 | **Effort**: 3 days | **Module**: `src/pipeline.py`

Main orchestration layer using LangGraph:

**Components**:
1. **StateGraph Setup**: 9 nodes with conditional routing
2. **Interrupt Points**: `feedback_stage1_node`, `feedback_stage2_node`
3. **Entry Point**: `intent_node`
4. **MCP Tool Wrapper**: `@mcp.tool() analyze_data()`

**Graph Structure**:
```python
workflow = StateGraph(ToolGeneratorState)
workflow.add_node("intent_node", intent_node)
# ... 8 more nodes
workflow.add_conditional_edges("intent_node", route_after_intent)
workflow.add_conditional_edges("validator_node", route_after_validation)
# ... etc
graph = workflow.compile(interrupt_before=["feedback_stage1_node", "feedback_stage2_node"])
```

**MCP Tool**:
```python
@mcp.tool()
def analyze_data(query: str, file_path: str) -> Dict[str, Any]:
    """Entry point for all data analysis requests."""
    result = run_pipeline(query, file_path)
    return {"status": "success", "tool_created": ..., "result": ...}
```

---

## 🚀 Implementation Roadmap

### Phase 1: Foundation (Week 1)
1. [x] 01_data_models_PR.md - Core data structures
2. [x] 02_llm_client_PR.md - Qwen integration
3. [x] 03_intent_extraction_PR.md - Intent extraction with implementation plans

### Phase 2: Generation Pipeline (Week 2)
4. [ ] 04_spec_generator_PR.md - ToolSpec generation
5. [ ] 05_code_generator_PR.md - Code generation + repair
6. [ ] 06_validator_PR.md - Multi-stage validation

### Phase 3: Execution & Feedback (Week 3)
7. [ ] 07_executor_PR.md - Tool execution
8. [ ] 08_feedback_handler_PR.md - Two-stage approval
9. [ ] 09_promoter_PR.md - Registry management

### Phase 4: Orchestration (Week 4)
10. [ ] 10_pipeline_orchestrator_PR.md - LangGraph integration
11. [ ] End-to-end testing
12. [ ] Performance optimization

---

## 📝 Key Design Decisions

### 1. No Traditional Base Classes
- **Pattern**: Decorator-based composition with `@mcp.tool()`
- **Rationale**: FastMCP uses function decorators, not inheritance
- **Benefit**: Simpler, more flexible tool definitions

### 2. LangGraph for Orchestration
- **Pattern**: StateGraph with conditional edges and interrupts
- **Rationale**: Native support for human-in-the-loop workflows
- **Benefit**: Declarative flow control, easy debugging

### 3. On-Premises LLM (Qwen 2.5-Coder)
- **Pattern**: vLLM server with OpenAI-compatible API
- **Rationale**: Data privacy, cost control, customization
- **Benefit**: Fast inference, no API costs, full control

### 4. State Machine Lifecycle
- **Pattern**: Explicit states (DRAFT → STAGED → APPROVED → PROMOTED)
- **Rationale**: Clear tool maturity tracking
- **Benefit**: Audit trail, rollback capability

### 5. Two-Stage Human Approval
- **Pattern**: Interrupt before feedback_stage1 and feedback_stage2
- **Rationale**: Validate both output quality AND tool worthiness
- **Benefit**: Prevents bad tools from reaching production

---

## 🔗 Dependencies Between Modules

```
01_data_models (foundation)
    ↓
02_llm_client ──→ 03_intent_extraction
    ↓                    ↓
    └──→ 04_spec_generator
              ↓
         05_code_generator ←── sandbox.py ←── 06_validator (repair loop)
              ↓
         07_executor
              ↓
         08_feedback_handler
              ↓
         09_promoter
              ↓
    10_pipeline_orchestrator (integrates all)
```

---

## 📁 Project File Structure

```
MCP_Tool_Code_Interpreter_Generator/
│
├── src/
│   ├── __init__.py
│   ├── models.py                    # 01 - All Pydantic models & state
│   ├── llm_client.py                # 02 - BaseLLMClient & QwenLLMClient
│   ├── intent_extraction.py         # 03 - IntentExtractor class + intent_node
│   ├── spec_generator.py            # 04 - SpecGenerator + spec_generator_node
│   ├── code_generator.py            # 05 - CodeGenerator + code_generator_node + repair_node
│   ├── sandbox.py                   # Sandbox executor for safe code execution
│   ├── validator.py                 # 06 - Validator + validator_node + routing (uses sandbox.py)
│   ├── executor.py                  # 07 - ToolExecutor + executor_node
│   ├── feedback_handler.py          # 08 - feedback_stage1_node + feedback_stage2_node
│   ├── promoter.py                  # 09 - ToolPromoter + promoter_node
│   ├── pipeline.py                  # 10 - LangGraph orchestration + run_pipeline()
│   └── server.py                    # Main MCP server entry point (@mcp.tool() analyze_data)
│
├── tools/                           # Generated tool storage
│   ├── draft/                       # DRAFT status tools
│   ├── staged/                      # STAGED status (post-validation)
│   ├── active/                      # PROMOTED tools (production-ready)
│   └── sandbox/                     # Temporary execution workspace
│       ├── temp_code/               # Generated code copies for testing
│       ├── temp_data/               # Sample data for sandbox runs
│       └── logs/                    # Execution logs for debugging
│
├── registry/
│   └── tools.json                   # RegistryMetadata - all promoted tools catalog
│
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_llm_client.py
│   ├── test_intent_extraction.py
│   ├── test_spec_generator.py
│   ├── test_code_generator.py
│   ├── test_sandbox.py              # Test sandbox isolation
│   ├── test_validator.py
│   ├── test_executor.py
│   ├── test_feedback_handler.py
│   ├── test_promoter.py
│   ├── test_pipeline.py
│   ├── test_integration.py          # End-to-end tests
│   └── fixtures/                    # Test data and mocks
│       ├── sample_queries.json
│       ├── sample_specs.json
│       └── test_data.csv
│
├── reference_files/                 # Existing samples (kept as-is)
│   ├── sample_mcp_tools/
│   ├── sample_planner_output/
│   └── sample_response_to_no_2/
│
├── module_prs/                      # PR documentation (this directory)
│   ├── README.md
│   ├── 01_data_models_PR.md
│   ├── 02_llm_client_PR.md
│   ├── 03_intent_extraction_PR.md
│   ├── 04_spec_generator_PR.md
│   ├── 05_code_generator_PR.md
│   ├── 06_validator_PR.md
│   ├── 07_executor_PR.md
│   ├── 08_feedback_handler_PR.md
│   ├── 09_promoter_PR.md
│   └── 10_pipeline_orchestrator_PR.md
│
├── docs/                            # Architecture documentation
│   ├── AGENTIC_FRAMEWORK_ARCHITECTURE.md
│   ├── LANGGRAPH_ARCHITECTURE.md
│   ├── MODULE_TASK_SPLIT.md
│   └── SANDBOX_SECURITY.md          # Sandbox security guidelines
│
├── data/                            # Sample datasets for testing
│   ├── traffic_accidents.csv
│   └── sample_datasets/
│
├── config/
│   ├── config.yaml                  # LLM endpoints, paths, validation settings
│   ├── sandbox_policy.yaml          # Security restrictions & resource limits
│   └── prompts/                     # Template prompts for LLM calls
│       ├── intent_extraction.txt
│       ├── spec_generation.txt
│       ├── code_generation.txt
│       └── code_repair.txt
│
├── docker/                          # Optional Docker-based sandbox
│   ├── Dockerfile.sandbox
│   ├── docker-compose.sandbox.yml
│   └── README.md
│
├── scripts/                         # Utility scripts
│   ├── setup_environment.sh
│   ├── clean_sandbox.py
│   └── migrate_tools.py
│
├── logs/                            # Application logs
│   └── .gitignore
│
├── .gitignore
├── .env.example                     # Environment variables template
├── README.md                        # Project-level README
├── ProjectRequirements.instructions.md
├── AGENTIC_FRAMEWORK_ARCHITECTURE.md
├── LANGGRAPH_ARCHITECTURE.md
├── MODULE_TASK_SPLIT.md
├── pyproject.toml                   # Modern Python packaging
├── requirements.txt                 # Python dependencies
├── requirements-dev.txt             # Development dependencies
└── run_server.py                    # Entry point: python run_server.py
```

---

## 🧪 Testing Strategy

Each module should include:
1. **Unit Tests**: Individual function/method testing
2. **Integration Tests**: Module interactions (e.g., spec_generator + llm_client)
3. **End-to-End Tests**: Full pipeline execution with sample queries
4. **Validation Tests**: Schema compliance, error handling

---

## 📚 Additional Resources

- **Architecture Docs**:
  - [../AGENTIC_FRAMEWORK_ARCHITECTURE.md](../AGENTIC_FRAMEWORK_ARCHITECTURE.md)
  - [../LANGGRAPH_ARCHITECTURE.md](../LANGGRAPH_ARCHITECTURE.md)
  - [../MODULE_TASK_SPLIT.md](../MODULE_TASK_SPLIT.md)

- **Reference Implementations**:
  - [../reference_files/sample_mcp_tools/](../reference_files/sample_mcp_tools/)
  - [../reference_files/sample_planner_output/](../reference_files/sample_planner_output/)

- **Configuration & Setup**:
  - See [../README.md](../README.md) for installation, configuration, and usage

---

## 👥 Contributing

When adding new modules:
1. Create PR specification in this directory (`##_module_name_PR.md`)
2. Follow existing template structure (Priority, Effort, Core Logic, Checklist)
3. Update this README with module summary and dependencies
4. Add integration tests

---

**Last Updated**: January 27, 2026  
**Version**: 1.0.0  
**Status**: In Development
