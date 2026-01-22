# Module PRs - Master Index

**Project**: MCP Tool Code Interpreter Generator  
**Main PR**: ProjectRequirements.instructions.md  
**Status**: Module breakdown complete

---

## Overview

This directory contains detailed Project Requirements for each module of the MCP Tool Code Generator. Each module PR is self-contained with:
- Purpose and scope
- Implementation details
- Data structures
- Testing requirements
- Dependencies
- Examples

---

## Module Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                     01_data_models                          │
│  (ToolSpec, ToolCandidate, ValidationReport, CodeMetrics)   │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
│02_llm_client │ │  utils (TBD) │ │ metrics (TBD)    │
│  (Qwen/vLLM) │ │              │ │ code_bleu, etc.  │
└──────┬───────┘ └──────────────┘ └──────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  03_intent_extraction                   │
│  (Parse query, gap detection)           │
└──────────────┬──────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  04_spec_generator (TBD)                     │
│  (Generate ToolSpec from intent)             │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  05_code_generator (TBD)                     │
│  (Generate Python code from spec)            │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  06_validator (TBD)                          │
│  (Schema, sandbox, tests, metrics)           │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  07_executor (TBD)                           │
│  (Run staged tool, capture artifacts)        │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  08_presenter (TBD)                          │
│  (Format output, request approval)           │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  09_feedback_handler (TBD)                   │
│  (Parse user response, decide approval)      │
└──────────────┬───────────────────────────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
┌──────────────┐ ┌──────────────┐
│10_promoter   │ │11_repair     │
│(Register)    │ │(Fix & retry) │
└──────────────┘ └──────────────┘
```

---

## Completed Module PRs

### ✅ 01 - Data Models (`01_data_models_PR.md`)
**Priority**: P0 (Foundation)  
**Status**: Complete  
**Effort**: 2-3 days

**Key Components**:
- `ToolStatus`, `ToolSpec`, `ToolCandidate`
- `CodeMetrics`, `FunctionalCorrectnessMetrics`, `SemanticClosenessMetrics`
- `ValidationReport`, `RunArtifacts`, `UserFeedback`
- `RegistryMetadata`

**Tests**: >95% coverage required

---

### ✅ 02 - LLM Client (`02_llm_client_PR.md`)
**Priority**: P0 (Core service)  
**Status**: Complete  
**Effort**: 2-3 days

**Key Components**:
- `BaseLLMClient` (abstract interface)
- `QwenLLMClient` (on-prem via vLLM)
- `AnthropicClient`, `OpenAIClient` (optional cloud)
- Prompt template management
- Structured JSON generation

**Configuration**: Qwen2.5-Coder-32B (4-bit) for 48GB VRAM

---

### ✅ 03 - Intent Extraction (`03_intent_extraction_PR.md`)
**Priority**: P0 (Entry point)  
**Status**: Complete  
**Effort**: 2-3 days

**Key Components**:
- `UserIntent` model
- `IntentExtractor` (pattern + LLM)
- `GapDetector` (overlap scoring)
- `analyze_user_request()` main function

**Gap Detection**: 5-component weighted scoring (≥85% threshold)

---

### ✅ 04 - Spec Generator (`04_spec_generator_PR.md`)
**Priority**: P0 (Core)  
**Status**: Complete  
**Effort**: 2-3 days

**Key Components**:
- Generate ToolSpec from UserIntent
- JSON Schema generation (input/output)
- Reference tool matching
- LLM-based spec generation

---

### ✅ 05 - Code Generator (`05_code_generator_PR.md`)
**Priority**: P0 (Core)  
**Status**: Complete  
**Effort**: 3-4 days

**Key Components**:
- Generate Python code from ToolSpec
- Template-based + LLM-based generation
- MCP decorator and metadata
- Test generation
- Code formatting (black/isort)

---

### ✅ 06 - Validator (`06_validator_PR.md`)
**Priority**: P0 (Quality gate)  
**Status**: Complete  
**Effort**: 4-5 days

**Key Components**:
- Syntax validation (AST)
- Static analysis (mypy, pylint)
- Schema compliance
- Sandbox execution
- Code metrics (optional)
- Repair suggestions

---

### ✅ 07 - Executor (`07_executor_PR.md`) - THE CODE INTERPRETER CORE
**Priority**: P1 (Execution engine)  
**Status**: Complete  
**Effort**: 2-3 days

**Key Components**:
- **Load and execute generated Python code**
- Isolated environment execution
- Timeout handling
- Resource measurement
- Artifact capture

**Note**: This is the actual "interpreter" that runs generated code

---

### ✅ 08 - Presenter (`08_presenter_PR.md`)
**Priority**: P1 (UI)  
**Status**: Complete  
**Effort**: 1-2 days

**Key Components**:
- Format execution results (markdown)
- Generate approval prompts (2-stage)
- DataFrame preview

---

### ✅ 09 - Feedback Handler (`09_feedback_handler_PR.md`)
**Priority**: P0 (Decision point)  
**Status**: Complete  
**Effort**: 1-2 days

**Key Components**:
- Parse user responses (strict token matching)
- Two-stage approval logic
- Ambiguous response handling

---

### ✅ 10 - Promoter (`10_promoter_PR.md`)
**Priority**: P0 (Registry)  
**Status**: Complete  
**Effort**: 2 days

**Key Components**:
- Promote tools from staging to active
- Version conflict handling
- Registry metadata updates

---

### ✅ 11 - Utils Package (`11_utils_package_PR.md`)
**Priority**: P0 (Foundation)  
**Status**: Complete  
**Effort**: 2-3 days

**Key Components**:
- CSV helpers (load, type detection)
- Type detection (numeric, categorical, datetime)
- Validation helpers
- Security helpers (import allowlist, path validation)

---

### ✅ 12 - Metrics Package (`12_metrics_package_PR.md`)
**Priority**: P2 (Optional for MVP)  
**Status**: Complete  
**Effort**: 5-7 days

**Key Components**:
- Functional correctness
- Pass@k calculation
- Code BLEU (n-gram, AST, dataflow)
- Test pass rate

**Note**: Optional for MVP, recommended for production

---

### ✅ 13 - Pipeline Orchestrator (`13_pipeline_orchestrator_PR.md`) - **THE CODE INTERPRETER**
**Priority**: P0 (CRITICAL - Main Entry Point)  
**Status**: Complete  
**Effort**: 3-4 days

**Key Components**:
- **Main `CodeInterpreterPipeline` class - This IS the "Code Interpreter"**
- Orchestrates all modules (Intent → Spec → Code → Validate → Execute → Approve)
- Gap detection (reuse existing tools)
- Validation with repair loop (max 3 attempts)
- Two-stage approval workflow
- CLI interface (`src/cli.py`)
- MCP server integration (`src/mcp_server.py`)
- Complete state machine
- Error recovery

**Note**: **This is the main interface users interact with to interpret queries and execute generated code**

---

## Remaining Module PRs (To Be Created)

### 🔲 04 - Spec Generator
**Priority**: P0  
**Estimated Effort**: 2 days

**Scope**:
- Generate ToolSpec from UserIntent
- Create JSON schemas (input/output)
- Generate documentation sections
- Validate spec completeness

**Depends on**: 01, 02, 03

---

### 🔲 05 - Code Generator
**Priority**: P0  
**Estimated Effort**: 3 days

**Scope**:
- Generate Python code from ToolSpec
- Apply code templates
- Add error handling
- Generate tests
- Format with black/isort

**Depends on**: 01, 02, 04

---

### 🔲 06 - Validator
**Priority**: P0  
**Estimated Effort**: 4-5 days

**Scope**:
- Schema validation
- Static analysis (mypy, pylint)
- Sandbox execution
- Test case generation and execution
- Code metrics calculation (functional correctness, pass@k, code BLEU)
- Repair loop coordinator

**Depends on**: 01, 02, 05, Metrics modules

---

### 🔲 07 - Executor
**Priority**: P1  
**Estimated Effort**: 2 days

**Scope**:
- Load staged tool in isolation
- Execute with user data
- Capture outputs and artifacts
- Measure execution time
- Handle errors gracefully

**Depends on**: 01, 05

---

### 🔲 08 - Presenter
**Priority**: P1  
**Estimated Effort**: 1-2 days

**Scope**:
- Format tool output (markdown)
- Generate tool summary
- Create approval prompt (2-stage)
- Handle output display

**Depends on**: 01, 07

---

### 🔲 09 - Feedback Handler
**Priority**: P0  
**Estimated Effort**: 1-2 days

**Scope**:
- Parse user responses
- Strict token matching (Approve/Reject)
- Two-stage approval flow
- Extract rejection reasons

**Depends on**: 01

---

### 🔲 10 - Promoter
**Priority**: P0  
**Estimated Effort**: 2 days

**Scope**:
- Copy tool from staging to active
- Update registry metadata
- Version management
- Idempotency checks
- Reload MCP server

**Depends on**: 01

---

### 🔲 11 - Repair Coordinator
**Priority**: P1  
**Estimated Effort**: 2 days

**Scope**:
- Parse validation errors
- Generate repair prompts
- Track repair iterations (max 3)
- Improvement delta tracking

**Depends on**: 02, 06

---

### 🔲 12 - Metrics Package
**Priority**: P2 (Optional for MVP)  
**Estimated Effort**: 5-7 days

**Submodules**:
- `functional_correctness.py` - Reference solution comparison, test execution
- `pass_at_k.py` - Pass@k calculation
- `test_pass_rate.py` - Test suite management
- `code_bleu.py` - Combined Code BLEU
- `ngram_matcher.py` - N-gram and weighted n-gram
- `ast_matcher.py` - AST-based similarity
- `dataflow_analyzer.py` - Variable flow analysis

**Depends on**: 01

---

### 🔲 13 - Pipeline Orchestrator (THE CODE INTERPRETER)
**Priority**: P0 (CRITICAL - Main Entry Point)  
**Estimated Effort**: 3-4 days

**Scope**:
- **This IS the "Code Interpreter"** - main interface users interact with
- Orchestrates all modules: Intent → Spec → Code → Validate → Execute → Approve
- Implements state machine (DRAFT → STAGED → APPROVED → PROMOTED)
- Gap detection (reuse existing tools when possible)
- Validation with automatic repair loop (max 3 attempts)
- Two-stage approval workflow
- CLI interface and MCP server integration
- Error recovery and logging

**Depends on**: All modules (01-12)

---

## Implementation Sequence (Recommended)

### Phase 1: Foundation (Week 1)
1. ✅ Data Models (01)
2. ✅ LLM Client (02)
3. 🔲 Utils Package (13)

### Phase 2: Core Pipeline (Week 2-3)
4. ✅ Intent Extraction (03)
5. 🔲 Spec Generator (04)
6. 🔲 Code Generator (05)

### Phase 3: Validation (Week 3-4)
7. 🔲 Validator (06) - basic (skip Code BLEU for MVP)
8. 🔲 Repair Coordinator (11)

### Phase 4: Execution & Approval (Week 4-5)
9. 🔲 Executor (07)
10. 🔲 Presenter (08)
11. 🔲 Feedback Handler (09)
12. 🔲 Promoter (10)

### Phase 5: Integration (Week 5-6)
13. 🔲 Main Pipeline Orchestrator (14)
14. 🔲 End-to-end testing
15. 🔲 Documentation

### Phase 6: Enhancement (Post-MVP)
16. 🔲 Metrics Package (12) - full Code BLEU implementation
17. 🔲 Advanced features (multi-tool composition, etc.)

---

## Testing Strategy

### Unit Tests
- Each module: >90% coverage
- Mock external dependencies (LLM, file system)
- Test error paths

### Integration Tests
- Module pairs (e.g., Intent → Spec → Code)
- Real LLM calls (with caching)
- Sample dataset processing

### End-to-End Tests
- Complete pipeline: query → tool → approval → promotion
- Multiple tool types (groupby, filter, join, etc.)
- Error recovery scenarios

---

## Next Steps

### Immediate Actions
1. ✅ Review completed PRs (01-03)
2. 🔲 Create PR for Spec Generator (04)
3. 🔲 Create PR for Code Generator (05)
4. 🔲 Create PR for Utils Package (13)
5. 🔲 Set up development environment
6. 🔲 Deploy Qwen2.5-Coder-32B with vLLM

### Week 1 Goals
- Complete foundation modules (Data Models, LLM Client, Utils)
- Unit tests passing for foundation
- vLLM server operational

---

## Module PR Template

Each module PR should contain:

1. **Module Purpose** - What and why
2. **Core Components** - Classes, functions
3. **Data Structures** - Input/output types
4. **Implementation** - Code with examples
5. **Testing Requirements** - Unit + integration tests
6. **Dependencies** - What it needs
7. **Configuration** - YAML/env vars
8. **Implementation Checklist** - Breakdown
9. **Estimated Effort** - Time estimate
10. **Examples** - Usage patterns

---

## Resources

- **Main PR**: `../ProjectRequirements.instructions.md`
- **Reference Files**: `../reference_files/sample_mcp_tools/`
- **Sample Outputs**: `../reference_files/sample_response_to_no_2/`
- **Config Template**: `../config.yaml` (to be created)

---

**Last Updated**: 2026-01-22  
**Next Review**: After completing Phase 1 modules  
**Maintainer**: MCP Tool Generator Team
