# Module PR 13: Pipeline Orchestrator (Code Interpreter Pipeline)

**Module**: `src/pipeline.py`  
**Priority**: P0 (Main entry point)  
**Estimated Effort**: 3-4 days  
**Dependencies**: ALL modules (01-12)

---

## 1. Module Purpose

The **Pipeline Orchestrator** is the main entry point that coordinates the complete **Code Generator + Interpreter** workflow:

1. **Extract** user intent from natural language
2. **Check** if existing tool can handle it (gap detection)
3. **Generate** tool specification
4. **Generate** Python code
5. **Validate** code (syntax, tests, sandbox)
6. **Execute** code with user data
7. **Present** results to user
8. **Collect** user approval
9. **Promote** approved tools to registry

**This is the "Code Interpreter" that users interact with.**

---

## 2. Core Components

### 2.1 CodeInterpreterPipeline Class

```python
class CodeInterpreterPipeline:
    """
    Main pipeline orchestrating code generation and interpretation.
    
    This is the entry point for the MCP Tool Code Interpreter.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        # Initialize all components
        self.llm_client = self._init_llm_client()
        self.intent_extractor = IntentExtractor(self.llm_client)
        self.gap_detector = GapDetector(config.registry_dir)
        self.spec_generator = SpecGenerator(self.llm_client, config.reference_tools_dir)
        self.code_generator = CodeGenerator(self.llm_client, config.templates_dir)
        self.validator = Validator(config.sandbox_dir, self.llm_client, metrics_enabled=False)
        self.executor = ToolExecutor(config.staging_dir)
        self.presenter = ResultPresenter()
        self.feedback_handler = FeedbackHandler()
        self.promoter = ToolPromoter(config.staging_dir, config.active_dir, config.registry_file)
        
        # State management
        self.current_candidate: Optional[ToolCandidate] = None
        self.repair_attempts = 0
        self.max_repair_attempts = 3
    
    def process_query(
        self,
        user_query: str,
        data_path: Optional[Path] = None
    ) -> str:
        """
        Main entry point: Process user query and return results.
        
        This is the "Code Interpreter" interface.
        
        Args:
            user_query: Natural language data analysis request
            data_path: Path to CSV data file
        
        Returns:
            Formatted response (markdown)
        """
        pass
    
    def _run_pipeline(
        self,
        user_query: str,
        data_path: Optional[Path]
    ) -> RunArtifacts:
        """Execute the complete pipeline."""
        pass
    
    def _check_for_existing_tool(self, intent: UserIntent) -> Optional[str]:
        """Check if existing tool can handle this intent."""
        pass
    
    def _generate_new_tool(self, intent: UserIntent) -> ToolCandidate:
        """Generate new tool from intent."""
        pass
    
    def _validate_and_repair(self, candidate: ToolCandidate) -> ValidationReport:
        """Validate with repair loop."""
        pass
    
    def _execute_tool(
        self,
        tool_name: str,
        data_path: Path
    ) -> RunArtifacts:
        """Execute tool with user data."""
        pass
    
    def _request_approval(
        self,
        tool_name: str,
        artifacts: RunArtifacts,
        spec: ToolSpec
    ) -> UserFeedback:
        """Two-stage approval process."""
        pass
    
    def _handle_approval(self, feedback: UserFeedback, tool_name: str) -> str:
        """Promote or archive based on approval."""
        pass
```

---

## 3. Implementation: Main Pipeline Flow

### 3.1 Process Query (Main Entry Point)

```python
def process_query(
    self,
    user_query: str,
    data_path: Optional[Path] = None
) -> str:
    """
    Main Code Interpreter interface.
    
    Example:
        >>> pipeline = CodeInterpreterPipeline(config)
        >>> result = pipeline.process_query(
        ...     "Group traffic accidents by state and severity",
        ...     data_path=Path("traffic_accidents.csv")
        ... )
        >>> print(result)
        # Formatted markdown with results and approval prompt
    """
    try:
        # Run the complete pipeline
        artifacts = self._run_pipeline(user_query, data_path)
        
        # Format and return results
        if artifacts.error:
            return f"❌ **Error**: {artifacts.error}"
        
        # Present results and request approval
        output = self.presenter.present_results(
            self.current_candidate.spec.tool_name,
            artifacts,
            self.current_candidate.spec
        )
        
        # Add approval prompt
        approval_prompt = self.presenter.generate_approval_prompt(
            self.current_candidate.spec.tool_name,
            stage=1
        )
        
        return output + "\n" + approval_prompt
    
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return f"❌ **Pipeline Error**: {str(e)}"
```

### 3.2 Complete Pipeline Execution

```python
def _run_pipeline(
    self,
    user_query: str,
    data_path: Optional[Path]
) -> RunArtifacts:
    """Execute all pipeline stages."""
    
    # Stage 1: Extract Intent
    logger.info("Stage 1: Extracting user intent")
    intent = self.intent_extractor.extract(user_query)
    
    # Stage 2: Check for Existing Tool (Gap Detection)
    logger.info("Stage 2: Checking for existing tools")
    existing_tool = self._check_for_existing_tool(intent)
    
    if existing_tool:
        logger.info(f"Using existing tool: {existing_tool}")
        # Execute existing tool directly
        return self._execute_tool(existing_tool, data_path)
    
    # Stage 3: Generate Tool Spec
    logger.info("Stage 3: Generating tool specification")
    spec = self.spec_generator.generate(intent)
    
    # Stage 4: Generate Code
    logger.info("Stage 4: Generating Python code")
    generated_code = self.code_generator.generate(spec)
    
    # Create candidate
    self.current_candidate = ToolCandidate(
        spec=spec,
        code=generated_code.code,
        status=ToolStatus.DRAFT,
        created_at=datetime.now()
    )
    
    # Stage 5: Validate with Repair Loop
    logger.info("Stage 5: Validating generated code")
    validation_report = self._validate_and_repair(self.current_candidate)
    
    if validation_report.status != "PASSED":
        raise ValueError(f"Validation failed after {self.max_repair_attempts} attempts")
    
    # Stage 6: Save to Staging
    logger.info("Stage 6: Saving to staging directory")
    staging_path = self.config.staging_dir / f"{spec.tool_name}.py"
    staging_path.write_text(self.current_candidate.code)
    
    self.current_candidate.status = ToolStatus.STAGED
    
    # Stage 7: Execute with User Data
    logger.info("Stage 7: Executing staged tool")
    if data_path:
        df = load_csv_data_with_types(data_path)
        artifacts = self.executor.execute(
            spec.tool_name,
            {"df": df}
        )
    else:
        # No data provided, return success message
        artifacts = RunArtifacts(
            result={"message": "Tool created successfully"},
            summary_markdown="Tool is ready for execution",
            execution_time=0.0
        )
    
    return artifacts
```

### 3.3 Gap Detection (Reuse Existing Tools)

```python
def _check_for_existing_tool(self, intent: UserIntent) -> Optional[str]:
    """
    Check if existing tool can handle this intent.
    
    Returns:
        Tool name if found with overlap >= 0.85, else None
    """
    result = self.gap_detector.check_gap(intent)
    
    if not result.has_gap and result.overlap_score >= 0.85:
        logger.info(f"Found existing tool with {result.overlap_score:.2%} overlap")
        return result.closest_tool
    
    logger.info(f"No suitable existing tool (max overlap: {result.overlap_score:.2%})")
    return None
```

### 3.4 Validation with Repair Loop

```python
def _validate_and_repair(self, candidate: ToolCandidate) -> ValidationReport:
    """
    Validate code with automatic repair.
    
    Retry up to 3 times with LLM-based fixes.
    """
    self.repair_attempts = 0
    
    while self.repair_attempts < self.max_repair_attempts:
        logger.info(f"Validation attempt {self.repair_attempts + 1}/{self.max_repair_attempts}")
        
        # Run validation
        report = self.validator.validate(candidate)
        
        if report.status == "PASSED":
            logger.info("✓ Validation passed")
            return report
        
        # Validation failed - attempt repair
        self.repair_attempts += 1
        
        if self.repair_attempts >= self.max_repair_attempts:
            logger.error("Max repair attempts reached")
            return report
        
        logger.warning(f"Validation failed, attempting repair {self.repair_attempts}")
        
        # Generate repair prompt
        repair_prompt = self._build_repair_prompt(candidate, report)
        
        # Ask LLM to fix code
        repaired_code = self.llm_client.generate(
            prompt=repair_prompt,
            temperature=0.3,
            max_tokens=2000
        )
        
        # Extract code block
        repaired_code = self._extract_code_block(repaired_code)
        
        # Update candidate
        candidate.code = repaired_code
    
    return report


def _build_repair_prompt(
    self,
    candidate: ToolCandidate,
    report: ValidationReport
) -> str:
    """Build prompt for LLM to repair code."""
    
    errors = "\n".join([
        f"- {r.check_name}: {', '.join(r.errors)}"
        for r in report.validation_results
        if not r.passed
    ])
    
    return f"""The following code has validation errors. Please fix them.

## Original Code
```python
{candidate.code}
```

## Validation Errors
{errors}

## Repair Suggestions
{chr(10).join(f"- {s}" for s in report.repair_suggestions)}

## Requirements
1. Fix ALL validation errors
2. Maintain the same functionality
3. Keep the function signature unchanged
4. Return ONLY the corrected code, no explanations

Generate the fixed code:"""
```

### 3.5 Two-Stage Approval

```python
def _request_approval(
    self,
    tool_name: str,
    artifacts: RunArtifacts,
    spec: ToolSpec
) -> UserFeedback:
    """
    Two-stage approval process.
    
    Stage 1: "Is the output correct?" (Yes/No)
    Stage 2: "Approve tool registration?" (Approve/Reject)
    """
    # Stage 1: Output validation
    stage1_prompt = self.presenter.generate_approval_prompt(tool_name, stage=1)
    print(stage1_prompt)
    
    stage1_response = input().strip()
    output_correct = self.feedback_handler.parse_stage1_response(stage1_response)
    
    if not output_correct:
        return UserFeedback(
            decision="REJECTED",
            reason="User indicated output is incorrect",
            timestamp=datetime.now()
        )
    
    # Stage 2: Registration approval
    stage2_prompt = self.presenter.generate_approval_prompt(tool_name, stage=2)
    print(stage2_prompt)
    
    stage2_response = input().strip()
    feedback = self.feedback_handler.parse_stage2_response(stage2_response)
    
    return feedback


def _handle_approval(self, feedback: UserFeedback, tool_name: str) -> str:
    """Handle user approval decision."""
    
    if feedback.decision == "APPROVED":
        # Promote tool to active registry
        result = self.promoter.promote(tool_name)
        
        if result.success:
            return f"""
✅ **Tool Registered Successfully**

Tool `{result.final_name}` has been added to the active registry and is now available for future use.

Path: {result.active_path}
"""
        else:
            return f"❌ **Promotion Failed**: {result.error}"
    
    else:
        # Archive rejected tool
        staging_path = self.config.staging_dir / f"{tool_name}.py"
        archive_path = self.config.archive_dir / f"{tool_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
        
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        staging_path.rename(archive_path)
        
        return f"""
❌ **Tool Rejected**

Reason: {feedback.reason}

The tool has been archived to: {archive_path}
"""
```

---

## 4. Configuration

### 4.1 PipelineConfig

```python
@dataclass
class PipelineConfig:
    """Configuration for the code interpreter pipeline."""
    
    # LLM settings
    llm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "qwen2.5-coder-32b-instruct-awq"
    
    # Directory paths
    staging_dir: Path = Path("src/tools/staging")
    active_dir: Path = Path("src/tools/active")
    archive_dir: Path = Path("src/tools/archive")
    sandbox_dir: Path = Path("sandbox")
    reference_tools_dir: Path = Path("reference_files/sample_mcp_tools")
    templates_dir: Path = Path("templates/code")
    
    # Registry
    registry_dir: Path = Path("registry")
    registry_file: Path = Path("registry/tools.json")
    
    # Validation settings
    max_repair_attempts: int = 3
    metrics_enabled: bool = False
    
    # Execution settings
    max_execution_time: int = 300
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> "PipelineConfig":
        """Load configuration from YAML file."""
        import yaml
        with open(config_path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
```

### 4.2 config.yaml

```yaml
# MCP Tool Code Interpreter Configuration

llm:
  base_url: "http://localhost:8000/v1"
  model: "qwen2.5-coder-32b-instruct-awq"

paths:
  staging_dir: "src/tools/staging"
  active_dir: "src/tools/active"
  archive_dir: "src/tools/archive"
  sandbox_dir: "sandbox"
  reference_tools_dir: "reference_files/sample_mcp_tools"
  templates_dir: "templates/code"
  registry_dir: "registry"
  registry_file: "registry/tools.json"

validation:
  max_repair_attempts: 3
  metrics_enabled: false

execution:
  max_execution_time: 300
```

---

## 5. Usage Examples

### 5.1 Basic Usage (Interactive)

```python
from src.pipeline import CodeInterpreterPipeline, PipelineConfig
from pathlib import Path

# Initialize pipeline
config = PipelineConfig.from_yaml(Path("config.yaml"))
interpreter = CodeInterpreterPipeline(config)

# Process query
result = interpreter.process_query(
    user_query="Group traffic accidents by state and severity, show counts",
    data_path=Path("data/traffic_accidents.csv")
)

print(result)
# Output: Formatted results + approval prompt

# User responds to approval prompts
# Stage 1: "Yes" (output is correct)
# Stage 2: "Approve" (register tool)
```

### 5.2 CLI Interface

```python
# src/cli.py

import click
from pathlib import Path
from src.pipeline import CodeInterpreterPipeline, PipelineConfig


@click.command()
@click.option('--query', '-q', required=True, help='Data analysis query')
@click.option('--data', '-d', type=click.Path(exists=True), help='Path to CSV data')
@click.option('--config', '-c', default='config.yaml', help='Config file path')
def run(query: str, data: str, config: str):
    """MCP Tool Code Interpreter - Generate and execute data analysis tools."""
    
    # Load config
    cfg = PipelineConfig.from_yaml(Path(config))
    
    # Initialize pipeline
    interpreter = CodeInterpreterPipeline(cfg)
    
    # Process query
    data_path = Path(data) if data else None
    result = interpreter.process_query(query, data_path)
    
    print(result)


if __name__ == '__main__':
    run()
```

**Usage**:
```bash
python -m src.cli --query "Group by state, count accidents" --data traffic_accidents.csv
```

### 5.3 MCP Server Integration

```python
# src/mcp_server.py

from fastmcp import FastMCP
from src.pipeline import CodeInterpreterPipeline, PipelineConfig

mcp = FastMCP("code_interpreter")
interpreter = CodeInterpreterPipeline(PipelineConfig.from_yaml("config.yaml"))


@mcp.tool()
def interpret_and_execute(query: str, data_path: str) -> dict:
    """
    Code interpreter: Generate and execute data analysis code from natural language.
    
    Args:
        query: Natural language description of analysis
        data_path: Path to CSV data file
    
    Returns:
        Execution results and approval status
    """
    result = interpreter.process_query(query, Path(data_path))
    
    return {
        "output": result,
        "status": "awaiting_approval"
    }


if __name__ == "__main__":
    mcp.run()
```

---

## 6. State Machine

```
┌─────────────┐
│ USER QUERY  │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ EXTRACT INTENT   │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐      ┌──────────────┐
│  GAP DETECTION   │─────→│ Use Existing │
└──────┬───────────┘      │    Tool      │
       │ (no match)       └──────┬───────┘
       │                         │
       ▼                         │
┌──────────────────┐             │
│  GENERATE SPEC   │             │
└──────┬───────────┘             │
       │                         │
       ▼                         │
┌──────────────────┐             │
│  GENERATE CODE   │             │
└──────┬───────────┘             │
       │                         │
       ▼                         │
┌──────────────────┐             │
│    VALIDATE      │             │
│  (with repair)   │             │
└──────┬───────────┘             │
       │ PASSED                  │
       ▼                         │
┌──────────────────┐             │
│   SAVE STAGING   │             │
└──────┬───────────┘             │
       │                         │
       ├─────────────────────────┘
       │
       ▼
┌──────────────────┐
│    EXECUTE       │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  PRESENT RESULTS │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ APPROVAL STAGE 1 │
│ (output correct?)│
└──────┬───────────┘
       │ Yes
       ▼
┌──────────────────┐
│ APPROVAL STAGE 2 │
│ (register tool?) │
└──────┬───────────┘
       │
   ┌───┴────┐
   │        │
Approve  Reject
   │        │
   ▼        ▼
┌────┐  ┌─────┐
│PROM│  │ARCH │
│OTE │  │IVE  │
└────┘  └─────┘
```

---

## 7. Error Handling & Recovery

```python
class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass


class IntentExtractionError(PipelineError):
    """Failed to extract user intent."""
    pass


class ValidationError(PipelineError):
    """Code validation failed after max retries."""
    pass


class ExecutionError(PipelineError):
    """Tool execution failed."""
    pass
```

---

## 8. Testing

### 8.1 End-to-End Test

```python
def test_complete_pipeline():
    """Test complete pipeline from query to execution."""
    
    config = PipelineConfig(
        staging_dir=Path("test_sandbox/staging"),
        active_dir=Path("test_sandbox/active")
    )
    
    pipeline = CodeInterpreterPipeline(config)
    
    # Mock user approval
    with patch('builtins.input', side_effect=["Yes", "Approve"]):
        result = pipeline.process_query(
            "Count rows by state",
            data_path=Path("test_data.csv")
        )
    
    assert "✓ Tool Executed Successfully" in result
```

---

## 9. Implementation Checklist

- [ ] Create `PipelineConfig` dataclass
- [ ] Create `CodeInterpreterPipeline` class
- [ ] Implement `process_query()` main entry point
- [ ] Implement `_run_pipeline()` orchestration
- [ ] Implement gap detection integration
- [ ] Implement validation with repair loop
- [ ] Implement two-stage approval flow
- [ ] Create CLI interface (`src/cli.py`)
- [ ] Create MCP server interface (`src/mcp_server.py`)
- [ ] Add comprehensive logging
- [ ] Add error handling and recovery
- [ ] Create end-to-end tests
- [ ] Create example notebooks
- [ ] Write user documentation

---

## 10. Performance Metrics

| Stage | Typical Time |
|-------|-------------|
| Intent Extraction | 1-2s |
| Gap Detection | <100ms |
| Spec Generation | 2-5s |
| Code Generation | 3-7s |
| Validation | 5-10s |
| Execution | 0.5-5s |
| **Total** | **12-30s** |

---

**Estimated Lines of Code**: 800-1000  
**Test Coverage Target**: >85%  
**Ready for Implementation**: ✅

---

## Notes

This is the **main interface** to the Code Interpreter system. Users interact with this pipeline, which coordinates all other modules to:
1. Understand natural language queries
2. Generate executable Python code
3. Validate and test the code
4. Execute it with real data
5. Present results for approval
6. Register approved tools for reuse

**This module IS the "Code Interpreter"** - it interprets user intent and generates/executes code to fulfill it.
