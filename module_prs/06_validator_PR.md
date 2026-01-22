# Module PR 06: Validator

**Module**: `src/validator.py`  
**Priority**: P0 (Critical - Quality gate)  
**Estimated Effort**: 4-5 days  
**Dependencies**: `01_data_models`, `02_llm_client`, `05_code_generator`, `12_metrics` (optional for MVP)

---

## 1. Module Purpose

The Validator is the quality gate that ensures generated code is:
- **Syntactically correct** - No Python syntax errors
- **Statically sound** - Passes mypy and pylint checks
- **Functionally correct** - Executes successfully in sandbox
- **Schema-compliant** - Matches input/output specifications
- **Performant** - Meets execution time requirements
- **Metrically validated** (optional for MVP) - Code BLEU, pass@k, functional correctness

**Key Principle**: Nothing reaches the user without passing validation. Failed tools trigger repair loop (max 3 attempts).

---

## 2. Core Components

### 2.1 Validator Class

```python
class Validator:
    """Comprehensive validation of generated code."""
    
    def __init__(
        self,
        sandbox_dir: Path,
        llm_client: BaseLLMClient,
        metrics_enabled: bool = False
    ):
        self.sandbox_dir = sandbox_dir
        self.llm = llm_client
        self.metrics_enabled = metrics_enabled
        self.metrics_calculator = CodeMetricsCalculator() if metrics_enabled else None
    
    def validate(
        self,
        candidate: ToolCandidate,
        test_data: Optional[pd.DataFrame] = None
    ) -> ValidationReport:
        """
        Run complete validation pipeline.
        
        Args:
            candidate: Tool to validate
            test_data: Optional sample data for testing
        
        Returns:
            ValidationReport with all validation results
        """
        pass
    
    def _check_syntax(self, code: str) -> ValidationResult:
        """Parse code with AST, check for syntax errors."""
        pass
    
    def _check_static_analysis(self, code_path: Path) -> ValidationResult:
        """Run mypy and pylint."""
        pass
    
    def _check_schema_compliance(
        self,
        code_path: Path,
        spec: ToolSpec
    ) -> ValidationResult:
        """Validate input/output schemas match spec."""
        pass
    
    def _run_sandbox_tests(
        self,
        code_path: Path,
        test_data: Optional[pd.DataFrame]
    ) -> ValidationResult:
        """Execute code in isolated sandbox."""
        pass
    
    def _calculate_metrics(
        self,
        code: str,
        reference_code: Optional[str],
        test_results: Dict
    ) -> CodeMetrics:
        """Calculate code quality metrics (optional)."""
        pass
    
    def _generate_repair_suggestions(
        self,
        validation_report: ValidationReport
    ) -> List[str]:
        """Analyze failures and suggest fixes."""
        pass
```

### 2.2 ValidationResult Model

```python
class ValidationResult(BaseModel):
    """Result of a single validation check."""
    
    check_name: str  # "syntax", "mypy", "sandbox", etc.
    passed: bool
    errors: List[str] = []
    warnings: List[str] = []
    execution_time: float = 0.0
    details: Dict[str, Any] = {}
```

---

## 3. Validation Pipeline

### 3.1 Sequential Validation Stages

```python
def validate(self, candidate: ToolCandidate, test_data: Optional[pd.DataFrame] = None) -> ValidationReport:
    """Run validation pipeline."""
    
    results = []
    start_time = time.time()
    
    # Stage 1: Syntax check (fail fast)
    syntax_result = self._check_syntax(candidate.code)
    results.append(syntax_result)
    if not syntax_result.passed:
        return self._build_report(candidate, results, "SYNTAX_ERROR")
    
    # Stage 2: Save code to temporary file for static analysis
    code_path = self.sandbox_dir / f"{candidate.spec.tool_name}.py"
    code_path.write_text(candidate.code)
    
    # Stage 3: Static analysis (mypy, pylint)
    static_result = self._check_static_analysis(code_path)
    results.append(static_result)
    
    # Stage 4: Schema compliance
    schema_result = self._check_schema_compliance(code_path, candidate.spec)
    results.append(schema_result)
    
    # Stage 5: Sandbox execution
    sandbox_result = self._run_sandbox_tests(code_path, test_data)
    results.append(sandbox_result)
    
    # Stage 6: Code metrics (optional)
    metrics = None
    if self.metrics_enabled:
        metrics = self._calculate_metrics(
            candidate.code,
            reference_code=None,  # Could load from reference tools
            test_results=sandbox_result.details
        )
    
    # Determine overall status
    all_critical_passed = all(
        r.passed for r in results 
        if r.check_name in ["syntax", "sandbox"]
    )
    
    status = "PASSED" if all_critical_passed else "FAILED"
    
    # Build report
    report = ValidationReport(
        tool_name=candidate.spec.tool_name,
        status=status,
        validation_results=results,
        code_metrics=metrics,
        total_time=time.time() - start_time,
        repair_suggestions=self._generate_repair_suggestions(results) if status == "FAILED" else []
    )
    
    return report


def _build_report(
    self,
    candidate: ToolCandidate,
    results: List[ValidationResult],
    status: str
) -> ValidationReport:
    """Build validation report."""
    return ValidationReport(
        tool_name=candidate.spec.tool_name,
        status=status,
        validation_results=results,
        code_metrics=None,
        total_time=sum(r.execution_time for r in results),
        repair_suggestions=self._generate_repair_suggestions(results)
    )
```

### 3.2 Syntax Validation

```python
def _check_syntax(self, code: str) -> ValidationResult:
    """Check Python syntax with AST."""
    
    start = time.time()
    errors = []
    
    try:
        ast.parse(code)
    except SyntaxError as e:
        errors.append(f"Line {e.lineno}: {e.msg}")
    
    return ValidationResult(
        check_name="syntax",
        passed=len(errors) == 0,
        errors=errors,
        execution_time=time.time() - start
    )
```

### 3.3 Static Analysis

```python
def _check_static_analysis(self, code_path: Path) -> ValidationResult:
    """Run mypy and pylint."""
    
    start = time.time()
    errors = []
    warnings = []
    
    # Run mypy
    try:
        result = subprocess.run(
            ["mypy", str(code_path), "--strict"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            mypy_errors = result.stdout.strip().split('\n')
            errors.extend([f"mypy: {e}" for e in mypy_errors if e])
    except subprocess.TimeoutExpired:
        errors.append("mypy: Timeout after 30s")
    
    # Run pylint
    try:
        result = subprocess.run(
            ["pylint", str(code_path), "--score=no"],
            capture_output=True,
            text=True,
            timeout=30
        )
        # Pylint exit codes: 0=clean, 1-4=messages, >4=error
        if result.returncode > 4:
            errors.append(f"pylint: Fatal error (code {result.returncode})")
        elif result.returncode > 0:
            pylint_warnings = result.stdout.strip().split('\n')
            warnings.extend([f"pylint: {w}" for w in pylint_warnings if w])
    except subprocess.TimeoutExpired:
        warnings.append("pylint: Timeout after 30s")
    
    return ValidationResult(
        check_name="static_analysis",
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        execution_time=time.time() - start,
        details={"mypy_checked": True, "pylint_checked": True}
    )
```

### 3.4 Schema Compliance Validation

```python
def _check_schema_compliance(self, code_path: Path, spec: ToolSpec) -> ValidationResult:
    """Validate function signature matches input schema."""
    
    start = time.time()
    errors = []
    
    # Parse code to extract function signature
    with open(code_path) as f:
        tree = ast.parse(f.read())
    
    # Find the main function
    func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == spec.tool_name:
            func = node
            break
    
    if not func:
        errors.append(f"Function '{spec.tool_name}' not found in code")
        return ValidationResult(
            check_name="schema_compliance",
            passed=False,
            errors=errors,
            execution_time=time.time() - start
        )
    
    # Extract parameter names
    param_names = [arg.arg for arg in func.args.args]
    
    # Check against input schema
    required_params = spec.input_schema.get("required", [])
    schema_props = spec.input_schema.get("properties", {}).keys()
    
    missing = set(required_params) - set(param_names)
    if missing:
        errors.append(f"Missing required parameters: {missing}")
    
    extra = set(param_names) - set(schema_props)
    if extra:
        errors.append(f"Unexpected parameters: {extra}")
    
    return ValidationResult(
        check_name="schema_compliance",
        passed=len(errors) == 0,
        errors=errors,
        execution_time=time.time() - start,
        details={
            "function_params": param_names,
            "schema_params": list(schema_props)
        }
    )
```

### 3.5 Sandbox Execution

```python
def _run_sandbox_tests(
    self,
    code_path: Path,
    test_data: Optional[pd.DataFrame]
) -> ValidationResult:
    """Execute code in isolated environment."""
    
    start = time.time()
    errors = []
    
    # Create isolated namespace
    namespace = {
        "pd": pd,
        "Dict": Dict,
        "Any": Any,
        "Optional": Optional
    }
    
    try:
        # Execute code to load function
        with open(code_path) as f:
            code_content = f.read()
        
        # Remove MCP decorator for testing
        code_content = re.sub(r'@mcp\.tool\(\)', '', code_content)
        code_content = re.sub(r'mcp = FastMCP\([^)]+\)', '', code_content)
        
        exec(code_content, namespace)
        
        # Extract function
        func_name = code_path.stem
        if func_name not in namespace:
            errors.append(f"Function '{func_name}' not found after execution")
            return ValidationResult(
                check_name="sandbox",
                passed=False,
                errors=errors,
                execution_time=time.time() - start
            )
        
        func = namespace[func_name]
        
        # Test with sample data
        if test_data is not None:
            try:
                result = func(test_data)
                
                # Validate output structure
                if not isinstance(result, dict):
                    errors.append(f"Expected dict output, got {type(result)}")
                elif "result" not in result:
                    errors.append("Output missing required 'result' key")
                
            except Exception as e:
                errors.append(f"Execution error: {str(e)}")
        
    except Exception as e:
        errors.append(f"Code loading error: {str(e)}")
    
    return ValidationResult(
        check_name="sandbox",
        passed=len(errors) == 0,
        errors=errors,
        execution_time=time.time() - start,
        details={"test_data_provided": test_data is not None}
    )
```

### 3.6 Repair Suggestions

```python
def _generate_repair_suggestions(self, results: List[ValidationResult]) -> List[str]:
    """Generate actionable repair suggestions from errors."""
    
    suggestions = []
    
    for result in results:
        if not result.passed:
            if result.check_name == "syntax":
                suggestions.append("Fix syntax errors in the code")
                suggestions.extend([f"  - {e}" for e in result.errors[:3]])
            
            elif result.check_name == "static_analysis":
                suggestions.append("Address static analysis issues:")
                suggestions.extend([f"  - {e}" for e in result.errors[:5]])
            
            elif result.check_name == "schema_compliance":
                suggestions.append("Update function signature to match spec:")
                suggestions.extend([f"  - {e}" for e in result.errors])
            
            elif result.check_name == "sandbox":
                suggestions.append("Fix runtime errors:")
                suggestions.extend([f"  - {e}" for e in result.errors[:3]])
    
    return suggestions
```

---

## 4. Code Metrics (Optional for MVP)

### 4.1 Metrics Calculator Integration

```python
def _calculate_metrics(
    self,
    code: str,
    reference_code: Optional[str],
    test_results: Dict
) -> CodeMetrics:
    """Calculate code quality metrics."""
    
    if not self.metrics_calculator:
        return None
    
    # Functional correctness (from test results)
    functional = FunctionalCorrectnessMetrics(
        tests_passed=test_results.get("tests_passed", 0),
        tests_total=test_results.get("tests_total", 0),
        pass_at_k_scores={}  # Computed separately
    )
    
    # Semantic closeness (if reference available)
    semantic = None
    if reference_code:
        semantic = self.metrics_calculator.calculate_code_bleu(
            candidate_code=code,
            reference_code=reference_code
        )
    
    return CodeMetrics(
        functional_correctness=functional,
        semantic_closeness=semantic
    )
```

---

## 5. Testing Requirements

### 5.1 Unit Tests

```python
def test_syntax_validation():
    """Test syntax error detection."""
    validator = Validator(Path("sandbox"), MockLLMClient())
    
    invalid_code = "def foo(:\n  return x"
    result = validator._check_syntax(invalid_code)
    
    assert not result.passed
    assert len(result.errors) > 0
    assert "syntax" in result.errors[0].lower()


def test_schema_compliance():
    """Test schema validation."""
    spec = ToolSpec(
        tool_name="test_func",
        description="Test",
        input_schema={
            "type": "object",
            "properties": {"df": {}, "col": {}},
            "required": ["df", "col"]
        },
        output_schema={},
        constraints=[]
    )
    
    code = "def test_func(df, col):\n    return {}"
    code_path = Path("test.py")
    code_path.write_text(code)
    
    validator = Validator(Path("sandbox"), MockLLMClient())
    result = validator._check_schema_compliance(code_path, spec)
    
    assert result.passed
    code_path.unlink()


def test_sandbox_execution():
    """Test code execution in sandbox."""
    code = '''
import pandas as pd
from typing import Dict, Any

def test_tool(df: pd.DataFrame) -> Dict[str, Any]:
    return {"result": df.to_dict(), "summary": "OK"}
'''
    
    code_path = Path("sandbox/test_tool.py")
    code_path.parent.mkdir(exist_ok=True)
    code_path.write_text(code)
    
    validator = Validator(Path("sandbox"), MockLLMClient())
    test_df = pd.DataFrame({"A": [1, 2, 3]})
    
    result = validator._run_sandbox_tests(code_path, test_df)
    
    assert result.passed
    code_path.unlink()
```

---

## 6. Configuration

```yaml
validation:
  sandbox_dir: "sandbox"
  
  checks:
    syntax: true
    mypy: true
    pylint: true
    schema_compliance: true
    sandbox_execution: true
  
  timeouts:
    mypy: 30
    pylint: 30
    sandbox: 60
  
  metrics:
    enabled: false  # MVP: false, Production: true
    code_bleu: true
    functional_correctness: true
    pass_at_k: true
  
  repair:
    max_iterations: 3
    auto_repair: true
```

---

## 7. Dependencies

### 7.1 Internal
- `src/models.py` - ToolCandidate, ValidationReport, CodeMetrics
- `src/llm_client.py` - For repair suggestions
- `src/metrics/` (optional) - Code quality metrics

### 7.2 External
```txt
mypy>=1.8.0
pylint>=3.0.0
pandas>=2.0.0
```

---

## 8. Implementation Checklist

- [ ] Create `Validator` class
- [ ] Implement syntax validation with AST
- [ ] Implement static analysis (mypy, pylint)
- [ ] Implement schema compliance validation
- [ ] Implement sandbox execution
- [ ] Implement repair suggestion generation
- [ ] Add metrics calculator integration (optional)
- [ ] Add unit tests (>90% coverage)
- [ ] Add integration tests with real tools
- [ ] Test repair loop coordination
- [ ] Document all validation checks

---

## 9. Example Usage

```python
from src.validator import Validator
from src.models import ToolCandidate, ToolSpec

# Initialize
validator = Validator(
    sandbox_dir=Path("sandbox"),
    llm_client=llm_client,
    metrics_enabled=False  # MVP: disabled
)

# Validate candidate
candidate = ToolCandidate(
    spec=spec,
    code=generated_code,
    status=ToolStatus.DRAFT
)

report = validator.validate(candidate, test_data=sample_df)

if report.status == "PASSED":
    print("✓ Validation passed")
else:
    print("✗ Validation failed:")
    for suggestion in report.repair_suggestions:
        print(f"  {suggestion}")
```

---

**Estimated Lines of Code**: 700-900  
**Test Coverage Target**: >90%  
**Ready for Implementation**: ✅
