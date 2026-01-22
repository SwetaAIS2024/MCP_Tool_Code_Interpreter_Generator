# Module PR 07: Executor

**Module**: `src/executor.py`  
**Priority**: P1 (Execution)  
**Estimated Effort**: 2-3 days  
**Dependencies**: `01_data_models`, `05_code_generator`

---

## 1. Module Purpose

The Executor runs validated tools in isolation and captures:
- **Output** - Tool execution results
- **Artifacts** - Files, plots, intermediate data
- **Metrics** - Execution time, memory usage
- **Errors** - Exceptions and stack traces

**Key Principle**: Execute in isolated environment with resource limits. Never modify active registry during execution.

---

## 2. Core Components

```python
class ToolExecutor:
    """Execute staged tools in isolated environment."""
    
    def __init__(self, staging_dir: Path, max_execution_time: int = 300):
        self.staging_dir = staging_dir
        self.max_execution_time = max_execution_time
    
    def execute(
        self,
        tool_name: str,
        input_data: Dict[str, Any],
        capture_artifacts: bool = True
    ) -> RunArtifacts:
        """
        Execute a staged tool.
        
        Args:
            tool_name: Name of tool to execute
            input_data: Input parameters (includes DataFrame)
            capture_artifacts: Whether to capture outputs
        
        Returns:
            RunArtifacts with results and metadata
        """
        pass
    
    def _load_tool(self, tool_name: str) -> Callable:
        """Load tool function from staging directory."""
        pass
    
    def _execute_with_timeout(
        self,
        func: Callable,
        kwargs: Dict[str, Any],
        timeout: int
    ) -> Dict[str, Any]:
        """Execute function with timeout."""
        pass
    
    def _capture_output(self, result: Any) -> Dict[str, Any]:
        """Convert tool output to serializable format."""
        pass
    
    def _measure_resources(self, start_time: float) -> Dict[str, float]:
        """Measure execution time and memory."""
        pass
```

---

## 3. Implementation

### 3.1 Tool Loading

```python
def _load_tool(self, tool_name: str) -> Callable:
    """Load tool from staging directory."""
    
    tool_path = self.staging_dir / f"{tool_name}.py"
    
    if not tool_path.exists():
        raise FileNotFoundError(f"Tool not found: {tool_name}")
    
    # Load module dynamically
    import importlib.util
    spec = importlib.util.spec_from_file_location(tool_name, tool_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Extract function
    if not hasattr(module, tool_name):
        raise AttributeError(f"Function '{tool_name}' not found in module")
    
    return getattr(module, tool_name)
```

### 3.2 Execution with Timeout

```python
def execute(
    self,
    tool_name: str,
    input_data: Dict[str, Any],
    capture_artifacts: bool = True
) -> RunArtifacts:
    """Execute tool and capture results."""
    
    start_time = time.time()
    
    try:
        # Load tool
        func = self._load_tool(tool_name)
        
        # Execute with timeout
        result = self._execute_with_timeout(
            func,
            input_data,
            self.max_execution_time
        )
        
        # Capture output
        captured = self._capture_output(result)
        
        # Measure resources
        metrics = self._measure_resources(start_time)
        
        return RunArtifacts(
            result=captured,
            summary_markdown=result.get("summary"),
            execution_time=metrics["execution_time"],
            memory_used=metrics.get("memory_mb", 0.0),
            error=None
        )
    
    except TimeoutError:
        return RunArtifacts(
            result={},
            error=f"Execution timeout after {self.max_execution_time}s",
            execution_time=time.time() - start_time
        )
    
    except Exception as e:
        return RunArtifacts(
            result={},
            error=f"{type(e).__name__}: {str(e)}",
            execution_time=time.time() - start_time
        )


def _execute_with_timeout(
    self,
    func: Callable,
    kwargs: Dict[str, Any],
    timeout: int
) -> Dict[str, Any]:
    """Execute with timeout using threading."""
    
    import concurrent.futures
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func, **kwargs)
        try:
            result = future.result(timeout=timeout)
            return result
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Execution exceeded {timeout}s")
```

### 3.3 Output Capture

```python
def _capture_output(self, result: Any) -> Dict[str, Any]:
    """Convert result to JSON-serializable format."""
    
    if isinstance(result, dict):
        # Already structured
        return result
    
    elif isinstance(result, pd.DataFrame):
        # Convert DataFrame to dict
        return {
            "result": result.to_dict(orient="records"),
            "summary": f"DataFrame with {len(result)} rows, {len(result.columns)} columns"
        }
    
    else:
        # Wrap other types
        return {"result": str(result)}
```

---

## 4. Testing

```python
def test_execute_simple_tool():
    """Test execution of simple tool."""
    # Create test tool
    code = '''
import pandas as pd
from typing import Dict, Any

def test_count(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "result": {"count": len(df)},
        "summary": f"Count: {len(df)}"
    }
'''
    
    staging_dir = Path("sandbox/staging")
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / "test_count.py").write_text(code)
    
    executor = ToolExecutor(staging_dir)
    
    result = executor.execute(
        "test_count",
        {"df": pd.DataFrame({"A": [1, 2, 3]})}
    )
    
    assert result.error is None
    assert result.result["result"]["count"] == 3
```

---

## 5. Configuration

```yaml
execution:
  staging_dir: "src/tools/staging"
  max_execution_time: 300  # 5 minutes
  memory_limit_mb: 2048
  capture_artifacts: true
```

---

**Estimated Lines of Code**: 400-500  
**Test Coverage Target**: >85%  
**Ready for Implementation**: ✅
