# Executor Module

**Module**: `src/executor.py`  
**Priority**: P1  
**Effort**: 2 days

---

## LangGraph Node

```python
def executor_node(state: ToolGeneratorState) -> ToolGeneratorState:
    """Execute tool and capture results."""
    result = execute_tool(state["generated_code"], state["data_path"])
    
    return {
        **state,
        "execution_output": result
    }
```

---

## Core Logic

```python
class ToolExecutor:
    def execute(self, code_path: str, data_path: str) -> RunArtifacts:
        start = time.time()
        
        try:
            # Load tool function
            func = self._load_function(code_path)
            
            # Execute with timeout
            result = self._execute_with_timeout(func, {"file_path": data_path}, timeout=300)
            
            return RunArtifacts(
                result=result,
                summary_markdown=result.get("summary"),
                execution_time_ms=(time.time() - start) * 1000
            )
        except Exception as e:
            return RunArtifacts(
                result={},
                execution_time_ms=(time.time() - start) * 1000,
                error=str(e)
            )
```

---

## Implementation Checklist

- [ ] Implement tool loading
- [ ] Add timeout handling
- [ ] Add artifact capture
- [ ] Write tests
