# Validator Module

**Module**: `src/validator.py`  
**Priority**: P0  
**Effort**: 3-4 days

---

## LangGraph Node

```python
def validator_node(state: ToolGeneratorState) -> ToolGeneratorState:
    """Validate generated code."""
    result = validate(state["generated_code"], state["tool_spec"])
    
    return {
        **state,
        "validation_result": result
    }

def route_after_validation(state: ToolGeneratorState):
    if state["validation_result"].is_valid:
        return "executor_node"
    elif state["repair_attempts"] < 3:
        return "repair_node"
    else:
        return END
```

---

## Core Logic

```python
class Validator:
    def validate(self, code: str, spec: ToolSpec) -> ValidationReport:
        errors = []
        
        # 1. Syntax check
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
            return ValidationReport(schema_ok=False, tests_ok=False, sandbox_ok=False, errors=errors)
        
        # 2. Schema compliance
        if not self._check_schema(code, spec):
            errors.append("Schema mismatch")
        
        # 3. Sandbox execution
        try:
            result = self._run_sandbox(code)
            if not isinstance(result, dict):
                errors.append("Invalid return type")
        except Exception as e:
            errors.append(f"Execution error: {e}")
        
        return ValidationReport(
            schema_ok=len(errors) == 0,
            tests_ok=len(errors) == 0,
            sandbox_ok=len(errors) == 0,
            errors=errors
        )
```

---

## Implementation Checklist

- [ ] Implement syntax validation
- [ ] Implement schema validation
- [ ] Implement sandbox execution
- [ ] Write tests
