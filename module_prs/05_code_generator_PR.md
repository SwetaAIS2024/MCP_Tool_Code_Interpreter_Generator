# Code Generator Module

**Module**: `src/code_generator.py`  
**Priority**: P0  
**Effort**: 3-4 days

---

## LangGraph Nodes

```python
def code_generator_node(state: ToolGeneratorState) -> ToolGeneratorState:
    """Generate code from spec."""
    code = generate_code(state["tool_spec"])
    
    return {
        **state,
        "generated_code": code,
        "repair_attempts": 0
    }

def repair_node(state: ToolGeneratorState) -> ToolGeneratorState:
    """Repair code based on errors."""
    errors = state["validation_result"].errors
    repaired = repair_code(state["generated_code"], errors)
    
    return {
        **state,
        "generated_code": repaired,
        "repair_attempts": state["repair_attempts"] + 1
    }
```

---

## Core Logic

```python
class CodeGenerator:
    def generate(self, spec: ToolSpec) -> str:
        # Build prompt
        prompt = f\"\"\"Generate Python function:
Name: {spec.tool_name}
Description: {spec.description}
Parameters: {spec.parameters}
Logic: {spec.what_it_does}

Include:
- pandas DataFrame input
- Type hints
- Error handling
- Return Dict[str, Any] with 'result' and 'metadata'
\"\"\"
        
        # Generate code
        code = self.llm.generate(prompt, temperature=0.2)
        
        # Add MCP decorator
        full_code = f\"\"\"
from fastmcp import FastMCP
import pandas as pd
from typing import Dict, Any

mcp = FastMCP("data_analysis_tools")

@mcp.tool()
{code}
\"\"\"
        
        # Format with black
        return black.format_str(full_code, mode=black.FileMode())
```

---

## Implementation Checklist

- [ ] Implement code generation
- [ ] Add MCP decorator wrapper
- [ ] Add code formatting
- [ ] Write tests
