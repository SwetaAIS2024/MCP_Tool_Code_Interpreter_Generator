# Promoter Module

**Module**: `src/promoter.py`  
**Priority**: P0  
**Effort**: 2 days

---

## LangGraph Node

```python
def promoter_node(state: ToolGeneratorState) -> ToolGeneratorState:
    """Promote tool to active registry."""
    promoted = promote_tool(state["tool_spec"], state["generated_code"])
    
    return {
        **state,
        "promoted_tool": promoted
    }
```

---

## Core Logic

```python
class ToolPromoter:
    def promote(self, spec: ToolSpec, code: str) -> Dict:
        # Handle version conflicts
        final_name = self._resolve_name(spec.tool_name)
        
        # Copy to active directory
        active_path = self.active_dir / f"{final_name}.py"
        active_path.write_text(code)
        
        # Update registry
        self._update_registry(final_name, spec)
        
        return {
            "name": final_name,
            "path": str(active_path),
            "version": spec.version
        }
    
    def _resolve_name(self, name: str) -> str:
        if not (self.active_dir / f"{name}.py").exists():
            return name
        
        # Add version suffix
        version = 2
        while (self.active_dir / f"{name}_v{version}.py").exists():
            version += 1
        return f"{name}_v{version}"
```

---

## Implementation Checklist

- [ ] Implement promotion logic
- [ ] Add version handling
- [ ] Update registry
- [ ] Write tests
