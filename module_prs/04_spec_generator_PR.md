# Spec Generator Module

**Module**: `src/spec_generator.py`  
**Priority**: P0  
**Effort**: 2-3 days

---

## LangGraph Node

```python
def spec_generator_node(state: ToolGeneratorState) -> ToolGeneratorState:
    """Generate ToolSpec from intent."""
    spec = generate_spec(state["extracted_intent"])
    
    return {
        **state,
        "tool_spec": spec
    }
```

---

## Core Logic

```python
class SpecGenerator:
    def generate(self, intent: Dict) -> ToolSpec:
        # Build prompt
        prompt = f\"\"\"Generate ToolSpec for:
Operation: {intent['operation']}
Columns: {intent['columns']}
Metrics: {intent.get('metrics', [])}

Return JSON with: tool_name, description, input_schema, output_schema, parameters
\"\"\"
        
        # Generate with LLM
        spec_dict = self.llm.generate_structured(prompt, TOOLSPEC_SCHEMA)
        
        # Validate and return
        return ToolSpec(**spec_dict)
```

---

## Implementation Checklist

- [ ] Implement spec generation
- [ ] Add schema inference
- [ ] Add validation
- [ ] Write tests
