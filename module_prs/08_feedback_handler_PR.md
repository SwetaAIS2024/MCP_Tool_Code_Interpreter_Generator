# Feedback Handler Module

**Module**: `src/feedback_handler.py`  
**Priority**: P0  
**Effort**: 1 day

---

## LangGraph Nodes

```python
def feedback_stage1_node(state: ToolGeneratorState) -> ToolGeneratorState:
    """Stage 1: Validate output."""
    approved = parse_response(state.get("user_response_stage1", ""))
    return {**state, "stage1_approved": approved}

def feedback_stage2_node(state: ToolGeneratorState) -> ToolGeneratorState:
    """Stage 2: Approve for promotion."""
    approved = parse_response(state.get("user_response_stage2", ""))
    return {**state, "stage2_approved": approved}

def route_after_stage1(state):
    return "feedback_stage2_node" if state["stage1_approved"] else END

def route_after_stage2(state):
    return "promoter_node" if state["stage2_approved"] else END
```

---

## Core Logic

```python
def parse_response(response: str) -> bool:
    normalized = response.strip().lower()
    return normalized in ["yes", "approve", "approved"]
```

---

## Implementation Checklist

- [ ] Implement response parsing
- [ ] Add routing logic
- [ ] Write tests
