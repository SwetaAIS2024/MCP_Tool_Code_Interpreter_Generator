# Pipeline Orchestrator Module

**Module**: `src/pipeline.py`  
**Priority**: P0  
**Effort**: 3 days

---

## LangGraph StateGraph

```python
from langgraph.graph import StateGraph, END

# Build graph
workflow = StateGraph(ToolGeneratorState)

# Add nodes
workflow.add_node("intent_node", intent_node)
workflow.add_node("spec_generator_node", spec_generator_node)
workflow.add_node("code_generator_node", code_generator_node)
workflow.add_node("validator_node", validator_node)
workflow.add_node("repair_node", repair_node)
workflow.add_node("executor_node", executor_node)
workflow.add_node("feedback_stage1_node", feedback_stage1_node)
workflow.add_node("feedback_stage2_node", feedback_stage2_node)
workflow.add_node("promoter_node", promoter_node)

# Set entry point
workflow.set_entry_point("intent_node")

# Add edges
workflow.add_conditional_edges("intent_node", route_after_intent)
workflow.add_edge("spec_generator_node", "code_generator_node")
workflow.add_edge("code_generator_node", "validator_node")
workflow.add_conditional_edges("validator_node", route_after_validation)
workflow.add_edge("repair_node", "validator_node")
workflow.add_edge("executor_node", "feedback_stage1_node")
workflow.add_conditional_edges("feedback_stage1_node", route_after_stage1)
workflow.add_conditional_edges("feedback_stage2_node", route_after_stage2)
workflow.add_edge("promoter_node", END)

# Compile with interrupts
graph = workflow.compile(interrupt_before=["feedback_stage1_node", "feedback_stage2_node"])
```

---

## Main Execution

```python
def run_pipeline(user_query: str, data_path: str):
    # Initialize state
    initial_state = {
        "user_query": user_query,
        "data_path": data_path,
        "repair_attempts": 0,
        "messages": []
    }
    
    # Run graph
    result = graph.invoke(initial_state)
    
    return result
```

---

## MCP Tool Wrapper

```python
@mcp.tool()
def analyze_data(query: str, file_path: str) -> Dict[str, Any]:
    \"\"\"Analyze data using natural language query.\"\"\"
    result = run_pipeline(query, file_path)
    
    if result.get("promoted_tool"):
        return {
            "status": "success",
            "tool_created": result["promoted_tool"]["name"],
            "result": result["execution_output"]
        }
    else:
        return {
            "status": "failed",
            "error": "Tool generation failed"
        }
```

---

## Configuration

```yaml
# config.yaml
llm:
  base_url: "http://localhost:8000/v1"
  model: "qwen2.5-coder-32b-instruct"

paths:
  staging_dir: "src/tools/staging"
  active_dir: "src/tools/active"
  registry: "src/tools/registry.json"
```

---

## Implementation Checklist

- [ ] Build StateGraph
- [ ] Add all nodes and edges
- [ ] Configure interrupts
- [ ] Create MCP wrapper
- [ ] Write end-to-end tests
