"""Pipeline orchestrator module for LangGraph workflow.

This module assembles the complete tool generation pipeline:
1. Intent extraction
2. Spec generation
3. Code generation
4. Validation (with repair loop)
5. Execution
6. Two-stage feedback
7. Promotion to registry
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, END

from .models import ToolGeneratorState
from .intent_extraction import intent_node, route_after_intent
from .spec_generator import spec_generator_node
from .code_generator import code_generator_node, repair_node
from .validator import validator_node, route_after_validation
from .executor import executor_node
from .feedback_handler import (
    feedback_stage1_node,
    feedback_stage2_node,
    route_after_stage1,
    route_after_stage2
)
from .promoter import promoter_node


def build_graph() -> StateGraph:
    """Build and compile the LangGraph StateGraph.
    
    Returns:
        Compiled graph ready for execution
    """
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
    
    # Compile with interrupts for human feedback
    graph = workflow.compile(
        interrupt_before=["feedback_stage1_node", "feedback_stage2_node"]
    )
    
    return graph


def run_pipeline(user_query: str, data_path: str) -> Dict[str, Any]:
    """Execute the complete tool generation pipeline.
    
    Args:
        user_query: Natural language query from user
        data_path: Path to data file to analyze
        
    Returns:
        Final state dict with all pipeline results
    """
    # Build graph
    graph = build_graph()
    
    # Initialize state
    initial_state: ToolGeneratorState = {
        "user_query": user_query,
        "data_path": data_path,
        "repair_attempts": 0,
        "messages": []
    }
    
    # Run graph
    result = graph.invoke(initial_state)
    
    return result