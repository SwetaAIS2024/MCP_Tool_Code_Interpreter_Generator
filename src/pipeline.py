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

from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .models import ToolGeneratorState
from .intent_extraction import intent_node, route_after_intent
from .spec_generator import spec_generator_node
from .code_generator import code_generator_node, repair_node
from .logger_config import get_logger

logger = get_logger(__name__)
from .validator import validator_node, route_after_validation
from .executor import executor_node, route_after_execution
from .feedback_handler import (
    feedback_stage1_node,
    feedback_stage2_node,
    route_after_stage1,
    route_after_stage2
)
from .promoter import promoter_node


def build_graph(checkpointer: Optional[MemorySaver] = None) -> StateGraph:
    """Build and compile the LangGraph StateGraph.
    
    Args:
        checkpointer: Optional checkpointer for interrupt handling.
                     If None, creates a MemorySaver for interrupt support.
    
    Returns:
        Compiled graph ready for execution
    """
    # Create checkpointer if not provided (needed for interrupts)
    if checkpointer is None:
        checkpointer = MemorySaver()
    
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
    workflow.add_conditional_edges("executor_node", route_after_execution)
    workflow.add_conditional_edges("feedback_stage1_node", route_after_stage1)
    workflow.add_conditional_edges("feedback_stage2_node", route_after_stage2)
    workflow.add_edge("promoter_node", END)
    
    # Compile with interrupts for human feedback
    graph = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["feedback_stage1_node", "feedback_stage2_node"]
    )
    
    # Generate graph visualization
    try:
        from pathlib import Path
        
        # Get graph structure
        graph_structure = graph.get_graph()
        
        # Save Mermaid diagram
        mermaid = graph_structure.draw_mermaid()
        mermaid_file = Path("pipeline_graph.mmd")
        mermaid_file.parent.mkdir(parents=True, exist_ok=True)
        mermaid_file.write_text(mermaid)
        
        # Try to generate PNG
        try:
            png_data = graph_structure.draw_mermaid_png()
            png_file = Path("pipeline_graph.png")
            png_file.write_bytes(png_data)
            logger.info(f"📊 Graph visualization saved to: {png_file}")
        except Exception:
            logger.info(f"📊 Graph Mermaid diagram saved to: {mermaid_file}")
            logger.info("   (Paste into https://mermaid.live for visualization)")
    except Exception as e:
        logger.warning(f"⚠️  Could not generate graph visualization: {e}")
    
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
        "extracted_intent": None,
        "has_gap": False,
        "tool_spec": None,
        "generated_code": None,
        "validation_result": None,
        "repair_attempts": 0,
        "execution_output": None,
        "stage1_approved": False,
        "stage2_approved": False,
        "promoted_tool": None,
        "messages": []
    }
    
    # class ToolGeneratorState(TypedDict):
    # """State shared across all LangGraph nodes."""
    
    # # Input
    # user_query: str
    # data_path: str
    
    # # Intent
    # extracted_intent: Optional[Dict]
    # has_gap: bool
    
    # # Generation
    # tool_spec: Optional[ToolSpec]
    # generated_code: Optional[str]
    
    # # Validation
    # validation_result: Optional[ValidationReport]
    # repair_attempts: int
    
    # # Execution
    # execution_output: Optional[RunArtifacts]
    
    # # Feedback
    # stage1_approved: bool
    # stage2_approved: bool
    
    # # Final
    # promoted_tool: Optional[Dict]
    # messages: Annotated[List[tuple], add]

    # Run graph
    result = graph.invoke(initial_state)
    
    return result