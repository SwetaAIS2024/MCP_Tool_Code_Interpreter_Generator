"""Feedback Handler Module - Handle two-stage human-in-the-loop approval."""

from src.models import ToolGeneratorState


# ============================================================================
# Response Parser
# ============================================================================

def parse_response(response: str) -> bool:
    """Parse user feedback response.
    
    Args:
        response: User's response string
        
    Returns:
        True if approved, False otherwise
    """
    if not response:
        return False
    
    normalized = response.strip().lower()
    
    # Approval keywords
    approved_keywords = [
        "yes", "y",
        "approve", "approved",
        "accept", "accepted",
        "ok", "okay",
        "true", "1"
    ]
    
    # Rejection keywords
    rejected_keywords = [
        "no", "n",
        "reject", "rejected",
        "deny", "denied",
        "false", "0"
    ]
    
    if normalized in approved_keywords:
        return True
    elif normalized in rejected_keywords:
        return False
    else:
        # Default to rejection for unclear responses
        return False


# ============================================================================
# LangGraph Nodes
# ============================================================================

def feedback_stage1_node(state: ToolGeneratorState) -> ToolGeneratorState:
    """LangGraph node: Stage 1 - Validate execution output quality.
    
    This node pauses for human review of the execution results.
    User validates whether the tool produces correct output.
    
    Note: The approval decision (stage1_approved) is set by update_state()
    when resuming from interrupt. This node just passes through the state.
    
    Args:
        state: Current generator state
        
    Returns:
        Updated state (unchanged, approval already set)
    """
    # Approval is already set by update_state() before this node runs
    # Just pass through - don't overwrite it
    stage1_approved = state.get("stage1_approved", False)
    
    # Debug output
    print(f"[DEBUG feedback_stage1_node] stage1_approved from state = {stage1_approved}")
    
    # Return state unchanged - approval was already set
    return state


def feedback_stage2_node(state: ToolGeneratorState) -> ToolGeneratorState:
    """LangGraph node: Stage 2 - Approve tool for promotion to registry.
    
    This node pauses for human review of the complete tool.
    User decides whether tool should be promoted to active registry.
    
    Note: The approval decision (stage2_approved) is set by update_state()
    when resuming from interrupt. This node just passes through the state.
    
    Args:
        state: Current generator state
        
    Returns:
        Updated state (unchanged, approval already set)
    """
    # Approval is already set by update_state() before this node runs
    # Just pass through - don't overwrite it
    stage2_approved = state.get("stage2_approved", False)
    
    # Debug output
    print(f"[DEBUG feedback_stage2_node] stage2_approved from state = {stage2_approved}")
    
    # Return state unchanged - approval was already set
    return state


# ============================================================================
# Routing Functions
# ============================================================================

def route_after_stage1(state: ToolGeneratorState) -> str:
    """Route after stage 1 feedback.
    
    Args:
        state: Current generator state
        
    Returns:
        Next node name or END
    """
    from langgraph.graph import END
    
    if state.get("stage1_approved", False):
        return "feedback_stage2_node"
    else:
        return END


def route_after_stage2(state: ToolGeneratorState) -> str:
    """Route after stage 2 feedback.
    
    Args:
        state: Current generator state
        
    Returns:
        Next node name or END
    """
    from langgraph.graph import END
    
    if state.get("stage2_approved", False):
        return "promoter_node"
    else:
        return END
