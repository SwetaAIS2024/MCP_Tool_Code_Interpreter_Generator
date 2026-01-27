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
    
    Args:
        state: Current generator state
        
    Returns:
        Updated state with stage1_approved
    """
    # Get user response (set by graph interrupt handling)
    response = state.get("user_response_stage1", "")
    approved = parse_response(response)
    
    return {
        **state,
        "stage1_approved": approved
    }


def feedback_stage2_node(state: ToolGeneratorState) -> ToolGeneratorState:
    """LangGraph node: Stage 2 - Approve tool for promotion to registry.
    
    This node pauses for human review of the complete tool.
    User decides whether tool should be promoted to active registry.
    
    Args:
        state: Current generator state
        
    Returns:
        Updated state with stage2_approved
    """
    # Get user response (set by graph interrupt handling)
    response = state.get("user_response_stage2", "")
    approved = parse_response(response)
    
    return {
        **state,
        "stage2_approved": approved
    }


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
