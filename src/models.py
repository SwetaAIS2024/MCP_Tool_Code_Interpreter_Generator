"""Data Models Module - Core Pydantic models and LangGraph state definitions."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from operator import add


# ============================================================================
# Enums
# ============================================================================

class ToolStatus(str, Enum):
    """Tool lifecycle status."""
    DRAFT = "DRAFT"
    STAGED = "STAGED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    PROMOTED = "PROMOTED"


# ============================================================================
# Core Models
# ============================================================================

class ToolSpec(BaseModel):
    """Complete specification for a generated tool."""
    tool_name: str
    description: str
    version: str = "1.0.0"
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    parameters: List[Dict[str, Any]]
    return_type: str = "Dict[str, Any]"
    when_to_use: str
    what_it_does: str
    returns: str
    prerequisites: str


class ValidationReport(BaseModel):
    """Validation results for generated code."""
    schema_ok: bool
    tests_ok: bool
    sandbox_ok: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        """Check if all validation stages passed."""
        return self.schema_ok and self.tests_ok and self.sandbox_ok


class RunArtifacts(BaseModel):
    """Execution results and metadata."""
    result: Dict[str, Any]
    summary_markdown: Optional[str] = None
    execution_time_ms: float
    error: Optional[str] = None


class UserFeedback(BaseModel):
    """User approval/rejection decision."""
    decision: str  # "APPROVED" | "REJECTED"
    notes: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ToolCandidate(BaseModel):
    """Complete bundle for a tool under development."""
    tool_id: str
    version: str = "1.0.0"
    spec: ToolSpec
    code_path: str
    status: ToolStatus = ToolStatus.DRAFT
    validation_report: Optional[ValidationReport] = None
    run_artifacts: Optional[RunArtifacts] = None
    user_feedback: Optional[UserFeedback] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class RegistryMetadata(BaseModel):
    """Registry catalog of all promoted tools."""
    tools: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# LangGraph State
# ============================================================================

class ToolGeneratorState(TypedDict):
    """State shared across all LangGraph nodes."""
    
    # Input
    user_query: str
    data_path: str
    
    # Intent
    extracted_intent: Optional[Dict]
    has_gap: bool
    
    # Generation
    tool_spec: Optional[ToolSpec]
    generated_code: Optional[str]
    
    # Validation
    validation_result: Optional[ValidationReport]
    repair_attempts: int
    
    # Execution
    execution_output: Optional[RunArtifacts]
    
    # Feedback
    stage1_approved: bool
    stage2_approved: bool
    
    # Final
    promoted_tool: Optional[Dict]
    messages: Annotated[List[tuple], add]
