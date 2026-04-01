"""Data Models Module - Core Pydantic models and LangGraph state definitions."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from operator import add
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


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


class A2ATaskStatus(str, Enum):
    """A2A protocol task lifecycle states."""
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"


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


class A2ATask(BaseModel):
    """A2A protocol Task object — one per client request."""
    task_id: str
    status: A2ATaskStatus = A2ATaskStatus.SUBMITTED
    query: str
    data_path: str
    thread_id: Optional[str] = None        # LangGraph thread_id for interrupt/resume
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    input_request: Optional[str] = None   # message sent to client when status=input-required
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


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
    matched_tool: Optional[Dict]  # Best-matching registry entry when has_gap=False
    
    # Generation
    tool_spec: Optional[ToolSpec]
    generated_code: Optional[str]
    draft_path: Optional[str]  # Path to code saved in draft folder
    
    # Validation
    validation_result: Optional[ValidationReport]
    repair_attempts: int
    
    # Execution
    execution_output: Optional[Dict[str, Any]]
    draft_output_path: Optional[str]  # Path to execution output in draft folder
    
    # Final
    promoted_tool: Optional[Dict]

    # Errors accumulated during the run (e.g. from spec_generator_node)
    errors: Optional[List[str]]

    # A2A correlation — set by the A2A server before invoking the graph so
    # that nodes can tag their log output with the originating task ID.
    task_id: Optional[str]

    # Projection outputs — pre-packaged for parent AnalysisPipelineState.
    # Populated by projection_node (terminal node). The parent owner copies
    # these directly into the corresponding parent channels without any
    # field-name mapping, avoiding extra='forbid' violations.
    projected_tool_transcript: Optional[List[Dict[str, Any]]]
    projected_artifact_log: Optional[List[str]]
    projected_capability_gap: Optional[Dict[str, Any]]
    projected_errors: Optional[List[str]]
    projected_warnings: Optional[List[str]]
    projected_final_artifacts: Optional[Dict[str, Any]]

    # messages uses add_messages to match parent AnalysisPipelineState exactly.
    # No existing node writes to this field — migration is zero-risk.
    messages: Annotated[List[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# Public re-exports consumed by the A2A server
# ---------------------------------------------------------------------------
__all__ = [
    "ToolStatus",
    "A2ATaskStatus",
    "A2ATask",
    "ToolSpec",
    "ValidationReport",
    "RunArtifacts",
    "UserFeedback",
    "ToolCandidate",
    "RegistryMetadata",
    "ToolGeneratorState",
]
