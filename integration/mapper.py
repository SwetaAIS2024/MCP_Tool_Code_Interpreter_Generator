"""Integration mapper: ToolGeneratorState <-> AnalysisPipelineState.

Two public functions form the complete integration contract:

    build_child_input(parent_state)  ->  dict   (initial ToolGeneratorState)
    apply_child_output(child_result, parent_state)  ->  None  (mutates parent in-place)

These are the ONLY entry points the parent-graph owner needs.
The parent schema (AnalysisPipelineState) is never modified.

Constraints this module respects
---------------------------------
- Parent has extra='forbid': no new top-level keys may be added to parent state.
- Child projected_* fields carry pre-packaged, parent-safe representations of
  every child output (populated by projection_node, the terminal child node).
- Parent list channels (tool_transcript, artifact_log, errors, warnings) use
  capped/deduped reducers; this module extends them, never replaces them.
- Parent final_artifacts is Dict[str, Any] with _replace_latest_dict reducer;
  this module merges into it with dict.update() semantics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Parent field names (string constants to avoid typos)
# ---------------------------------------------------------------------------
_P_INSTRUCTION       = "instruction"
_P_DATASET_PATH      = "dataset_path"
_P_TOOL_TRANSCRIPT   = "tool_transcript"
_P_ARTIFACT_LOG      = "artifact_log"
_P_CAPABILITY_GAP    = "capability_gap"
_P_ERRORS            = "errors"
_P_WARNINGS          = "warnings"
_P_FINAL_ARTIFACTS   = "final_artifacts"

# Child projected field names
_C_TRANSCRIPT        = "projected_tool_transcript"
_C_ARTIFACT_LOG      = "projected_artifact_log"
_C_CAPABILITY_GAP    = "projected_capability_gap"
_C_ERRORS            = "projected_errors"
_C_WARNINGS          = "projected_warnings"
_C_FINAL_ARTIFACTS   = "projected_final_artifacts"


# ---------------------------------------------------------------------------
# 1. Parent -> Child: build initial ToolGeneratorState
# ---------------------------------------------------------------------------

def build_child_input(parent_state: Any) -> Dict[str, Any]:
    """Build a valid initial ToolGeneratorState dict from a parent state.

    Accepts either an AnalysisPipelineState Pydantic instance or a plain dict
    carrying the same field names.

    Mapping
    -------
    ToolGeneratorState.user_query  <- parent instruction
    ToolGeneratorState.data_path   <- parent dataset_path

    All other child fields are initialised to safe defaults so the child graph
    can start cleanly regardless of what the parent has or has not set.

    Args:
        parent_state: AnalysisPipelineState instance or equivalent dict.

    Returns:
        A fully-populated initial ToolGeneratorState dict ready for
        child_graph.invoke() or child_graph.stream().

    Raises:
        ValueError: If instruction or dataset_path cannot be resolved from
                    the parent state.

    Example
    -------
    >>> from integration import build_child_input
    >>> child_init = build_child_input(parent_state)
    >>> child_result = child_graph.invoke(child_init, config)
    """
    # Support both Pydantic model instances and plain dicts
    def _get(key: str, default: Any = None) -> Any:
        if isinstance(parent_state, dict):
            return parent_state.get(key, default)
        return getattr(parent_state, key, default)

    user_query: Optional[str] = _get(_P_INSTRUCTION, None) or ""
    data_path: Optional[str] = _get(_P_DATASET_PATH, None) or ""

    if not user_query:
        raise ValueError(
            "build_child_input: parent state has no 'instruction' field or it is empty. "
            "Provide a non-empty instruction before invoking the child graph."
        )
    if not data_path:
        raise ValueError(
            "build_child_input: parent state has no 'dataset_path' field or it is empty. "
            "Provide a dataset_path before invoking the child graph."
        )

    return {
        # --- Mapped from parent ---
        "user_query":   user_query,
        "data_path":    data_path,

        # --- Child-internal defaults (all nodes initialise from these) ---
        "extracted_intent":   None,
        "has_gap":            False,
        "matched_tool":       None,
        "tool_spec":          None,
        "generated_code":     None,
        "draft_path":         None,
        "validation_result":  None,
        "repair_attempts":    0,
        "execution_output":   None,
        "draft_output_path":  None,
        "promoted_tool":      None,
        "errors":             None,
        "task_id":            None,

        # --- Projection output fields (populated by projection_node) ---
        "projected_tool_transcript":  None,
        "projected_artifact_log":     None,
        "projected_capability_gap":   None,
        "projected_errors":           None,
        "projected_warnings":         None,
        "projected_final_artifacts":  None,

        # --- Messages (BaseMessage list, matches parent type exactly) ---
        "messages": [],
    }


# ---------------------------------------------------------------------------
# 2. Child -> Parent: write projected outputs into parent state
# ---------------------------------------------------------------------------

def apply_child_output(
    child_result: Dict[str, Any],
    parent_state: Any,
    *,
    overwrite_capability_gap: bool = True,
) -> None:
    """Apply child graph outputs to the parent state in-place.

    Reads the six projected_* fields from the completed child state and
    merges them into the corresponding parent AnalysisPipelineState channels.
    The parent's extra='forbid' constraint is never violated because only
    already-declared parent fields are touched.

    Channel mapping
    ---------------
    projected_tool_transcript  ->  parent.tool_transcript   (list extend)
    projected_artifact_log     ->  parent.artifact_log      (list extend)
    projected_capability_gap   ->  parent.capability_gap    (replace)
    projected_errors           ->  parent.errors            (list extend)
    projected_warnings         ->  parent.warnings          (list extend)
    projected_final_artifacts  ->  parent.final_artifacts   (dict.update)

    Args:
        child_result:
            The dict returned by child_graph.invoke() / last event from
            child_graph.stream().
        parent_state:
            AnalysisPipelineState Pydantic instance to be updated in-place.
            Also accepts a plain dict for testing purposes.
        overwrite_capability_gap:
            If True (default), always overwrite parent.capability_gap with
            the child value (including None, to clear a stale gap).
            If False, only write when the child produced a non-None value.

    Raises:
        TypeError: If parent_state is neither a Pydantic model nor a dict.

    Example
    -------
    >>> from integration import build_child_input, apply_child_output
    >>> child_init   = build_child_input(parent_state)
    >>> child_result = child_graph.invoke(child_init, config)
    >>> apply_child_output(child_result, parent_state)
    """
    is_dict_parent = isinstance(parent_state, dict)

    def _get_parent(key: str, default: Any = None) -> Any:
        if is_dict_parent:
            return parent_state.get(key, default)
        return getattr(parent_state, key, default)

    def _set_parent(key: str, value: Any) -> None:
        if is_dict_parent:
            parent_state[key] = value
        else:
            setattr(parent_state, key, value)

    # ---- tool_transcript ----------------------------------------------------
    new_events: Optional[List[Dict[str, Any]]] = child_result.get(_C_TRANSCRIPT)
    if new_events:
        existing: List[Dict[str, Any]] = list(_get_parent(_P_TOOL_TRANSCRIPT) or [])
        existing.extend(new_events)
        _set_parent(_P_TOOL_TRANSCRIPT, existing)

    # ---- artifact_log -------------------------------------------------------
    new_paths: Optional[List[str]] = child_result.get(_C_ARTIFACT_LOG)
    if new_paths:
        existing_paths: List[str] = list(_get_parent(_P_ARTIFACT_LOG) or [])
        # Deduplicate while preserving order (mirrors parent's ARTIFACT_LOG_REDUCER)
        seen = set(existing_paths)
        for p in new_paths:
            if p and p not in seen:
                seen.add(p)
                existing_paths.append(p)
        _set_parent(_P_ARTIFACT_LOG, existing_paths)

    # ---- capability_gap -----------------------------------------------------
    gap_value: Optional[Dict[str, Any]] = child_result.get(_C_CAPABILITY_GAP)
    if overwrite_capability_gap or gap_value is not None:
        _set_parent(_P_CAPABILITY_GAP, gap_value)

    # ---- errors -------------------------------------------------------------
    new_errors: Optional[List[str]] = child_result.get(_C_ERRORS)
    if new_errors:
        existing_errors: List[str] = list(_get_parent(_P_ERRORS) or [])
        existing_errors.extend(new_errors)
        _set_parent(_P_ERRORS, existing_errors)

    # ---- warnings -----------------------------------------------------------
    new_warnings: Optional[List[str]] = child_result.get(_C_WARNINGS)
    if new_warnings:
        existing_warnings: List[str] = list(_get_parent(_P_WARNINGS) or [])
        existing_warnings.extend(new_warnings)
        _set_parent(_P_WARNINGS, existing_warnings)

    # ---- final_artifacts ----------------------------------------------------
    new_artifacts: Optional[Dict[str, Any]] = child_result.get(_C_FINAL_ARTIFACTS)
    if new_artifacts:
        existing_artifacts: Dict[str, Any] = dict(_get_parent(_P_FINAL_ARTIFACTS) or {})
        existing_artifacts.update(new_artifacts)
        _set_parent(_P_FINAL_ARTIFACTS, existing_artifacts)
