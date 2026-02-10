"""Promoter module for finalizing and deploying tools to active registry.

This module handles:
- Tool promotion from draft to active (only successful executions)
- Version conflict resolution
- Registry updates
- File management
- Comprehensive logging and archival
"""

from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime
import shutil

from .models import ToolSpec, RegistryMetadata, ToolGeneratorState
from .logger_config import get_logger, log_section, log_success

logger = get_logger(__name__)


class ToolPromoter:
    """Promotes validated tools to active registry."""
    
    def __init__(self, tools_dir: Path):
        """Initialize promoter.
        
        Args:
            tools_dir: Base tools directory containing active/, draft/
        """
        self.tools_dir = Path(tools_dir)
        self.draft_dir = self.tools_dir / "draft"
        self.active_dir = self.tools_dir / "active"
        self.registry_file = self.tools_dir / "registry.json"
        
        # Ensure directories exist
        self.draft_dir.mkdir(parents=True, exist_ok=True)
        self.active_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize registry if it doesn't exist
        if not self.registry_file.exists():
            self.registry_file.write_text(json.dumps({"tools": []}, indent=2))
    
    def promote(self, spec: ToolSpec, code: str, state: Dict[str, Any] = None) -> Dict[str, str]:
        """Promote tool to active registry with comprehensive logging.
        
        Args:
            spec: Tool specification (ToolSpec object or dict)
            code: Generated and validated Python code
            state: Full pipeline state for logging
            
        Returns:
            Dict with promotion details (name, path, registry_path, output_path)
        """
        # Extract values from spec (handle both dict and ToolSpec object)
        tool_name = spec["tool_name"] if isinstance(spec, dict) else spec.tool_name
        
        # Extract timestamp from draft_path if available, otherwise generate new one
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if state and state.get('draft_path'):
            # Extract timestamp from draft filename: tool_name_TIMESTAMP.py
            from pathlib import Path
            draft_filename = Path(state['draft_path']).stem
            # Get the timestamp part (last 15 characters: YYYYMMDD_HHMMSS)
            if len(draft_filename) > 15 and draft_filename[-15:-7].isdigit():
                timestamp = draft_filename[-15:]
        
        # Use same naming convention as draft: tool_name_timestamp.py
        final_name = f"{tool_name}_{timestamp}"
        
        # Copy to active directory
        active_path = self.active_dir / f"{final_name}.py"
        active_path.write_text(code)
        
        # Move output file from draft to active if it exists
        output_active_path = None
        if state and state.get('draft_output_path'):
            draft_output_path = Path(state['draft_output_path'])
            if draft_output_path.exists():
                output_active_dir = Path("output/active")
                output_active_dir.mkdir(parents=True, exist_ok=True)
                output_active_path = output_active_dir / draft_output_path.name
                shutil.copy2(draft_output_path, output_active_path)
                logger.info(f"💾 Moved output file to: {output_active_path}")
        
        # Update registry with tool path and user query
        self._update_registry(final_name, spec, timestamp, str(active_path), str(output_active_path) if output_active_path else None, state)
        
        return {
            "name": final_name,
            "path": str(active_path),
            "registry_path": str(self.registry_file),
            "output_path": str(output_active_path) if output_active_path else None
        }
    
    def _update_registry(self, tool_name: str, spec: ToolSpec, timestamp: str, tool_path: str, output_path: str = None, state: Dict[str, Any] = None) -> None:
        """Update registry.json with new tool entry.
        
        Args:
            tool_name: Final tool name (with timestamp)
            spec: Tool specification (ToolSpec object or dict)
            timestamp: Creation timestamp
            tool_path: Path to the tool file in active directory
            output_path: Path to the execution output JSON file (optional)
            state: Full pipeline state (contains user_query)
        """
        # Load existing registry, handle empty/corrupted file
        try:
            registry_data = json.loads(self.registry_file.read_text())
            if not isinstance(registry_data, dict) or "tools" not in registry_data:
                registry_data = {"tools": []}
        except (json.JSONDecodeError, FileNotFoundError):
            registry_data = {"tools": []}
        
        # Extract values from spec (handle both dict and ToolSpec object)
        if isinstance(spec, dict):
            description = spec["description"]
            parameters = spec["parameters"]  # Already in dict form
            return_type = spec.get("return_type", "dict")
        else:
            description = spec.description
            # Parameters might be dicts even in ToolSpec objects
            params = spec.parameters if hasattr(spec, 'parameters') else []
            if params and isinstance(params[0], dict):
                parameters = params  # Already dicts
            else:
                parameters = [
                    {
                        "name": param.name,
                        "type": param.param_type,
                        "description": param.description,
                        "required": param.required
                    }
                    for param in params
                ]
            return_type = getattr(spec, 'return_type', 'dict')
        
        # Create metadata entry
        # Get actual log file path from logger
        from .logger_config import PipelineLogger
        pipeline_logger = PipelineLogger()
        log_file_path = pipeline_logger.current_log_file if hasattr(pipeline_logger, 'current_log_file') and pipeline_logger.current_log_file else f"logs/pipeline_{timestamp}.log"
        
        # Get user query from state
        user_query = state.get('user_query', '') if state else ''
        
        metadata = {
            "name": tool_name,
            "description": description,
            "user_query": user_query,
            "tool_path": tool_path,
            "log_file": log_file_path,
            "parameters": parameters,
            "return_type": return_type
        }
        
        # Add output_file if available
        if output_path:
            metadata["output_file"] = output_path
        
        # Add to registry
        registry_data["tools"].append(metadata)
        
        # Save updated registry
        self.registry_file.write_text(json.dumps(registry_data, indent=2))


def promoter_node(state: ToolGeneratorState) -> ToolGeneratorState:
    """LangGraph node: Promote tool to active registry.
    
    This is the final step in the pipeline. The tool has passed all validation
    and approval stages and is now ready for production use.
    
    Saves:
    - Tool code to tools/active/
    - Execution output to output/active/
    - Updates tools/registry.json with log file and output file references
    
    Args:
        state: Current graph state with tool_spec and generated_code
        
    Returns:
        Updated state with promoted_tool containing deployment details
    """
    tools_dir = Path("tools")
    promoter = ToolPromoter(tools_dir)
    
    promoted = promoter.promote(
        state["tool_spec"],
        state["generated_code"],
        state=dict(state)  # Pass full state for logging
    )
    
    log_section(logger, "🎉 TOOL PROMOTED TO ACTIVE REGISTRY")
    logger.info(f"Tool Name: {promoted['name']}")
    if state.get('draft_path'):
        logger.info(f"Draft Path: {state['draft_path']}")
    logger.info(f"Active Path: {promoted['path']}")
    if promoted.get('output_path'):
        logger.info(f"Output Path: {promoted['output_path']}")
    logger.info(f"Registry: {promoted['registry_path']}")
    logger.info("✅ Tool successfully executed and promoted to active")
    
    return {
        **state,
        "promoted_tool": promoted
    }