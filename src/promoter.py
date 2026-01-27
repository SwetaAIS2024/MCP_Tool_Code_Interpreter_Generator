"""Promoter module for finalizing and deploying tools to active registry.

This module handles:
- Tool promotion from draft/staged to active
- Version conflict resolution
- Registry updates
- File management
"""

from pathlib import Path
from typing import Dict
import json
from datetime import datetime

from .models import ToolSpec, RegistryMetadata, ToolGeneratorState


class ToolPromoter:
    """Promotes validated tools to active registry."""
    
    def __init__(self, tools_dir: Path):
        """Initialize promoter.
        
        Args:
            tools_dir: Base tools directory containing active/, draft/, staged/
        """
        self.tools_dir = Path(tools_dir)
        self.active_dir = self.tools_dir / "active"
        self.registry_file = self.tools_dir / "registry.json"
        
        # Ensure directories exist
        self.active_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize registry if it doesn't exist
        if not self.registry_file.exists():
            self.registry_file.write_text(json.dumps({"tools": []}, indent=2))
    
    def promote(self, spec: ToolSpec, code: str) -> Dict[str, str]:
        """Promote tool to active registry.
        
        Args:
            spec: Tool specification
            code: Generated and validated Python code
            
        Returns:
            Dict with promotion details (name, path, version)
        """
        # Handle version conflicts
        final_name = self._resolve_name(spec.tool_name)
        
        # Copy to active directory
        active_path = self.active_dir / f"{final_name}.py"
        active_path.write_text(code)
        
        # Update registry
        self._update_registry(final_name, spec)
        
        return {
            "name": final_name,
            "path": str(active_path),
            "version": spec.version
        }
    
    def _resolve_name(self, name: str) -> str:
        """Resolve tool name, handling version conflicts.
        
        If tool with same name exists, appends _v2, _v3, etc.
        
        Args:
            name: Desired tool name
            
        Returns:
            Final unique tool name
        """
        if not (self.active_dir / f"{name}.py").exists():
            return name
        
        # Add version suffix
        version = 2
        while (self.active_dir / f"{name}_v{version}.py").exists():
            version += 1
        return f"{name}_v{version}"
    
    def _update_registry(self, tool_name: str, spec: ToolSpec) -> None:
        """Update registry.json with new tool entry.
        
        Args:
            tool_name: Final tool name (with version suffix if any)
            spec: Tool specification
        """
        # Load existing registry
        registry_data = json.loads(self.registry_file.read_text())
        
        # Create metadata entry
        metadata = {
            "name": tool_name,
            "original_name": spec.tool_name,
            "version": spec.version,
            "description": spec.description,
            "created_at": datetime.now().isoformat(),
            "parameters": [
                {
                    "name": param.name,
                    "type": param.param_type,
                    "description": param.description,
                    "required": param.required
                }
                for param in spec.parameters
            ],
            "return_type": spec.return_type
        }
        
        # Add to registry
        registry_data["tools"].append(metadata)
        
        # Save updated registry
        self.registry_file.write_text(json.dumps(registry_data, indent=2))


def promoter_node(state: ToolGeneratorState) -> ToolGeneratorState:
    """LangGraph node: Promote tool to active registry.
    
    This is the final step in the pipeline. The tool has passed all validation
    and approval stages and is now ready for production use.
    
    Args:
        state: Current graph state with tool_spec and generated_code
        
    Returns:
        Updated state with promoted_tool containing deployment details
    """
    tools_dir = Path("tools")
    promoter = ToolPromoter(tools_dir)
    
    promoted = promoter.promote(
        state["tool_spec"],
        state["generated_code"]
    )
    
    return {
        **state,
        "promoted_tool": promoted
    }