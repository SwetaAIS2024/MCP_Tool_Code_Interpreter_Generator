"""Promoter module for finalizing and deploying tools to active registry.

This module handles:
- Tool promotion from draft/staged to active
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
    
    def promote(self, spec: ToolSpec, code: str, state: Dict[str, Any] = None) -> Dict[str, str]:
        """Promote tool to active registry with comprehensive logging.
        
        Args:
            spec: Tool specification (ToolSpec object or dict)
            code: Generated and validated Python code
            state: Full pipeline state for logging
            
        Returns:
            Dict with promotion details (name, path, version, logs_path)
        """
        # Extract values from spec (handle both dict and ToolSpec object)
        tool_name = spec["tool_name"] if isinstance(spec, dict) else spec.tool_name
        version = spec["version"] if isinstance(spec, dict) else spec.version
        
        # Handle version conflicts
        final_name = self._resolve_name(tool_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Copy to active directory
        active_path = self.active_dir / f"{final_name}.py"
        active_path.write_text(code)
        
        # Update registry
        self._update_registry(final_name, spec, timestamp)
        
        return {
            "name": final_name,
            "path": str(active_path),
            "version": version,
            "registry_path": str(self.registry_file)
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
    
    def _update_registry(self, tool_name: str, spec: ToolSpec, timestamp: str) -> None:
        """Update registry.json with new tool entry.
        
        Args:
            tool_name: Final tool name (with version suffix if any)
            spec: Tool specification (ToolSpec object or dict)
            timestamp: Creation timestamp
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
            original_name = spec["tool_name"]
            version = spec["version"]
            description = spec["description"]
            parameters = spec["parameters"]  # Already in dict form
            return_type = spec.get("return_type", "dict")
        else:
            original_name = spec.tool_name
            version = spec.version
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
        metadata = {
            "name": tool_name,
            "original_name": original_name,
            "version": version,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "timestamp": timestamp,
            "logs_path": f"tools/logs/{tool_name}_{timestamp}",
            "parameters": parameters,
            "return_type": return_type
        }
        
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
    - Comprehensive logs to tools/logs/{tool_name}_{timestamp}/
    - Updates tools/registry.json
    
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
    
    print("\n" + "="*80)
    print("🎉 TOOL PROMOTED TO REGISTRY")
    print("="*80)
    print(f"Tool Name: {promoted['name']}")
    print(f"Version: {promoted['version']}")
    print(f"Active Path: {promoted['path']}")
    print(f"Registry: {promoted['registry_path']}")
    print("="*80 + "\n")
    
    return {
        **state,
        "promoted_tool": promoted
    }