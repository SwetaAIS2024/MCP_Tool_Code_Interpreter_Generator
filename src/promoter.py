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
            spec: Tool specification
            code: Generated and validated Python code
            state: Full pipeline state for logging
            
        Returns:
            Dict with promotion details (name, path, version, logs_path)
        """
        # Handle version conflicts
        final_name = self._resolve_name(spec.tool_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Copy to active directory
        active_path = self.active_dir / f"{final_name}.py"
        active_path.write_text(code)
        
        # Create tool-specific logs directory
        logs_dir = self.tools_dir / "logs" / f"{final_name}_{timestamp}"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comprehensive logs
        self._save_comprehensive_logs(logs_dir, spec, code, state)
        
        # Update registry
        self._update_registry(final_name, spec, timestamp)
        
        return {
            "name": final_name,
            "path": str(active_path),
            "version": spec.version,
            "logs_path": str(logs_dir),
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
    
    def _save_comprehensive_logs(self, logs_dir: Path, spec: ToolSpec, code: str, state: Dict[str, Any]) -> None:
        """Save comprehensive logs for the tool generation process.
        
        Args:
            logs_dir: Directory to save logs
            spec: Tool specification
            code: Generated code
            state: Full pipeline state
        """
        # Save generated code
        (logs_dir / "generated_code.py").write_text(code)
        
        # Save tool spec
        spec_data = {
            "tool_name": spec.tool_name,
            "description": spec.description,
            "version": spec.version,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.param_type,
                    "description": p.description,
                    "required": p.required
                }
                for p in spec.parameters
            ],
            "prerequisites": spec.prerequisites,
            "return_type": spec.return_type,
            "implementation_plan": spec.implementation_plan
        }
        (logs_dir / "tool_spec.json").write_text(json.dumps(spec_data, indent=2))
        
        # Save extracted intent
        if state and "extracted_intent" in state:
            (logs_dir / "extracted_intent.json").write_text(
                json.dumps(state["extracted_intent"], indent=2)
            )
        
        # Save validation results
        if state and "validation_result" in state:
            val = state["validation_result"]
            validation_data = {
                "schema_ok": val.schema_ok,
                "tests_ok": val.tests_ok,
                "sandbox_ok": val.sandbox_ok,
                "errors": val.errors
            }
            (logs_dir / "validation_results.json").write_text(
                json.dumps(validation_data, indent=2)
            )
        
        # Save execution output
        if state and "execution_output" in state:
            exec_out = state["execution_output"]
            execution_data = {
                "result": exec_out.result if hasattr(exec_out, 'result') else str(exec_out),
                "execution_time_ms": exec_out.execution_time_ms if hasattr(exec_out, 'execution_time_ms') else None,
                "error": exec_out.error if hasattr(exec_out, 'error') else None,
                "summary": exec_out.summary_markdown if hasattr(exec_out, 'summary_markdown') else None
            }
            (logs_dir / "execution_output.json").write_text(
                json.dumps(execution_data, indent=2, default=str)
            )
        
        # Save complete pipeline state
        state_snapshot = {
            k: str(v) if not isinstance(v, (dict, list, str, int, float, bool, type(None))) else v
            for k, v in (state or {}).items()
            if k not in ["tool_spec", "validation_result", "execution_output"]  # Already saved separately
        }
        (logs_dir / "pipeline_state.json").write_text(
            json.dumps(state_snapshot, indent=2, default=str)
        )
        
        # Create README with summary
        readme = f"""# Tool Generation Log: {spec.tool_name}

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Files in this directory:

- `generated_code.py` - Final Python code
- `tool_spec.json` - Tool specification
- `extracted_intent.json` - Intent extracted from user query
- `validation_results.json` - Validation results (schema, tests, sandbox)
- `execution_output.json` - Execution results on test data
- `pipeline_state.json` - Complete pipeline state snapshot

## Tool Description:

{spec.description}

## Parameters:

{chr(10).join(f"- {p.name} ({p.param_type}): {p.description}" for p in spec.parameters)}
"""
        (logs_dir / "README.md").write_text(readme)
    
    def _update_registry(self, tool_name: str, spec: ToolSpec, timestamp: str) -> None:
        """Update registry.json with new tool entry.
        
        Args:
            tool_name: Final tool name (with version suffix if any)
            spec: Tool specification
            timestamp: Creation timestamp
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
            "timestamp": timestamp,
            "logs_path": f"tools/logs/{tool_name}_{timestamp}",
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
    print(f"Logs Path: {promoted['logs_path']}")
    print(f"Registry: {promoted['registry_path']}")
    print("="*80 + "\n")
    
    return {
        **state,
        "promoted_tool": promoted
    }