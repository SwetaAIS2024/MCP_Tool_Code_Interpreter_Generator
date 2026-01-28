"""Spec Generator Module - Generate formal ToolSpec from extracted intent."""

from pathlib import Path
from typing import Dict, Any
from src.models import ToolSpec, ToolGeneratorState
from src.llm_client import QwenLLMClient


# ============================================================================
# Spec Generator
# ============================================================================

class SpecGenerator:
    """Generate formal ToolSpec from extracted intent."""
    
    def __init__(self, llm_client: QwenLLMClient):
        """Initialize with LLM client.
        
        Args:
            llm_client: Configured Qwen LLM client
        """
        self.llm = llm_client
        self.prompt_template_path = Path("config/prompts/spec_generation.txt")
    
    def generate(self, intent: Dict[str, Any]) -> ToolSpec:
        """Generate ToolSpec from intent.
        
        Args:
            intent: Extracted intent dictionary
            
        Returns:
            Complete ToolSpec object
        """
        # Build prompt for spec generation
        prompt = self._build_prompt(intent)
        
        # Define expected schema for ToolSpec
        toolspec_schema = {
            "type": "object",
            "properties": {
                "tool_name": {"type": "string"},
                "description": {"type": "string"},
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
                "parameters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string"},
                            "description": {"type": "string"},
                            "required": {"type": "boolean"}
                        }
                    }
                },
                "return_type": {"type": "string"},
                "when_to_use": {"type": "string"},
                "what_it_does": {"type": "string"},
                "returns": {"type": "string"},
                "prerequisites": {"type": "string"}
            },
            "required": ["tool_name", "description", "input_schema", "output_schema"]
        }
        
        # Generate spec with LLM
        spec_dict = self.llm.generate_structured(prompt, toolspec_schema)
        
        # Ensure defaults for optional fields
        spec_dict.setdefault("return_type", "Dict[str, Any]")
        spec_dict.setdefault("when_to_use", "When performing data analysis operations")
        spec_dict.setdefault("what_it_does", f"Performs {intent.get('operation', 'data analysis')}")
        spec_dict.setdefault("returns", "Dictionary with 'result' and 'metadata' keys")
        spec_dict.setdefault("prerequisites", "CSV file with required columns")
        
        # Handle fields that might be lists instead of strings
        for field in ["when_to_use", "what_it_does", "returns", "prerequisites"]:
            if isinstance(spec_dict.get(field), list):
                spec_dict[field] = " ".join(spec_dict[field])
        
        # Validate and return ToolSpec
        return ToolSpec(**spec_dict)
    
    def _build_prompt(self, intent: Dict[str, Any]) -> str:
        """Build prompt for spec generation.
        
        Args:
            intent: Extracted intent dictionary
            
        Returns:
            Formatted prompt string
        """
        # Try to load template
        if self.prompt_template_path.exists():
            with open(self.prompt_template_path) as f:
                template = f.read()
            return template.format(
                operation=intent.get("operation", "unknown"),
                columns=intent.get("columns", []),
                metrics=intent.get("metrics", []),
                filters=intent.get("filters", []),
                implementation_plan=intent.get("implementation_plan", [])
            )
        
        # Fallback inline prompt
        operation = intent.get("operation", "unknown")
        columns = intent.get("columns", [])
        metrics = intent.get("metrics", [])
        filters = intent.get("filters", [])
        impl_plan = intent.get("implementation_plan", [])
        
        return f"""Generate a complete ToolSpec for the following data analysis operation.

OPERATION: {operation}
COLUMNS: {columns}
METRICS: {metrics}
FILTERS: {filters}
IMPLEMENTATION PLAN: {impl_plan}

Create a ToolSpec with:
1. tool_name: A descriptive snake_case name based on the operation
2. description: Clear explanation of what this tool does
3. input_schema: JSON Schema for input parameters (must include 'file_path')
4. output_schema: JSON Schema for output structure
5. parameters: List of parameter definitions with name, type, description, required
6. return_type: "Dict[str, Any]"
7. when_to_use: When this tool should be selected
8. what_it_does: Technical description of the operation
9. returns: Description of return value structure
10. prerequisites: Any requirements before using this tool

Return JSON matching ToolSpec structure. Ensure tool_name is snake_case and descriptive.

Example:
{{
  "tool_name": "analyze_accidents_by_state",
  "description": "Analyze traffic accidents grouped by state with count statistics",
  "input_schema": {{
    "type": "object",
    "properties": {{
      "file_path": {{"type": "string", "description": "Path to CSV file"}},
      "limit": {{"type": "integer", "description": "Max results", "default": 10}}
    }},
    "required": ["file_path"]
  }},
  "output_schema": {{
    "type": "object",
    "properties": {{
      "result": {{"type": "object"}},
      "metadata": {{"type": "object"}}
    }}
  }},
  "parameters": [
    {{"name": "file_path", "type": "str", "description": "Path to CSV file", "required": true}},
    {{"name": "limit", "type": "int", "description": "Maximum number of results", "required": false}}
  ],
  "return_type": "Dict[str, Any]",
  "when_to_use": "When analyzing accident patterns by state",
  "what_it_does": "Groups accidents by state and counts occurrences",
  "returns": "Dictionary with 'result' containing grouped data and 'metadata' with execution info",
  "prerequisites": "CSV file with 'state' column"
}}
"""


# ============================================================================
# Helper Functions
# ============================================================================

def generate_spec(intent: Dict[str, Any], llm_client: QwenLLMClient = None) -> ToolSpec:
    """Generate ToolSpec from intent.
    
    Args:
        intent: Extracted intent dictionary
        llm_client: Optional LLM client
        
    Returns:
        ToolSpec object
    """
    if llm_client is None:
        from src.llm_client import create_llm_client
        llm_client = create_llm_client()
    
    generator = SpecGenerator(llm_client)
    return generator.generate(intent)


# ============================================================================
# LangGraph Node
# ============================================================================

def spec_generator_node(state: ToolGeneratorState) -> ToolGeneratorState:
    """LangGraph node: Generate ToolSpec from extracted intent.
    
    Args:
        state: Current generator state
        
    Returns:
        Updated state with tool_spec
    """
    from src.llm_client import create_llm_client
    
    llm_client = create_llm_client()
    spec = generate_spec(state["extracted_intent"], llm_client)
    
    return {
        **state,
        "tool_spec": spec
    }
