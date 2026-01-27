"""Code Generator Module - Generate Python code from ToolSpec."""

import black
from pathlib import Path
from typing import List
from src.models import ToolSpec, ValidationReport, ToolGeneratorState
from src.llm_client import QwenLLMClient


# ============================================================================
# Code Generator
# ============================================================================

class CodeGenerator:
    """Generate Python code from ToolSpec."""
    
    def __init__(self, llm_client: QwenLLMClient):
        """Initialize with LLM client.
        
        Args:
            llm_client: Configured Qwen LLM client
        """
        self.llm = llm_client
        self.prompt_template_path = Path("config/prompts/code_generation.txt")
    
    def generate(self, spec: ToolSpec) -> str:
        """Generate complete Python function code.
        
        Args:
            spec: ToolSpec with function specification
            
        Returns:
            Complete formatted Python code with MCP decorator
        """
        # Build prompt for code generation
        prompt = self._build_prompt(spec)
        
        # Generate code with low temperature for consistency
        raw_code = self.llm.generate(prompt, temperature=0.2)
        
        # Wrap with MCP decorator and imports
        full_code = self._wrap_with_mcp(raw_code, spec.tool_name)
        
        # Format with black
        try:
            formatted_code = black.format_str(full_code, mode=black.FileMode())
            return formatted_code
        except Exception as e:
            # If black fails, return unformatted code
            print(f"Warning: Black formatting failed: {e}")
            return full_code
    
    def _build_prompt(self, spec: ToolSpec) -> str:
        """Build prompt for code generation.
        
        Args:
            spec: ToolSpec object
            
        Returns:
            Formatted prompt string
        """
        # Try to load template
        if self.prompt_template_path.exists():
            with open(self.prompt_template_path) as f:
                template = f.read()
            return template.format(
                tool_name=spec.tool_name,
                description=spec.description,
                parameters=spec.parameters,
                implementation_plan=spec.what_it_does,
                what_it_does=spec.what_it_does
            )
        
        # Fallback inline prompt
        return f"""Generate Python function code for this tool specification.

TOOL NAME: {spec.tool_name}
DESCRIPTION: {spec.description}
PARAMETERS: {spec.parameters}
WHAT IT DOES: {spec.what_it_does}
PREREQUISITES: {spec.prerequisites}

Requirements:
1. Function name should be: {spec.tool_name}
2. Use pandas for data manipulation
3. Include type hints for all parameters and return value
4. Add comprehensive error handling (try/except blocks)
5. Return Dict[str, Any] with these keys:
   - 'result': The actual analysis result (dict, list, or dataframe converted to dict)
   - 'metadata': Execution metadata (execution_time, row_count, etc.)
6. Add docstring with:
   - Brief description
   - Args section
   - Returns section
   - Example usage
7. Handle edge cases:
   - Missing columns
   - Empty datasets
   - Invalid data types
8. Add clear variable names and comments for complex operations

Generate ONLY the function definition (def {spec.tool_name}(...):), NOT the imports or decorators.
Do NOT include 'from fastmcp import FastMCP' or '@mcp.tool()' - these will be added separately.

Example structure:
```python
def {spec.tool_name}(file_path: str, **kwargs) -> Dict[str, Any]:
    \"\"\"
    {spec.description}
    
    Args:
        file_path: Path to CSV file
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with 'result' and 'metadata'
    \"\"\"
    import time
    start = time.time()
    
    try:
        # Load data
        df = pd.read_csv(file_path)
        
        # Validate columns
        required_cols = [...]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {{missing}}")
        
        # Perform analysis
        # ... your code here ...
        
        # Return result
        return {{
            "result": result_dict,
            "metadata": {{
                "execution_time_ms": (time.time() - start) * 1000,
                "rows_processed": len(df),
                "columns_used": required_cols
            }}
        }}
    except Exception as e:
        return {{
            "result": {{}},
            "metadata": {{
                "error": str(e),
                "execution_time_ms": (time.time() - start) * 1000
            }}
        }}
```

Generate the complete function now:
"""
    
    def _wrap_with_mcp(self, code: str, tool_name: str) -> str:
        """Wrap generated code with MCP decorator and imports.
        
        Args:
            code: Raw function code
            tool_name: Name of the tool
            
        Returns:
            Complete code with imports and decorators
        """
        # Remove any existing imports or decorators from LLM output
        lines = code.strip().split('\n')
        clean_lines = []
        skip_until_def = True
        
        for line in lines:
            if skip_until_def:
                if line.strip().startswith('def '):
                    skip_until_def = False
                    clean_lines.append(line)
                elif 'import' not in line.lower() and '@' not in line and line.strip():
                    # Keep comments or docstrings before function
                    clean_lines.append(line)
            else:
                clean_lines.append(line)
        
        clean_code = '\n'.join(clean_lines)
        
        # Build complete code with proper structure
        full_code = f'''"""Generated MCP tool: {tool_name}"""

from fastmcp import FastMCP
import pandas as pd
from typing import Dict, Any
import time

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
{clean_code}
'''
        
        return full_code


# ============================================================================
# Code Repair
# ============================================================================

class CodeRepairer:
    """Repair code based on validation errors."""
    
    def __init__(self, llm_client: QwenLLMClient):
        """Initialize with LLM client.
        
        Args:
            llm_client: Configured Qwen LLM client
        """
        self.llm = llm_client
        self.prompt_template_path = Path("config/prompts/code_repair.txt")
    
    def repair(self, code: str, errors: List[str], spec: ToolSpec) -> str:
        """Repair code based on validation errors.
        
        Args:
            code: Original code with errors
            errors: List of error messages
            spec: Original ToolSpec
            
        Returns:
            Repaired code
        """
        # Build repair prompt
        prompt = self._build_repair_prompt(code, errors, spec)
        
        # Generate repaired code
        repaired = self.llm.generate(prompt, temperature=0.1)
        
        # Format with black
        try:
            return black.format_str(repaired, mode=black.FileMode())
        except Exception:
            return repaired
    
    def _build_repair_prompt(self, code: str, errors: List[str], spec: ToolSpec) -> str:
        """Build prompt for code repair.
        
        Args:
            code: Original code
            errors: List of errors
            spec: ToolSpec
            
        Returns:
            Repair prompt
        """
        # Try to load template
        if self.prompt_template_path.exists():
            with open(self.prompt_template_path) as f:
                template = f.read()
            return template.format(
                code=code,
                errors='\n'.join(f"- {error}" for error in errors),
                spec=spec.model_dump_json(indent=2)
            )
        
        # Fallback inline prompt
        errors_str = '\n'.join(f"- {error}" for error in errors)
        
        return f"""Fix the following code based on validation errors.

ORIGINAL CODE:
```python
{code}
```

VALIDATION ERRORS:
{errors_str}

TOOL SPEC:
{spec.model_dump_json(indent=2)}

Instructions:
1. Fix each error listed above
2. Maintain the original functionality
3. Keep the same function signature
4. Ensure code follows best practices
5. Add any missing error handling
6. Verify return type matches spec (Dict[str, Any] with 'result' and 'metadata')
7. Keep all imports and decorators intact

Return the complete corrected Python code.
"""


# ============================================================================
# Helper Functions
# ============================================================================

def generate_code(spec: ToolSpec, llm_client: QwenLLMClient = None) -> str:
    """Generate code from ToolSpec.
    
    Args:
        spec: ToolSpec object
        llm_client: Optional LLM client
        
    Returns:
        Generated Python code
    """
    if llm_client is None:
        from src.llm_client import create_llm_client
        llm_client = create_llm_client()
    
    generator = CodeGenerator(llm_client)
    return generator.generate(spec)


def repair_code(code: str, errors: List[str], spec: ToolSpec = None, 
                llm_client: QwenLLMClient = None) -> str:
    """Repair code based on errors.
    
    Args:
        code: Original code
        errors: List of error messages
        spec: Optional ToolSpec
        llm_client: Optional LLM client
        
    Returns:
        Repaired code
    """
    if llm_client is None:
        from src.llm_client import create_llm_client
        llm_client = create_llm_client()
    
    repairer = CodeRepairer(llm_client)
    
    # If no spec provided, create a minimal one
    if spec is None:
        from src.models import ToolSpec
        spec = ToolSpec(
            tool_name="unknown",
            description="",
            input_schema={},
            output_schema={},
            parameters=[],
            when_to_use="",
            what_it_does="",
            returns="",
            prerequisites=""
        )
    
    return repairer.repair(code, errors, spec)


# ============================================================================
# LangGraph Nodes
# ============================================================================

def code_generator_node(state: ToolGeneratorState) -> ToolGeneratorState:
    """LangGraph node: Generate code from ToolSpec.
    
    Args:
        state: Current generator state
        
    Returns:
        Updated state with generated_code
    """
    from src.llm_client import create_llm_client
    
    llm_client = create_llm_client()
    code = generate_code(state["tool_spec"], llm_client)
    
    return {
        **state,
        "generated_code": code,
        "repair_attempts": 0
    }


def repair_node(state: ToolGeneratorState) -> ToolGeneratorState:
    """LangGraph node: Repair code based on validation errors.
    
    Args:
        state: Current generator state
        
    Returns:
        Updated state with repaired code
    """
    from src.llm_client import create_llm_client
    
    llm_client = create_llm_client()
    errors = state["validation_result"].errors
    repaired = repair_code(state["generated_code"], errors, state["tool_spec"], llm_client)
    
    return {
        **state,
        "generated_code": repaired,
        "repair_attempts": state["repair_attempts"] + 1
    }
