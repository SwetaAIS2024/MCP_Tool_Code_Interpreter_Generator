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
        
        # DEBUG: Print prompt to see what's being sent
        print("\n" + "="*80)
        print("CODE GENERATION PROMPT:")
        print("="*80)
        print(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)
        print("="*80 + "\n")
        
        # Generate code with low temperature for consistency
        raw_code = self.llm.generate(prompt, temperature=0.2)
        
        # DEBUG: Print raw LLM response
        print("\n" + "="*80)
        print("RAW LLM RESPONSE:")
        print("="*80)
        print(raw_code[:1000] + "..." if len(raw_code) > 1000 else raw_code)
        print("="*80 + "\n")
        
        # Extract code from markdown blocks or conversational text
        code = self._extract_code(raw_code)
        
        # DEBUG: Print extracted code
        print("\n" + "="*80)
        print("EXTRACTED CODE (before wrapping):")
        print("="*80)
        print(code[:1000] + "..." if len(code) > 1000 else code)
        print("="*80 + "\n")
        
        # Wrap with MCP decorator and imports
        full_code = self._wrap_with_mcp(code, spec.tool_name, spec.parameters)
        
        # DEBUG: Print wrapped code before formatting
        print("\n" + "="*80)
        print("WRAPPED CODE (before black formatting):")
        print("="*80)
        print(full_code[:1000] + "..." if len(full_code) > 1000 else full_code)
        print("="*80 + "\n")
        
        # Format with black
        try:
            formatted_code = black.format_str(full_code, mode=black.FileMode())
            
            # DEBUG: Print final formatted code
            print("\n" + "="*80)
            print("FINAL FORMATTED CODE:")
            print("="*80)
            print(formatted_code)
            print("="*80 + "\n")
            
            return formatted_code
        except Exception as e:
            # If black fails, return unformatted code
            print(f"Warning: Black formatting failed: {e}")
            return full_code
    
    def _extract_code(self, text: str) -> str:
        """Extract code from LLM response, handling markdown blocks and conversational text.
        
        Args:
            text: Raw LLM response
            
        Returns:
            Extracted code
        """
        import re
        
        # Strategy 1: Find last code block (in case of multiple)
        code_blocks = re.findall(r'```(?:python)?\s*(.*?)\s*```', text, re.DOTALL)
        if code_blocks:
            # Use the last code block (usually the corrected one)
            code = code_blocks[-1].strip()
            # Now extract just the function from this block
            return self._extract_function_only(code)
        
        # Strategy 2: Extract from plain text
        return self._extract_function_only(text)
    
    def _extract_function_only(self, code: str) -> str:
        """Extract ONLY the actual function definition, skipping mocks/placeholders.
        
        Args:
            code: Code that may contain mocks, imports, placeholders
            
        Returns:
            Just the function definition
        """
        lines = code.split('\n')
        function_lines = []
        in_function = False
        function_indent = 0
        skip_mock = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Skip mock/placeholder decorator definitions
            if 'def mcp_tool' in stripped or 'def decorator' in stripped or 'class FastMCP' in stripped:
                skip_mock = True
                continue
            
            # Skip mock imports
            if skip_mock and (stripped.startswith('return') or stripped.startswith('func.')):
                continue
            
            # Reset skip_mock when we hit a real function
            if stripped.startswith('def ') and 'mcp_tool' not in stripped and 'decorator' not in stripped:
                skip_mock = False
                in_function = True
                function_indent = len(line) - len(line.lstrip())
                function_lines.append(line)
                continue
            
            # If we're in the function, keep collecting lines
            if in_function:
                current_indent = len(line) - len(line.lstrip())
                # Continue while indented or empty lines
                if line.strip() == '' or current_indent > function_indent:
                    function_lines.append(line)
                else:
                    # Function ended
                    break
        
        if function_lines:
            return '\n'.join(function_lines)
        
        # Fallback
        return code.strip()
    
    def _build_prompt(self, spec: ToolSpec) -> str:
        """Build prompt for code generation.
        
        Args:
            spec: ToolSpec object
            
        Returns:
            Formatted prompt string
        """
        # Try to load template
        if self.prompt_template_path.exists():
            print(f"\n✓ Using template file: {self.prompt_template_path}")
            with open(self.prompt_template_path, encoding='utf-8') as f:
                template = f.read()
            
            # Convert parameters list to readable string
            import json
            params_str = json.dumps(spec.parameters, indent=2) if isinstance(spec.parameters, list) else str(spec.parameters)
            
            return template.format(
                tool_name=spec.tool_name,
                description=spec.description,
                parameters=params_str,
                required_columns=spec.prerequisites,  # Prerequisites contains column requirements
                implementation_plan=spec.what_it_does,
                what_it_does=spec.what_it_does
            )
        
        # Fallback inline prompt
        print(f"\n⚠ Using FALLBACK prompt (template not found at {self.prompt_template_path})")
        return f"""Generate Python function code for this tool specification.

TOOL NAME: {spec.tool_name}
DESCRIPTION: {spec.description}
PARAMETERS: {spec.parameters}
WHAT IT DOES: {spec.what_it_does}
PREREQUISITES: {spec.prerequisites}

Requirements:
1. Function name should be: {spec.tool_name}
2. Use pandas for data manipulation
3. Function signature: def {spec.tool_name}(file_path: str):
4. Add comprehensive error handling (try/except blocks)
5. Return a dictionary with these keys:
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

CRITICAL: Use {{}} or dict() to create dictionaries. Do NOT import typing or use Dict() constructor.

Generate ONLY the function definition (def {spec.tool_name}(...):), NOT the imports or decorators.
Do NOT include 'from fastmcp import FastMCP' or '@mcp.tool()' - these will be added separately.
Do NOT create placeholder decorators or mock classes.

Example structure:
```python
def {spec.tool_name}(file_path: str, **kwargs):
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
    
    def _wrap_with_mcp(self, code: str, tool_name: str, parameters: list = None) -> str:
        """Wrap generated code with MCP decorator and imports.
        
        Args:
            code: Raw function code
            tool_name: Name of the tool
            parameters: List of parameter specs from ToolSpec
            
        Returns:
            Complete code with imports and decorators
        """
        import re
        
        # Fix parameter names to match spec
        if parameters:
            for param in parameters:
                spec_param_name = param.get("name", "")
                # Replace data_path with the spec's parameter name (usually file_path)
                if spec_param_name and spec_param_name != "data_path":
                    code = re.sub(r'\bdata_path\b', spec_param_name, code)
        
        # Remove any existing imports or decorators from LLM output
        lines = code.strip().split('\n')
        clean_lines = []
        skip_until_def = True
        
        for line in lines:
            if skip_until_def:
                if line.strip().startswith('def '):
                    skip_until_def = False
                    # Keep the line as-is, don't add type hints
                    clean_lines.append(line)
                elif 'import' not in line.lower() and '@' not in line and 'class' not in line.lower() and line.strip():
                    # Skip placeholder classes/decorators but keep docstrings
                    if not line.strip().startswith('#'):
                        continue
            else:
                clean_lines.append(line)
        
        clean_code = '\n'.join(clean_lines)
        
        # Build complete code with proper structure
        full_code = f'''"""Generated MCP tool: {tool_name}"""

from fastmcp import FastMCP
import pandas as pd
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
        raw_repaired = self.llm.generate(prompt, temperature=0.1)
        
        # Extract code (repair responses also have conversational text)
        repaired = self._extract_repair_code(raw_repaired)
        
        # If the repaired code doesn't have the decorator, re-wrap it
        if "@mcp.tool()" not in repaired:
            # Extract just the function
            generator = CodeGenerator(self.llm)
            function_code = generator._extract_function_only(repaired)
            # Re-wrap with decorator
            repaired = generator._wrap_with_mcp(function_code, spec.tool_name, spec.parameters)
        
        # Format with black
        try:
            return black.format_str(repaired, mode=black.FileMode())
        except Exception:
            return repaired
    
    def _extract_repair_code(self, text: str) -> str:
        """Extract code from repair response.
        
        Args:
            text: Raw repair response
            
        Returns:
            Extracted code
        """
        import re
        
        # Find last code block (repaired version)
        code_blocks = re.findall(r'```(?:python)?\s*(.*?)\s*```', text, re.DOTALL)
        if code_blocks:
            return code_blocks[-1].strip()
        
        # Look for complete code starting from imports/decorators
        lines = text.split('\n')
        code_lines = []
        started = False
        
        for line in lines:
            stripped = line.strip()
            if not started:
                # Start from imports, decorators, or function definition
                if stripped.startswith(('from ', 'import ', '@', 'def ', '"""Generated')):
                    started = True
                    code_lines.append(line)
            else:
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines)
        
        return text.strip()
    
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
            with open(self.prompt_template_path, encoding='utf-8') as f:
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
6. Verify return value is a dictionary with 'result' and 'metadata' keys
7. Keep all imports and decorators intact
8. CRITICAL: Use {{}} or dict() to create dictionaries, NOT Dict() constructor

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
    
    # DEBUG: Print repair info
    print("\n" + "="*80)
    print(f"REPAIR ATTEMPT #{state['repair_attempts'] + 1}")
    print("="*80)
    print(f"Errors to fix: {len(state['validation_result'].errors)}")
    for i, err in enumerate(state['validation_result'].errors, 1):
        print(f"  {i}. {err}")
    print("="*80 + "\n")
    
    llm_client = create_llm_client()
    errors = state["validation_result"].errors
    repaired = repair_code(state["generated_code"], errors, state["tool_spec"], llm_client)
    
    return {
        **state,
        "generated_code": repaired,
        "repair_attempts": state["repair_attempts"] + 1
    }
