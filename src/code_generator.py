"""Code Generator Module - Generate Python code from ToolSpec."""

import black
from pathlib import Path
from typing import List
from src.models import ToolSpec, ValidationReport, ToolGeneratorState
from src.llm_client import QwenLLMClient
from src.logger_config import get_logger, log_section

logger = get_logger(__name__)


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
        
        # Debug: Log prompt
        log_section(logger, "CODE GENERATION PROMPT")
        logger.debug(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)
        
        # Generate code with low temperature for consistency
        raw_code = self.llm.generate(prompt, temperature=0.2)
        
        # Debug: Log raw response
        log_section(logger, "RAW LLM RESPONSE")
        logger.debug(raw_code[:1000] + "..." if len(raw_code) > 1000 else raw_code)
        
        # Extract code from markdown blocks or conversational text
        code = self._extract_code(raw_code)
        
        # Debug: Log extracted code
        log_section(logger, "EXTRACTED CODE (before wrapping)")
        logger.debug(code[:1000] + "..." if len(code) > 1000 else code)
        
        # Wrap with MCP decorator and imports
        full_code = self._wrap_with_mcp(code, spec.tool_name, spec.parameters)
        
        # Debug: Log wrapped code
        log_section(logger, "WRAPPED CODE (before black formatting)")
        logger.debug(full_code[:1000] + "..." if len(full_code) > 1000 else full_code)
        
        # Format with black
        try:
            formatted_code = black.format_str(full_code, mode=black.FileMode())
            
            # Debug: Log final formatted code
            log_section(logger, "FINAL FORMATTED CODE")
            logger.debug(formatted_code)
            
            return formatted_code
        except Exception as e:
            # If black fails, return unformatted code
            logger.warning(f"Black formatting failed: {e}")
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
        """Extract ONLY the actual function definition, skipping mocks/placeholders but preserving custom imports.
        
        Args:
            code: Code that may contain mocks, imports, placeholders
            
        Returns:
            Custom imports + function definition
        """
        lines = code.split('\n')
        custom_imports = []
        function_lines = []
        in_function = False
        function_indent = 0
        skip_mock = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Collect custom imports (scipy, statsmodels, matplotlib, numpy, etc.) before function
            if not in_function and (stripped.startswith('import ') or stripped.startswith('from ')):
                # Preserve imports that are NOT standard ones we add ourselves
                is_standard = (
                    'fastmcp' in stripped.lower() or
                    stripped == 'import pandas as pd' or
                    stripped == 'import time'
                )
                if not is_standard:
                    custom_imports.append(line)
                continue
            
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
            result = []
            if custom_imports:
                result.extend(custom_imports)
                result.append('')  # Blank line after imports
            result.extend(function_lines)
            return '\n'.join(result)
        
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
        
        # Extract imports and clean code
        lines = code.strip().split('\n')
        clean_lines = []
        custom_imports = []
        skip_until_def = True
        
        for line in lines:
            stripped = line.strip()
            
            if skip_until_def:
                # Preserve custom imports (scipy, statsmodels, numpy, etc.)
                if stripped.startswith('import ') or stripped.startswith('from '):
                    # Check if it's NOT a standard import we'll add ourselves
                    # Must match the EXACT imports we're adding, not just contain the string
                    is_standard = (
                        stripped == 'from fastmcp import FastMCP' or
                        stripped == 'import pandas as pd' or
                        stripped == 'import time' or
                        'fastmcp' in stripped.lower() and 'FastMCP' in stripped
                    )
                    if not is_standard:
                        custom_imports.append(line)
                    continue
                
                # Skip decorators, mcp definitions, and comments until we reach the function
                if stripped.startswith('def '):
                    skip_until_def = False
                    clean_lines.append(line)
                elif stripped.startswith('@') or 'mcp' in stripped.lower() or 'FastMCP' in stripped:
                    continue
                elif stripped and not stripped.startswith('#'):
                    # Keep docstrings and other content
                    if '"""' in stripped or "'''" in stripped:
                        clean_lines.append(line)
            else:
                clean_lines.append(line)
        
        clean_code = '\n'.join(clean_lines)
        
        # Build import section
        import_section = '''from fastmcp import FastMCP
import pandas as pd
import time'''
        
        if custom_imports:
            import_section += '\n' + '\n'.join(custom_imports)
        
        # Build complete code with proper structure
        full_code = f'''"""Generated MCP tool: {tool_name}"""

{import_section}

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
        llm_client = create_llm_client(model_type="coding")
    
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
        llm_client = create_llm_client(model_type="coding")
    
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
    from pathlib import Path
    from datetime import datetime
    
    # Use coding model for code generation
    llm_client = create_llm_client(model_type="coding")
    code = generate_code(state["tool_spec"], llm_client)
    
    # Save to draft folder immediately (all generated tools go here)
    draft_dir = Path("tools/draft")
    draft_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename with timestamp
    tool_name = state["tool_spec"].tool_name if hasattr(state["tool_spec"], 'tool_name') else state["tool_spec"]["tool_name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    draft_path = draft_dir / f"{tool_name}_{timestamp}.py"
    draft_path.write_text(code)
    
    from src.logger_config import get_logger
    logger = get_logger(__name__)
    logger.info(f"📝 Generated code saved to draft: {draft_path}")
    
    return {
        **state,
        "generated_code": code,
        "repair_attempts": 0,
        "draft_path": str(draft_path)  # Store path for later use
    }


def repair_node(state: ToolGeneratorState) -> ToolGeneratorState:
    """LangGraph node: Repair code based on validation or execution errors.
    
    Args:
        state: Current generator state
        
    Returns:
        Updated state with repaired code
    """
    from src.llm_client import create_llm_client
    
    # Collect errors from validation and execution
    errors = []
    
    # Validation errors
    if state.get("validation_result") and state["validation_result"].errors:
        errors.extend(state["validation_result"].errors)
    
    # Execution errors
    if state.get("execution_output"):
        exec_output = state["execution_output"]
        # Check for error in top-level error field
        if exec_output.get("error"):
            exec_error = f"Execution Error: {exec_output['error']}"
            errors.append(exec_error)
            print(f"\n🔧 Repairing execution error: {exec_output['error']}\n")
        # Also check for error in result metadata
        elif exec_output.get("result") and isinstance(exec_output.get("result"), dict):
            metadata = exec_output["result"].get("metadata", {})
            if "error" in metadata:
                exec_error = f"Execution Error: {metadata['error']}"
                errors.append(exec_error)
                print(f"\n🔧 Repairing execution error: {metadata['error']}\n")
    
    if not errors:
        print("\n⚠️  No errors found to repair")
        return state
    
    # DEBUG: Print repair info
    print("\n" + "="*80)
    print(f"REPAIR ATTEMPT #{state.get('repair_attempts', 0) + 1}")
    print("="*80)
    print(f"Errors to fix: {len(errors)}")
    for i, err in enumerate(errors, 1):
        print(f"  {i}. {err}")
    print("="*80 + "\n")
    
    # Use coding model for code repair
    llm_client = create_llm_client(model_type="coding")
    repaired = repair_code(state["generated_code"], errors, state["tool_spec"], llm_client)
    
    # Update the draft file with repaired code
    if state.get("draft_path"):
        from pathlib import Path
        draft_path = Path(state["draft_path"])
        draft_path.write_text(repaired)
    
    return {
        **state,
        "generated_code": repaired,
        "repair_attempts": state.get("repair_attempts", 0) + 1
    }
