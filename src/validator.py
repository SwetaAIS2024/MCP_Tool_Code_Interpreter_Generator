"""Validator Module - Validate generated code before execution."""

import ast
import re
from typing import List, Optional, Dict, Any    
from src.models import ToolSpec, ValidationReport, ToolGeneratorState
from src.sandbox import create_sandbox


# ============================================================================
# Validator
# ============================================================================

class Validator:
    """Validate generated code through multiple stages."""
    
    def __init__(self):
        """Initialize validator with sandbox."""
        self.sandbox = create_sandbox()
    
    def validate(self, code: str, spec: ToolSpec, test_data_path: Optional[str] = None) -> ValidationReport:
        """Perform multi-stage validation.
        
        Args:
            code: Generated Python code
            spec: Original ToolSpec
            test_data_path: Optional path to test data
            
        Returns:
            ValidationReport with results
        """
        errors = []
        warnings = []
        
        # Stage 1: Syntax validation
        syntax_ok, syntax_errors = self._validate_syntax(code)
        if not syntax_ok:
            return ValidationReport(
                schema_ok=False,
                tests_ok=False,
                sandbox_ok=False,
                errors=syntax_errors,
                warnings=warnings
            )
        
        # Stage 2: Schema compliance
        schema_ok, schema_errors = self._validate_schema(code, spec)
        errors.extend(schema_errors)
        
        # Stage 3: Sandbox execution
        sandbox_ok = True
        if test_data_path:
            sandbox_ok, sandbox_errors = self._validate_sandbox(code, test_data_path)
            errors.extend(sandbox_errors)
        else:
            warnings.append("Sandbox validation skipped: no test data provided")
        
        # Determine overall test status
        tests_ok = len(errors) == 0
        
        return ValidationReport(
            schema_ok=schema_ok,
            tests_ok=tests_ok,
            sandbox_ok=sandbox_ok,
            errors=errors,
            warnings=warnings
        )
    
    def _validate_syntax(self, code: str) -> tuple[bool, List[str]]:
        """Validate Python syntax.
        
        Args:
            code: Python code
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        try:
            ast.parse(code)
            return True, []
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
            return False, errors
        except Exception as e:
            errors.append(f"Parse error: {str(e)}")
            return False, errors
    
    def _validate_schema(self, code: str, spec: ToolSpec) -> tuple[bool, List[str]]:
        """Validate code matches ToolSpec schema.
        
        Args:
            code: Python code
            spec: ToolSpec
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check function name exists
        if f"def {spec.tool_name}" not in code:
            errors.append(f"Function '{spec.tool_name}' not found in code")
        
        # Check required imports (only essential ones, not typing)
        required_imports = ["pandas"]
        for imp in required_imports:
            if imp not in code:
                errors.append(f"Missing required import: {imp}")
        
        # Check MCP decorator
        if "@mcp.tool()" not in code:
            errors.append("Missing @mcp.tool() decorator")
        
        # Check that function signature exists (don't enforce type annotations)
        function_pattern = rf"def {spec.tool_name}\(.*\):"
        if not re.search(function_pattern, code):
            errors.append(f"Function signature for '{spec.tool_name}' is malformed")
        
        # Check for required parameters
        for param in spec.parameters:
            if param.get("required", True):
                param_name = param["name"]
                if param_name not in code:
                    errors.append(f"Missing required parameter: {param_name}")
        
        return len(errors) == 0, errors
    
    def _validate_sandbox(self, code: str, test_data_path: str) -> tuple[bool, List[str]]:
        """Validate code execution in sandbox.
        
        Args:
            code: Python code
            test_data_path: Path to test data
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        try:
            # Execute in sandbox
            result = self.sandbox.execute(code, test_data_path, timeout=30)
            
            # Check basic execution failure
            if not result["success"]:
                errors.append(f"Sandbox execution failed: {result['stderr']}")
                return False, errors
            
            # Check for import errors in stderr (even if returncode is 0)
            stderr = result.get("stderr", "")
            if stderr:
                import_error_patterns = [
                    "ModuleNotFoundError",
                    "ImportError",
                    "name '.*' is not defined",
                    "No module named"
                ]
                for pattern in import_error_patterns:
                    if re.search(pattern, stderr, re.IGNORECASE):
                        errors.append(f"Import/dependency error detected: {stderr[:300]}")
                        return False, errors
            
            # CRITICAL: Check stdout for runtime errors that were caught by try/except
            # Generated code returns {"result": {}, "metadata": {"error": "..."}} on errors
            stdout = result.get("stdout", "")
            
            # Try to parse stdout as JSON/dict to check for errors in metadata
            if stdout.strip():
                try:
                    # Look for error patterns in stdout
                    error_indicators = [
                        '"error"',
                        "'error'",
                        "Traceback",
                        "Exception:",
                        "Error:",
                        "complex() first argument",
                        "list' object has no attribute",
                        "name '.*' is not defined"
                    ]
                    
                    for indicator in error_indicators:
                        if re.search(indicator, stdout, re.IGNORECASE):
                            errors.append(f"Runtime error detected in output: {stdout[:500]}")
                            return False, errors
                except:
                    pass
            
            # Check execution time
            if result["execution_time_ms"] > 30000:
                errors.append(f"Execution too slow: {result['execution_time_ms']}ms")
            
            # SEMANTIC VALIDATION: Check if execution returned errors about missing columns
            combined_output = (stdout + " " + stderr).lower()
            column_error_patterns = [
                "missing required columns",
                "keyerror:",
                "column.*not found",
                "does not exist",
                "invalid column"
            ]
            
            for pattern in column_error_patterns:
                if re.search(pattern, combined_output, re.IGNORECASE):
                    errors.append(f"Semantic error detected: execution returned column-related error")
                    errors.append(f"Output: {stdout[:200]}")
                    return False, errors
            
            return len(errors) == 0, errors
        
        except Exception as e:
            errors.append(f"Sandbox validation error: {str(e)}")
            return False, errors


# ============================================================================
# Helper Functions
# ============================================================================

def validate(code: str, spec: ToolSpec, test_data_path: Optional[str] = None) -> ValidationReport:
    """Validate generated code.
    
    Args:
        code: Python code to validate
        spec: ToolSpec
        test_data_path: Optional test data path
        
    Returns:
        ValidationReport
    """
    validator = Validator()
    return validator.validate(code, spec, test_data_path)


# ============================================================================
# LangGraph Nodes
# ============================================================================

def validator_node(state: ToolGeneratorState) -> ToolGeneratorState:
    """LangGraph node: Validate generated code.
    
    Args:
        state: Current generator state
        
    Returns:
        Updated state with validation_result
    """
    result = validate(
        state["generated_code"], 
        state["tool_spec"],
        state.get("data_path")
    )
    
    # DEBUG: Print validation results
    print("\n" + "="*80)
    print("VALIDATION RESULTS:")
    print("="*80)
    print(f"Schema OK: {result.schema_ok}")
    print(f"Tests OK: {result.tests_ok}")
    print(f"Sandbox OK: {result.sandbox_ok}")
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for i, err in enumerate(result.errors, 1):
            print(f"  {i}. {err}")
    if result.warnings:
        print(f"\nWARNINGS ({len(result.warnings)}):")
        for i, warn in enumerate(result.warnings, 1):
            print(f"  {i}. {warn}")
    print("="*80 + "\n")
    
    return {
        **state,
        "validation_result": result
    }


def route_after_validation(state: ToolGeneratorState) -> str:
    """Route after validation based on results.
    
    Args:
        state: Current generator state
        
    Returns:
        Next node name
    """
    validation_result = state.get("validation_result")
    if validation_result and validation_result.is_valid:
        return "executor_node"
    elif state.get("repair_attempts", 0) < 3:
        return "repair_node"
    else:
        # Max repair attempts reached, end the flow
        from langgraph.graph import END
        return END
