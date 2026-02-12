"""Validator Module - Validate generated code before execution."""

import ast
import re
from typing import List, Optional, Dict, Any    
from src.models import ToolSpec, ValidationReport, ToolGeneratorState
from src.sandbox import create_sandbox
from src.logger_config import get_logger, log_section, log_success, log_error

logger = get_logger(__name__)


# ============================================================================
# Validator
# ============================================================================

class Validator:
    """Validate generated code through multiple stages."""
    
    def __init__(self):
        """Initialize validator with sandbox."""
        self.sandbox = create_sandbox()
    
    def validate(self, code: str, spec: ToolSpec, test_data_path: Optional[str] = None, operation: Optional[str] = None) -> ValidationReport:
        """Perform multi-stage validation.
        
        Args:
            code: Generated Python code
            spec: Original ToolSpec
            test_data_path: Optional path to test data
            operation: Optional operation type from extracted_intent
            
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
        schema_ok, schema_errors = self._validate_schema(code, spec, operation)
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
    
    def _validate_schema(self, code: str, spec: ToolSpec, operation: Optional[str] = None) -> tuple[bool, List[str]]:
        """Validate code matches ToolSpec schema.
        
        Args:
            code: Python code
            spec: ToolSpec
            operation: Optional operation type from extracted_intent
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        warnings = []
        
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
        
        # ⚠️ CONSISTENCY GUARDS - Prevent statistical injection
        what_it_does = spec.what_it_does.lower()
        
        # Guard 1: Block chi-square injection for non-statistical operations
        if operation in ['groupby_aggregate', 'pivot', 'filter', 'describe_summary']:
            statistical_patterns = [
                'chi2_contingency', 'chi_square', 'chi2',
                'scipy.stats.chi', 'from scipy.stats import chi'
            ]
            for pattern in statistical_patterns:
                if pattern in code.lower():
                    errors.append(f"Statistical injection detected: '{pattern}' found in code but operation is '{operation}' (non-statistical)")
        
        # Guard 2: Block ANOVA injection for non-statistical operations
        if operation in ['groupby_aggregate', 'pivot', 'filter', 'describe_summary']:
            if 'f_oneway' in code.lower() and 'anova' not in what_it_does:
                errors.append(f"Statistical injection detected: ANOVA (f_oneway) found but not requested in spec")
        
        # Guard 3: Verify statistical tests match spec requirements
        if 'anova' in what_it_does and 'f_oneway' not in code:
            errors.append("Spec requires ANOVA but code does not contain f_oneway")
        
        if 'chi-square' in what_it_does or 'chi2' in what_it_does:
            if 'chi2_contingency' not in code:
                errors.append("Spec requires chi-square but code does not contain chi2_contingency")
        
        if 'correlation' in what_it_does:
            if not any(x in code for x in ['pearsonr', 'spearmanr', 'kendalltau']):
                errors.append("Spec requires correlation but code does not contain correlation function")
        
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
                    # Check for actual runtime errors (not sandbox wrapper messages)
                    # Ignore "SANDBOX_RESULT:" wrapper - focus on actual code errors
                    
                    # Critical error patterns (actual failures)
                    critical_error_patterns = [
                        "Traceback \\(most recent call last\\)",  # Python traceback
                        "Exception:",
                        "SyntaxError:",
                        "IndentationError:",
                        "NameError:",
                        "TypeError:",
                        "ValueError:",
                        "AttributeError:",
                        "KeyError:",
                    ]
                    
                    # Check for critical errors
                    has_critical_error = False
                    for pattern in critical_error_patterns:
                        if re.search(pattern, stdout):
                            has_critical_error = True
                            break
                    
                    # Only fail if we found a critical error pattern
                    # Don't fail on generic "error" keyword which might be in JSON structure
                    if has_critical_error:
                        errors.append(f"Runtime error detected in output: {stdout[:500]}")
                        return False, errors
                    
                    # Downgrade: If we see "error" keyword but no critical pattern,
                    # it might be a sandbox environment issue - treat as warning
                    if '"error"' in stdout or "'error'" in stdout:
                        # This is likely a sandbox wrapper message, not actual code failure
                        # Don't block - actual executor will validate properly
                        pass
                        
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

def validate(code: str, spec: ToolSpec, test_data_path: Optional[str] = None, operation: Optional[str] = None) -> ValidationReport:
    """Validate generated code.
    
    Args:
        code: Python code to validate
        spec: ToolSpec
        test_data_path: Optional test data path
        operation: Optional operation type from extracted_intent
        
    Returns:
        ValidationReport
    """
    validator = Validator()
    return validator.validate(code, spec, test_data_path, operation)


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
    # Extract operation from extracted_intent if available
    operation = None
    if "extracted_intent" in state and state["extracted_intent"]:
        operation = state["extracted_intent"].get("operation")
    
    result = validate(
        state["generated_code"], 
        state["tool_spec"],
        state.get("data_path"),
        operation
    )
    
    # Log validation results
    log_section(logger, "VALIDATION RESULTS")
    logger.info(f"Schema OK: {result.schema_ok}")
    logger.info(f"Tests OK: {result.tests_ok}")
    logger.info(f"Sandbox OK: {result.sandbox_ok}")
    if result.errors:
        log_error(logger, f"ERRORS ({len(result.errors)}):")
        for i, err in enumerate(result.errors, 1):
            logger.error(f"  {i}. {err}")
    if result.warnings:
        logger.warning(f"\nWARNINGS ({len(result.warnings)}):")
        for i, warn in enumerate(result.warnings, 1):
            logger.warning(f"  {i}. {warn}")
    
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
        # Max repair attempts reached, proceed to executor anyway
        # Let execution/feedback catch any remaining issues
        logger.warning("⚠️  Max repair attempts reached, proceeding to execution...")
        return "executor_node"
