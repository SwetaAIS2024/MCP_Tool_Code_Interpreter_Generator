"""Executor Module - Execute validated tools and capture results."""

import importlib.util
import time
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, Callable
from concurrent.futures import TimeoutError, ThreadPoolExecutor
from datetime import datetime
from src.models import RunArtifacts, ToolGeneratorState
from src.logger_config import get_logger, log_section, log_success, log_error

logger = get_logger(__name__)


# ============================================================================
# Tool Executor
# ============================================================================

class ToolExecutor:
    """Execute validated tools and capture execution artifacts."""
    
    def __init__(self, timeout: int = 300):
        """Initialize executor.
        
        Args:
            timeout: Maximum execution time in seconds
        """
        self.timeout = timeout
    
    def execute(self, code: str, data_path: str) -> RunArtifacts:
        """Execute tool code and capture results.
        
        Args:
            code: Python code (as string or path to file)
            data_path: Path to input data
            
        Returns:
            RunArtifacts with results
        """
        start = time.time()
        
        try:
            # Write code to draft folder (where all generated tools are stored)
            draft_dir = Path("tools/draft")
            draft_dir.mkdir(parents=True, exist_ok=True)
            
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False,
                dir=str(draft_dir)
            ) as f:
                code_path = Path(f.name)
                f.write(code)
            
            # Load function from file
            func = self._load_function(code_path)
            
            # Execute with timeout
            result = self._execute_with_timeout(
                func, 
                {"file_path": data_path}, 
                timeout=self.timeout
            )
            
            logger.debug(f"Result type: {type(result)}")
            logger.debug(f"Result value: {result}")
            
            # Ensure result is a dictionary (wrap if necessary)
            if not isinstance(result, dict):
                wrapped_result = {
                    "result": result,
                    "metadata": {
                        "note": "Result was not returned as dict, auto-wrapped"
                    }
                }
                logger.debug("Auto-wrapped non-dict result into dict")
            else:
                wrapped_result = result
            
            # Extract summary if available
            summary = wrapped_result.get("summary") or wrapped_result.get("result", {}).get("summary") if isinstance(wrapped_result.get("result"), dict) else None
            
            return RunArtifacts(
                result=wrapped_result,
                summary_markdown=summary,
                execution_time_ms=(time.time() - start) * 1000,
                error=None
            )
        
        except TimeoutError:
            return RunArtifacts(
                result={},
                summary_markdown=None,
                execution_time_ms=(time.time() - start) * 1000,
                error=f"Execution timed out after {self.timeout} seconds"
            )
        
        except Exception as e:
            return RunArtifacts(
                result={},
                summary_markdown=None,
                execution_time_ms=(time.time() - start) * 1000,
                error=f"Execution error: {str(e)}"
            )
        
        finally:
            # Cleanup temp file
            try:
                if 'code_path' in locals():
                    code_path.unlink()
            except Exception:
                pass
    
    def _load_function(self, code_path: Path) -> Callable:
        """Load function from Python file.
        
        Args:
            code_path: Path to Python file
            
        Returns:
            Loaded function
        """
        # Load module from file
        spec = importlib.util.spec_from_file_location("generated_tool", code_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {code_path}")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # DEBUG: Print what's in the module
        all_attrs = [name for name in dir(module) if not name.startswith('_')]
        logger.debug(f"Module contents: {all_attrs}")
        
        # Find the actual function - look for user-defined functions
        import types
        for attr_name in dir(module):
            if not attr_name.startswith('_') and attr_name not in ['mcp', 'FastMCP', 'pd', 'pandas', 'time', 'sns', 'plt', 'seaborn', 'matplotlib', 'chi2_contingency']:
                attr = getattr(module, attr_name)
                logger.debug(f"Checking '{attr_name}': type={type(attr)}, callable={callable(attr)}, is_function={isinstance(attr, types.FunctionType)}")
                
                # Check if it's a FunctionTool from FastMCP (decorated function)
                if type(attr).__name__ == 'FunctionTool':
                    logger.debug("Found FunctionTool, extracting underlying function...")
                    # The FunctionTool has the original function stored
                    # Try common attribute names for the wrapped function
                    for key in ['func', 'fn', '_func', 'function', '_fn', '__wrapped__']:
                        if hasattr(attr, key):
                            func = getattr(attr, key)
                            if callable(func):
                                logger.debug(f"Successfully extracted function from FunctionTool via '{key}'")
                                return func
                    # If we can't extract, try calling the FunctionTool directly
                    # but this might not work as expected
                    logger.warning("Could not extract function from FunctionTool, trying direct call")
                    return attr
                
                # Check if it's a regular function
                if isinstance(attr, types.FunctionType):
                    logger.debug(f"Found regular function: {attr_name}")
                    return attr
                
                # Check if it's any other callable (but not a class)
                if callable(attr) and not isinstance(attr, type):
                    logger.debug(f"Found callable: {attr_name}")
                    return attr
        
        raise ValueError("No executable function found in module")
    
    def _execute_with_timeout(self, func: Callable, kwargs: Dict[str, Any], 
                             timeout: int) -> Dict[str, Any]:
        """Execute function with timeout.
        
        Args:
            func: Function to execute
            kwargs: Function arguments
            timeout: Timeout in seconds
            
        Returns:
            Function result
            
        Raises:
            TimeoutError: If execution exceeds timeout
        """
        from concurrent.futures import TimeoutError as FuturesTimeoutError
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, **kwargs)
            try:
                result = future.result(timeout=timeout)
                return result
            except FuturesTimeoutError as e:
                raise TimeoutError(f"Function execution timed out after {timeout}s") from e


# ============================================================================
# Helper Functions
# ============================================================================

def execute_tool(code: str, data_path: str, timeout: int = 300) -> RunArtifacts:
    """Execute tool and capture artifacts.
    
    Args:
        code: Python code string
        data_path: Path to input data
        timeout: Execution timeout in seconds
        
    Returns:
        RunArtifacts
    """
    executor = ToolExecutor(timeout=timeout)
    return executor.execute(code, data_path)


def _convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for serialization.
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to Python native types
    """
    import numpy as np
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        # Convert both keys and values - keys might also be numpy types
        return {
            _convert_numpy_types(key): _convert_numpy_types(value) 
            for key, value in obj.items()
        }
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_types(item) for item in obj]
    else:
        return obj


# ============================================================================
# LangGraph Node
# ============================================================================

def executor_node(state: ToolGeneratorState) -> ToolGeneratorState:
    """LangGraph node: Execute tool and capture results.
    
    Args:
        state: Current generator state
        
    Returns:
        Updated state with execution_output and draft_output_path
    """
    result = execute_tool(
        state["generated_code"],
        state["data_path"],
        timeout=300
    )
    
    # Convert RunArtifacts to dict for serialization
    # LangGraph's checkpointer uses msgpack which doesn't handle Pydantic models or numpy types
    result_dict = result.model_dump() if hasattr(result, 'model_dump') else result
    
    # Convert numpy types to Python native types
    result_dict = _convert_numpy_types(result_dict)
    
    # Save execution results to output/draft
    output_draft_dir = Path("output/draft")
    output_draft_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract timestamp from draft_path or generate new one
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if state.get('draft_path'):
        draft_filename = Path(state['draft_path']).stem
        # Extract timestamp from filename (last 15 chars: YYYYMMDD_HHMMSS)
        if len(draft_filename) > 15 and draft_filename[-15:-7].isdigit():
            timestamp = draft_filename[-15:]
    
    # Get tool name from spec
    tool_spec = state.get('tool_spec', {})
    tool_name = tool_spec.get('tool_name', 'tool') if isinstance(tool_spec, dict) else getattr(tool_spec, 'tool_name', 'tool')
    
    # Save as JSON with timestamp
    output_filename = f"{tool_name}_{timestamp}_output.json"
    output_path = output_draft_dir / output_filename
    
    # Add metadata to output
    output_data = {
        "tool_name": f"{tool_name}_{timestamp}",
        "user_query": state.get('user_query', ''),
        "execution_timestamp": datetime.now().isoformat(),
        "data_path": state.get('data_path', ''),
        **result_dict
    }
    
    output_path.write_text(json.dumps(output_data, indent=2, default=str))
    logger.info(f"💾 Execution results saved to: {output_path}")
    
    return {
        **state,
        "execution_output": result_dict,
        "draft_output_path": str(output_path)
    }


def route_after_execution(state: ToolGeneratorState) -> str:
    """Route after execution based on success/failure.
    
    Args:
        state: Current generator state
        
    Returns:
        Next node name
    """
    from langgraph.graph import END
    
    execution_output = state.get("execution_output")
    repair_attempts = state.get("repair_attempts", 0)
    max_repair_attempts = 3  # Maximum automatic repair attempts
    
    if not execution_output:
        # No execution output - something went wrong, end
        return END
    
    # execution_output is now a dict (converted from RunArtifacts)
    # Structure: {"result": {...}, "error": "...", "execution_time_ms": ...}
    has_error = False
    error_msg = "Unknown error"
    
    # Check for error field
    if execution_output.get("error"):
        has_error = True
        error_msg = execution_output.get("error")
    # Also check for error in result metadata
    elif execution_output.get("result") and isinstance(execution_output.get("result"), dict):
        metadata = execution_output["result"].get("metadata", {})
        if "error" in metadata:
            has_error = True
            error_msg = metadata.get("error", "Unknown error")
        # Check if result is empty (no actual result returned)
        elif not execution_output["result"].get("result"):
            has_error = True
            error_msg = "Empty result returned"
    
    # If there's an error and we haven't exceeded repair attempts, try repair
    if has_error and repair_attempts < max_repair_attempts:
        logger.warning(f"⚠️  Execution error detected (attempt {repair_attempts + 1}/{max_repair_attempts})")
        logger.error(f"Error: {error_msg}")
        logger.info("🔧 Attempting automatic code repair...")
        
        # Store the error message in a way repair_node can access it
        # Update the state to ensure repair node can see the error
        return "repair_node"
    
    # If repair attempts exceeded, end the pipeline
    if has_error and repair_attempts >= max_repair_attempts:
        logger.error(f"❌ Maximum repair attempts ({max_repair_attempts}) exceeded")
        logger.error(f"Final error: {error_msg}")
        logger.info("⚠️  Tool generation failed - ending pipeline")
        return END
    
    # Always proceed directly to promoter (no human feedback)
    return "promoter_node"
