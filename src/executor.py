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
    """Recursively convert numpy/pandas types to Python native types for serialization.
    
    Args:
        obj: Object that may contain numpy or pandas types
        
    Returns:
        Object with non-serializable types converted to Python native types
    """
    import numpy as np
    try:
        import pandas as pd
        _has_pandas = True
    except ImportError:
        _has_pandas = False

    if _has_pandas:
        # pd.Period (e.g. Period('2024-01', 'M')) -> string
        if isinstance(obj, pd.Period):
            return str(obj)
        # pd.Timestamp -> ISO string
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        # pd.NaT -> None
        if obj is pd.NaT:
            return None

    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        # Convert both keys and values - keys might also be numpy/pandas types
        return {
            _convert_numpy_types(key): _convert_numpy_types(value)
            for key, value in obj.items()
        }
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_types(item) for item in obj]
    else:
        return obj


def _find_and_extract_plot_base64(obj):
    """Recursively search any nested dict/list for 'plot_base64' and return its value."""
    if isinstance(obj, dict):
        if "plot_base64" in obj:
            return obj["plot_base64"]
        for v in obj.values():
            result = _find_and_extract_plot_base64(v)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = _find_and_extract_plot_base64(item)
            if result is not None:
                return result
    return None


def _strip_plot_base64_recursive(obj, plot_path):
    """Recursively strip 'plot_base64' from dicts and replace with 'plot_path'."""
    if isinstance(obj, dict):
        cleaned = {}
        had_plot = "plot_base64" in obj
        for k, v in obj.items():
            if k == "plot_base64":
                continue
            cleaned[k] = _strip_plot_base64_recursive(v, plot_path)
        if had_plot and plot_path:
            cleaned["plot_path"] = plot_path
        return cleaned
    elif isinstance(obj, list):
        return [_strip_plot_base64_recursive(item, plot_path) for item in obj]
    return obj


def _save_plot_from_result(result_dict: dict, tool_name: str, timestamp: str) -> str:
    """Extract plot_base64 from execution result (at any nesting depth) and save as PNG.

    Args:
        result_dict: The execution result dictionary
        tool_name: Tool name for the filename
        timestamp: Timestamp string for the filename

    Returns:
        Path to saved plot PNG, or None if no plot found
    """
    import base64

    plot_b64 = _find_and_extract_plot_base64(result_dict)
    if not plot_b64:
        return None

    try:
        plots_dir = Path("output/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)

        plot_filename = f"{tool_name}_{timestamp}_plot.png"
        plot_path = plots_dir / plot_filename

        plot_bytes = base64.b64decode(plot_b64)
        plot_path.write_bytes(plot_bytes)

        return str(plot_path)
    except Exception as e:
        logger.warning(f"⚠️ Failed to save plot: {e}")
        return None


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
    
    # Extract and save plot if present in result — BEFORE writing JSON
    plot_path = _save_plot_from_result(result_dict, tool_name, timestamp)
    if plot_path:
        logger.info(f"📊 Plot saved to: {plot_path}")

    # Strip plot_base64 from result_dict so it never lands in the JSON output.
    # Replace with a lightweight plot_path reference at the same nesting level.
    serializable_result = _strip_plot_base64_recursive(result_dict, plot_path)

    # Add metadata to output
    output_data = {
        "tool_name": f"{tool_name}_{timestamp}",
        "user_query": state.get('user_query', ''),
        "execution_timestamp": datetime.now().isoformat(),
        "data_path": state.get('data_path', ''),
        **serializable_result
    }

    output_path.write_text(json.dumps(output_data, indent=2, default=str))
    logger.info(f"💾 Execution results saved to: {output_path}")

    return {
        **state,
        "execution_output": serializable_result,
        "draft_output_path": str(output_path)
    }


def route_after_execution(state: ToolGeneratorState) -> str:
    """Route after execution based on success/failure."""
    from langgraph.graph import END

    execution_output = state.get("execution_output")
    repair_attempts = state.get("repair_attempts", 0)

    # Read max_repair_attempts from config
    try:
        import yaml
        with open("config/config.yaml") as f:
            _cfg = yaml.safe_load(f)
        max_repair_attempts = _cfg.get("validation", {}).get("max_repair_attempts", 5)
    except Exception:
        max_repair_attempts = 5

    if not execution_output:
        return END

    has_error = False
    error_msg = ""

    # Top-level error field (set by sandbox/subprocess runner)
    if execution_output.get("error"):
        has_error = True
        error_msg = execution_output["error"]
    # Error buried inside result dict
    elif isinstance(execution_output.get("result"), dict):
        result_dict = execution_output["result"]
        # Case 1: result itself is {"error": "..."}  (generated code returned error dict)
        if "error" in result_dict and result_dict["error"]:
            has_error = True
            error_msg = result_dict["error"]
        else:
            # Case 2: error buried in result["metadata"]["error"]
            metadata = result_dict.get("metadata", {})
            if isinstance(metadata, dict) and metadata.get("error"):
                has_error = True
                error_msg = metadata["error"]
        # Empty result dict — not repairable (no message to give LLM), just proceed
        # elif len(result_dict) == 0 → fall through to promoter

    if not has_error:
        # Success — promote the tool
        return "promoter_node"

    # There is a real error with a message — attempt repair if budget allows
    if repair_attempts < max_repair_attempts:
        logger.warning(f"⚠️  Execution error detected (attempt {repair_attempts + 1}/{max_repair_attempts})")
        logger.error(f"Error: {error_msg}")
        logger.info("🔧 Attempting automatic code repair...")
        return "repair_node"

    # Budget exhausted
    logger.error(f"❌ Maximum repair attempts ({max_repair_attempts}) exceeded")
    logger.error(f"Final error: {error_msg}")
    logger.info("⚠️  Tool generation failed - ending pipeline")
    return END
