"""Executor Module - Execute validated tools and capture results."""

import importlib.util
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, Callable
from concurrent.futures import TimeoutError, ThreadPoolExecutor
from src.models import RunArtifacts, ToolGeneratorState


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
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False,
                dir="tools/staged"
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
            
            print(f"[EXECUTOR DEBUG] Result type: {type(result)}")
            print(f"[EXECUTOR DEBUG] Result value: {result}")
            
            # Extract summary if available (handle both dict and non-dict results)
            summary = None
            if isinstance(result, dict):
                summary = result.get("summary") or result.get("result", {}).get("summary")
            
            return RunArtifacts(
                result=result,
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
        print(f"\n[EXECUTOR] Module contents: {all_attrs}")
        
        # Find the actual function - look for user-defined functions
        import types
        for attr_name in dir(module):
            if not attr_name.startswith('_') and attr_name not in ['mcp', 'FastMCP', 'pd', 'pandas', 'time']:
                attr = getattr(module, attr_name)
                print(f"[EXECUTOR] Checking '{attr_name}': type={type(attr)}, callable={callable(attr)}, is_function={isinstance(attr, types.FunctionType)}")
                
                # Check if it's a FunctionTool from FastMCP (decorated function)
                if type(attr).__name__ == 'FunctionTool':
                    print(f"[EXECUTOR] Found FunctionTool, extracting underlying function...")
                    # The FunctionTool has the original function stored
                    if hasattr(attr, 'func'):
                        return attr.func
                    elif hasattr(attr, 'fn'):
                        return attr.fn
                    elif hasattr(attr, '_func'):
                        return attr._func
                    # Try to get it from __dict__
                    for key in ['func', 'fn', '_func', 'function']:
                        if hasattr(attr, key):
                            func = getattr(attr, key)
                            if callable(func):
                                return func
                
                # Check if it's callable (decorated functions might not be FunctionType)
                if callable(attr) and not isinstance(attr, type):
                    print(f"[EXECUTOR] Found callable: {attr_name}")
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


# ============================================================================
# LangGraph Node
# ============================================================================

def executor_node(state: ToolGeneratorState) -> ToolGeneratorState:
    """LangGraph node: Execute tool and capture results.
    
    Args:
        state: Current generator state
        
    Returns:
        Updated state with execution_output
    """
    result = execute_tool(
        state["generated_code"],
        state["data_path"],
        timeout=300
    )
    
    return {
        **state,
        "execution_output": result
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
    
    if not execution_output:
        # No execution output - something went wrong, end
        return END
    
    # Always proceed to user feedback regardless of execution success
    # User will review the output and decide whether to approve or reject
    # This includes cases where execution failed or returned empty results
    return "feedback_stage1_node"
