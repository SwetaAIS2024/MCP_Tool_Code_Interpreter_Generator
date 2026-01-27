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
            
            # Extract summary if available
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
        
        # Find the decorated function (should have @mcp.tool())
        # Look for functions that are not private/magic
        for attr_name in dir(module):
            if not attr_name.startswith('_'):
                attr = getattr(module, attr_name)
                if callable(attr) and not isinstance(attr, type):
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
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, **kwargs)
            try:
                result = future.result(timeout=timeout)
                return result
            except Exception as e:
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
