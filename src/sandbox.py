"""Sandbox Module - Safe execution environment for untrusted code."""

import subprocess
import tempfile
import time
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional


# ============================================================================
# Base Sandbox Interface
# ============================================================================

class BaseSandbox(ABC):
    """Abstract base class for sandbox executors."""
    
    @abstractmethod
    def execute(self, code: str, data_path: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute code in isolated environment.
        
        Args:
            code: Python code to execute
            data_path: Path to test data
            timeout: Execution timeout in seconds
            
        Returns:
            Dictionary with execution results
        """
        pass


# ============================================================================
# Subprocess Sandbox (Default)
# ============================================================================

class SubprocessSandboxExecutor(BaseSandbox):
    """Execute code in subprocess with resource limits."""
    
    def __init__(self, config_path: str = "config/sandbox_policy.yaml"):
        """Initialize sandbox executor.
        
        Args:
            config_path: Path to sandbox policy configuration
        """
        self.workspace = Path("tools/sandbox/temp_code")
        self.data_dir = Path("tools/sandbox/temp_data")
        self.logs_dir = Path("tools/sandbox/logs")
        
        # Create directories if they don't exist
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Load policy
        self.policy = self._load_policy(config_path)
        self.timeout = self.policy.get("limits", {}).get("timeout_seconds", 30)
    
    def execute(self, code: str, data_path: str, timeout: int = None) -> Dict[str, Any]:
        """Execute code in subprocess.
        
        Args:
            code: Python code to execute
            data_path: Path to test data
            timeout: Execution timeout (uses config default if None)
            
        Returns:
            Dictionary with stdout, stderr, returncode, execution_time
        """
        if timeout is None:
            timeout = self.timeout
        
        # Strip MCP-specific code for sandbox testing
        code = self._strip_mcp_code(code)
        
        # Create temporary file for code
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.py', 
            dir=self.workspace, 
            delete=False
        ) as f:
            temp_file = Path(f.name)
            f.write(code)
        
        try:
            # Execute in subprocess
            start = time.time()
            
            # Use sys.executable to ensure same Python as current process
            import sys
            python_executable = sys.executable
            
            result = subprocess.run(
                [python_executable, str(temp_file), data_path],  # Pass data_path as argument
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.workspace
            )
            
            execution_time = (time.time() - start) * 1000
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "execution_time_ms": execution_time
            }
        
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution timed out after {timeout} seconds",
                "returncode": -1,
                "execution_time_ms": timeout * 1000
            }
        
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution error: {str(e)}",
                "returncode": -1,
                "execution_time_ms": 0
            }
        
        finally:
            # Cleanup temporary file
            try:
                temp_file.unlink()
            except Exception:
                pass
    
    def _validate_imports(self, code: str) -> bool:
        """Check if code contains blocked imports and only uses allowed imports.
        
        Args:
            code: Python code
            
        Returns:
            True if code is safe, False if blocked/disallowed imports detected
        """
        blocked = self.policy.get("blocked", {}).get("imports", [])
        allowed = self.policy.get("allowed", {}).get("imports", [])
        
        # Simple check for import statements
        lines = code.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                # Check blocked imports first
                for blocked_module in blocked:
                    if blocked_module in stripped:
                        return False
                
                # Check if import is in allowed list (whitelist enforcement)
                import_allowed = False
                for allowed_module in allowed:
                    # Handle "import pandas", "from pandas import ...", and "from scipy.stats import ..."
                    # Check if the allowed module appears after 'import' or 'from'
                    if (f'import {allowed_module}' in stripped or 
                        f'from {allowed_module}.' in stripped or 
                        f'from {allowed_module} import' in stripped):
                        import_allowed = True
                        break
                
                # If we found an import statement and it's not allowed, reject
                if not import_allowed:
                    return False
        
        return True
    
    def _strip_mcp_code(self, code: str) -> str:
        """Strip MCP-specific imports and decorators for sandbox testing, and add function invocation.
        
        Args:
            code: Full code with MCP decorators
            
        Returns:
            Code with MCP parts removed and test invocation added
        """
        lines = code.split('\n')
        cleaned_lines = []
        function_name = None
        
        for line in lines:
            stripped = line.strip()
            # Skip MCP-specific lines
            if any([
                'from fastmcp import' in line,
                'import fastmcp' in line,
                'mcp = FastMCP' in line,
                stripped == '@mcp.tool()' or stripped.startswith('@mcp.tool('),
                stripped.startswith('"""Generated MCP tool:'),
                stripped == 'import time'  # Added by wrapper but not needed in sandbox
            ]):
                continue
            
            # Extract function name for test invocation
            if stripped.startswith('def ') and '(' in stripped and function_name is None:
                # Extract function name: "def function_name(params):" -> "function_name"
                func_def = stripped.split('def ')[1].split('(')[0].strip()
                function_name = func_def
            
            cleaned_lines.append(line)
        
        # Add test invocation at the end if function was found
        if function_name:
            # Add invocation that calls the function and prints result
            test_code = f'''
# Test invocation
if __name__ == "__main__":
    import sys
    import json
    # Get data path from command line or use default
    data_path = sys.argv[1] if len(sys.argv) > 1 else "test_data.csv"
    
    # Call the function
    result = {function_name}(data_path)
    
    # Print result for validation
    print("SANDBOX_RESULT:", json.dumps(result, default=str))
'''
            cleaned_lines.append(test_code)
        
        return '\n'.join(cleaned_lines)
    
    def _load_policy(self, config_path: str) -> Dict[str, Any]:
        """Load sandbox policy from YAML.
        
        Args:
            config_path: Path to policy file
            
        Returns:
            Policy dictionary
        """
        policy_file = Path(config_path)
        if not policy_file.exists():
            # Return default policy
            return {
                "limits": {"timeout_seconds": 30},
                "blocked": {
                    "imports": ["os", "subprocess", "socket", "requests", "urllib"]
                }
            }
        
        try:
            with open(policy_file) as f:
                return yaml.safe_load(f)
        except Exception:
            return {"limits": {"timeout_seconds": 30}, "blocked": {"imports": []}}


# ============================================================================
# Docker Sandbox (Production)
# ============================================================================

class DockerSandboxExecutor(BaseSandbox):
    """Execute code in Docker container (production-grade isolation)."""
    
    def __init__(self, config_path: str = "config/sandbox_policy.yaml"):
        """Initialize Docker sandbox.
        
        Args:
            config_path: Path to sandbox policy
        """
        self.workspace = Path("tools/sandbox")
        self.policy = self._load_policy(config_path)
        self.timeout = self.policy.get("limits", {}).get("timeout_seconds", 30)
    
    def execute(self, code: str, data_path: str, timeout: int = None) -> Dict[str, Any]:
        """Execute code in Docker container.
        
        Args:
            code: Python code to execute
            data_path: Path to test data
            timeout: Execution timeout
            
        Returns:
            Execution results dictionary
        """
        if timeout is None:
            timeout = self.timeout
        
        # Strip MCP-specific code for sandbox testing
        code = self._strip_mcp_code(code)
        
        # Write code to temp file
        code_file = self.workspace / "temp_code" / "tool.py"
        code_file.parent.mkdir(parents=True, exist_ok=True)
        code_file.write_text(code)
        
        # Copy data file to sandbox directory for mounting
        data_file = Path(data_path)
        if not data_file.exists():
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Data file not found: {data_path}",
                "returncode": -1,
                "execution_time_ms": 0
            }
        
        sandbox_data_file = self.workspace / "temp_data" / data_file.name
        sandbox_data_file.parent.mkdir(parents=True, exist_ok=True)
        
        import shutil
        shutil.copy2(data_file, sandbox_data_file)
        
        try:
            # Run docker-compose with data path
            start = time.time()
            
            # Use container-relative path for data
            container_data_path = f"/sandbox/temp_data/{data_file.name}"
            
            result = subprocess.run(
                [
                    'docker-compose', '-f', 'docker/docker-compose.sandbox.yml',
                    'run', '--rm',
                    'sandbox',
                    'python', '/sandbox/temp_code/tool.py', container_data_path
                ],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=timeout,
                cwd=Path.cwd()
            )
            
            execution_time = (time.time() - start) * 1000
            
            # Cleanup
            subprocess.run(
                ['docker-compose', '-f', 'docker/docker-compose.sandbox.yml', 'down'],
                capture_output=True,
                encoding='utf-8',
                errors='replace'
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "execution_time_ms": execution_time
            }
        
        except subprocess.TimeoutExpired:
            # Force cleanup on timeout
            subprocess.run(
                ['docker-compose', '-f', 'docker/docker-compose.sandbox.yml', 'down'],
                capture_output=True,
                encoding='utf-8',
                errors='replace'
            )
            
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Docker execution timed out after {timeout} seconds",
                "returncode": -1,
                "execution_time_ms": timeout * 1000
            }
        
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Docker execution error: {str(e)}",
                "returncode": -1,
                "execution_time_ms": 0
            }
        
        finally:
            # Cleanup temp files
            try:
                code_file.unlink()
            except Exception:
                pass
            try:
                sandbox_data_file.unlink()
            except Exception:
                pass
    
    def _strip_mcp_code(self, code: str) -> str:
        """Strip MCP-specific imports and decorators for sandbox testing.
        
        Args:
            code: Full code with MCP decorators
            
        Returns:
            Code with MCP parts removed and test invocation added
        """
        # Reuse the implementation from SubprocessSandboxExecutor
        lines = code.split('\n')
        cleaned_lines = []
        function_name = None
        
        for line in lines:
            stripped = line.strip()
            if any([
                'from fastmcp import' in line,
                'import fastmcp' in line,
                'mcp = FastMCP' in line,
                stripped == '@mcp.tool()' or stripped.startswith('@mcp.tool('),
                stripped.startswith('"""Generated MCP tool:'),
                stripped == 'import time'
            ]):
                continue
            
            if stripped.startswith('def ') and '(' in stripped and function_name is None:
                func_def = stripped.split('def ')[1].split('(')[0].strip()
                function_name = func_def
            
            cleaned_lines.append(line)
        
        if function_name:
            test_code = f'''
# Test invocation
if __name__ == "__main__":
    import sys
    import json
    data_path = sys.argv[1] if len(sys.argv) > 1 else "test_data.csv"
    result = {function_name}(data_path)
    print("SANDBOX_RESULT:", json.dumps(result, default=str))
'''
            cleaned_lines.append(test_code)
        
        return '\n'.join(cleaned_lines)
    
    def _load_policy(self, config_path: str) -> Dict[str, Any]:
        """Load sandbox policy."""
        policy_file = Path(config_path)
        if policy_file.exists():
            with open(policy_file) as f:
                return yaml.safe_load(f)
        return {"limits": {"timeout_seconds": 30}}


# ============================================================================
# Sandbox Factory
# ============================================================================

class SandboxFactory:
    """Factory for creating sandbox executors."""
    
    @staticmethod
    def create(config_path: str = "config/config.yaml") -> BaseSandbox:
        """Create sandbox executor based on configuration.
        
        Args:
            config_path: Path to main config file
            
        Returns:
            Configured sandbox executor
        """
        config_file = Path(config_path)
        
        if config_file.exists():
            with open(config_file) as f:
                config = yaml.safe_load(f)
            
            mode = config.get("sandbox", {}).get("mode", "subprocess")
        else:
            mode = "subprocess"
        
        if mode == "docker":
            return DockerSandboxExecutor()
        else:
            return SubprocessSandboxExecutor()


# ============================================================================
# Convenience Function
# ============================================================================

def create_sandbox(config_path: str = "config/config.yaml") -> BaseSandbox:
    """Create sandbox executor.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured sandbox instance
    """
    return SandboxFactory.create(config_path)
