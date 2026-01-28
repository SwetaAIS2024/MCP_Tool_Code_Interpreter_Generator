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
        
        # Validate imports before execution
        if not self._validate_imports(code):
            return {
                "success": False,
                "stdout": "",
                "stderr": "Blocked imports detected in code",
                "returncode": -1,
                "execution_time_ms": 0
            }
        
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
            
            result = subprocess.run(
                ['python', str(temp_file)],
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
        """Check if code contains blocked imports.
        
        Args:
            code: Python code
            
        Returns:
            True if code is safe, False if blocked imports detected
        """
        blocked = self.policy.get("blocked", {}).get("imports", [])
        
        # Simple check for import statements
        lines = code.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                for blocked_module in blocked:
                    if blocked_module in stripped:
                        return False
        
        return True
    
    def _strip_mcp_code(self, code: str) -> str:
        """Strip MCP-specific imports and decorators for sandbox testing.
        
        Args:
            code: Full code with MCP decorators
            
        Returns:
            Code with MCP parts removed
        """
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            # Skip MCP-specific lines
            if any([
                'from fastmcp import' in line,
                'import fastmcp' in line,
                'mcp = FastMCP' in line,
                stripped == '@mcp.tool()' or stripped.startswith('@mcp.tool('),
                stripped.startswith('"""Generated MCP tool:')
            ]):
                continue
            cleaned_lines.append(line)
        
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
        
        # Write code to temp file
        code_file = self.workspace / "temp_code" / "tool.py"
        code_file.write_text(code)
        
        try:
            # Run docker-compose
            start = time.time()
            
            result = subprocess.run(
                ['docker-compose', '-f', 'docker/docker-compose.sandbox.yml', 'up', '--abort-on-container-exit'],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Path.cwd()
            )
            
            execution_time = (time.time() - start) * 1000
            
            # Cleanup
            subprocess.run(
                ['docker-compose', '-f', 'docker/docker-compose.sandbox.yml', 'down'],
                capture_output=True
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
                capture_output=True
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
            # Cleanup temp file
            try:
                code_file.unlink()
            except Exception:
                pass
    
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
