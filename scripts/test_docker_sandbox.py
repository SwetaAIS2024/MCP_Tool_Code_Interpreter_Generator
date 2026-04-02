"""Quick smoke test for DockerSandboxExecutor."""
import sys, importlib.util
sys.path.insert(0, ".")

# Load sandbox directly to avoid src/__init__ triggering langgraph import
spec = importlib.util.spec_from_file_location("sandbox", "src/sandbox.py")
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)
DockerSandboxExecutor = _mod.DockerSandboxExecutor

s = DockerSandboxExecutor()

code = """
import pandas as pd
def test_tool(file_path: str):
    df = pd.read_csv(file_path)
    return {"rows": len(df), "cols": len(df.columns)}
"""

data = r"reference_files\sample_planner_output\traffic_accidents.csv"
result = s.execute(code, data, timeout=60)
print("success    :", result["success"])
print("returncode :", result["returncode"])
print("stdout     :", result["stdout"][:300] if result["stdout"] else "")
print("stderr     :", result["stderr"][:300] if result["stderr"] else "")
