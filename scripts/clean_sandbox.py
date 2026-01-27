#!/usr/bin/env python3
"""Clean up sandbox temporary files."""

import shutil
from pathlib import Path


def clean_sandbox():
    """Remove all temporary files from sandbox workspace."""
    sandbox_dir = Path("tools/sandbox")
    
    # Clean temp_code
    temp_code = sandbox_dir / "temp_code"
    if temp_code.exists():
        for file in temp_code.glob("*.py"):
            file.unlink()
        print(f"Cleaned {temp_code}")
    
    # Clean temp_data
    temp_data = sandbox_dir / "temp_data"
    if temp_data.exists():
        for file in temp_data.glob("*"):
            if file.is_file() and file.name != ".gitkeep":
                file.unlink()
        print(f"Cleaned {temp_data}")
    
    # Clean logs
    logs = sandbox_dir / "logs"
    if logs.exists():
        for file in logs.glob("*.log"):
            file.unlink()
        print(f"Cleaned {logs}")
    
    print("Sandbox cleanup complete!")


if __name__ == "__main__":
    clean_sandbox()
