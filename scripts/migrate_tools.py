#!/usr/bin/env python3
"""Utility script for migrating tools between states."""

import json
import shutil
from pathlib import Path
from typing import Literal


def migrate_tool(
    tool_name: str,
    from_state: Literal["draft", "staged", "active"],
    to_state: Literal["draft", "staged", "active"]
):
    """Move a tool from one state directory to another."""
    from_dir = Path(f"tools/{from_state}")
    to_dir = Path(f"tools/{to_state}")
    
    source = from_dir / f"{tool_name}.py"
    dest = to_dir / f"{tool_name}.py"
    
    if not source.exists():
        print(f"Error: {tool_name}.py not found in {from_state}/")
        return
    
    shutil.copy2(source, dest)
    print(f"Migrated {tool_name} from {from_state}/ to {to_state}/")
    
    # Optionally remove from source
    # source.unlink()


def list_tools(state: Literal["draft", "staged", "active"]):
    """List all tools in a given state."""
    state_dir = Path(f"tools/{state}")
    tools = [f.stem for f in state_dir.glob("*.py")]
    
    print(f"\nTools in {state}/ ({len(tools)}):")
    for tool in tools:
        print(f"  - {tool}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  List tools: python migrate_tools.py list <state>")
        print("  Migrate: python migrate_tools.py migrate <tool_name> <from> <to>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "list":
        state = sys.argv[2] if len(sys.argv) > 2 else "active"
        list_tools(state)
    elif command == "migrate":
        if len(sys.argv) < 5:
            print("Error: migrate requires <tool_name> <from> <to>")
            sys.exit(1)
        migrate_tool(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print(f"Unknown command: {command}")
