"""Subprocess entrypoint for running the Tool Generator graph.

This module exists to support parent-graph integration without Python import
namespace collisions (both repos use a top-level package named `src`).

Contract
--------
- Input:  JSON file containing an initial ToolGeneratorState dict.
- Output: JSON file containing only parent-safe projected_* fields plus a
          small status envelope.

The parent should run this via `python integration/subprocess_runner.py ...`
with `cwd` set to the child repo root and `PYTHONPATH` including that root.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict


def _read_json(path: Path) -> Dict[str, Any]:
    # Accept BOM-prefixed UTF-8 (common when files are produced by some Windows tools)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _project_only(child_state: Dict[str, Any]) -> Dict[str, Any]:
    """Return only the projected_* keys expected by the parent."""
    keys = [
        "projected_tool_transcript",
        "projected_artifact_log",
        "projected_capability_gap",
        "projected_errors",
        "projected_warnings",
        "projected_final_artifacts",
    ]
    return {k: child_state.get(k) for k in keys}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Tool Generator child graph (subprocess mode)")
    parser.add_argument("--input", required=True, help="Path to input JSON (initial ToolGeneratorState)")
    parser.add_argument("--output", required=True, help="Path to output JSON (projected_* fields)")
    parser.add_argument("--thread-id", default=None, help="Optional thread_id (only used if checkpointer enabled)")
    args = parser.parse_args()

    t0 = time.time()

    input_path = Path(args.input)
    output_path = Path(args.output)

    try:
        # Prefer subprocess sandbox when invoked in integration mode.
        # Docker sandbox is optional and often unavailable on Windows dev machines.
        os.environ.setdefault("TOOLGEN_SANDBOX_MODE", "subprocess")

        # Ensure the repo root (parent of integration/) is importable.
        repo_root = Path(__file__).resolve().parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        initial_state = _read_json(input_path)

        # Import inside try so import failures get returned as structured output.
        from src.pipeline import build_graph  # type: ignore

        graph = build_graph(checkpointer=None, visualize=False)

        config = {}
        if args.thread_id:
            config = {"configurable": {"thread_id": str(args.thread_id)}}

        result_state = graph.invoke(initial_state, config=config)

        projected = _project_only(result_state if isinstance(result_state, dict) else dict(result_state))

        payload: Dict[str, Any] = {
            "ok": True,
            "error": None,
            "meta": {
                "duration_ms": int((time.time() - t0) * 1000),
                "cwd": os.getcwd(),
                "python": sys.executable,
            },
            **projected,
        }
        _write_json(output_path, payload)
        return 0

    except Exception as e:
        payload = {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "meta": {
                "duration_ms": int((time.time() - t0) * 1000),
                "cwd": os.getcwd(),
                "python": sys.executable,
            },
            "projected_tool_transcript": None,
            "projected_artifact_log": None,
            "projected_capability_gap": {
                "component": "tool_generator_subprocess",
                "reason": "child graph invocation failed",
                "exception": f"{type(e).__name__}",
            },
            "projected_errors": [f"child_subprocess_failed: {type(e).__name__}: {e}"],
            "projected_warnings": None,
            "projected_final_artifacts": None,
        }
        try:
            _write_json(output_path, payload)
        except Exception:
            # As an absolute last resort, print to stderr.
            print(json.dumps(payload, default=str), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
