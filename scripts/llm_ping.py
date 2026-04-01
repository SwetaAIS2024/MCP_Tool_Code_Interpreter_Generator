"""
Quick connectivity test for the configured LLM endpoints.

Usage (from repo root):
    python scripts/llm_ping.py

Set LLM_REASONING_BASE_URL / LLM_CODING_BASE_URL (or LLM_BASE_URL) in .env or
the environment before running to test a remote llama-server.

Exit codes:
    0 — both models responded correctly
    1 — one or more models failed / timed out
"""

import os
import sys
import time

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Load .env if python-dotenv is available (soft dependency)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.llm_client import create_llm_client  # noqa: E402

PROMPT = "Reply with exactly one word: pong"
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def ping_model(model_type: str) -> bool:
    """Return True if the model responds sensibly."""
    base_url = os.getenv("LLM_BASE_URL") or "(from config/config.yaml)"
    print(f"  [{model_type}] {base_url} ...", end=" ", flush=True)
    try:
        client = create_llm_client(model_type=model_type)
        start = time.time()
        reply = client.generate(PROMPT, temperature=0.0)
        elapsed = time.time() - start
        reply_short = reply.strip().splitlines()[0][:80]
        ok = "pong" in reply.lower()
        status = PASS if ok else FAIL
        print(f"{status}  {reply_short!r}  ({elapsed:.1f}s)")
        return ok
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL}  {exc}")
        return False


def main() -> int:
    print("=== LLM connectivity check ===")
    print("Both models share the same Ollama endpoint (LLM_BASE_URL).\n")
    results = [ping_model(m) for m in ("reasoning", "coding")]
    print()
    if all(results):
        print("All models OK.")
        return 0
    else:
        failed = [m for m, ok in zip(("reasoning", "coding"), results) if not ok]
        print(f"Failed: {', '.join(failed)}")
        url = os.getenv("LLM_BASE_URL") or "(from config/config.yaml)"
        print(f"Endpoint used: {url}")
        print("Check LLM_BASE_URL in .env — it should point to your GPU machine's Ollama.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
