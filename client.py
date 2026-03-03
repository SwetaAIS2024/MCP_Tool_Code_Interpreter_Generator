"""Client script for the Code Generator Pipeline API.

Usage:
    python client.py                          # interactive prompt
    python client.py "your query here"        # single query, auto data path
    python client.py "your query" data.csv    # specify data path explicitly

Install dependency:
    pip install requests
"""

import sys
import time
import json
import requests

# ── Configuration ──────────────────────────────────────────────────────────
SERVER = "http://10.50.128.61:8000"          # change to your GPU machine IP
DEFAULT_DATA_PATH = "reference_files/sample_planner_output/traffic_accidents.csv"
POLL_INTERVAL = 10   # seconds between status checks


# ── Core functions ─────────────────────────────────────────────────────────

def submit_job(query: str, data_path: str) -> str:
    """Submit a pipeline job and return the job_id."""
    resp = requests.post(
        f"{SERVER}/api/generate",
        json={"query": query, "data_path": data_path},
        timeout=30,
    )
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    print(f"\nJob submitted  → {job_id}")
    return job_id


def poll_job(job_id: str) -> dict:
    """Poll until the job finishes; return the final job dict."""
    print("Waiting for result", end="", flush=True)
    while True:
        resp = requests.get(f"{SERVER}/api/jobs/{job_id}", timeout=30)
        resp.raise_for_status()
        job = resp.json()
        status = job["status"]

        if status == "completed":
            print(" done.")
            return job
        elif status == "failed":
            print(" failed.")
            return job
        else:
            print(".", end="", flush=True)
            time.sleep(POLL_INTERVAL)


def print_result(job: dict) -> None:
    """Pretty-print the job result."""
    print("\n" + "=" * 70)
    print(f"Query   : {job.get('query')}")
    print(f"Status  : {job['status']}")

    elapsed = None
    if job.get("started_at") and job.get("completed_at"):
        elapsed = job["completed_at"] - job["started_at"]
        print(f"Duration: {elapsed:.1f}s")

    result = job.get("result") or {}

    if job["status"] == "failed":
        print(f"Error   : {job.get('error')}")
        return

    print(f"Promoted: {result.get('promoted')}")
    print(f"Tool    : {result.get('tool_name')}")
    print(f"Repairs : {result.get('repair_attempts', 0)}")

    exec_info = result.get("execution", {})
    print(f"Success : {exec_info.get('success')}")
    if exec_info.get("error"):
        print(f"Exec err: {exec_info['error']}")

    output = exec_info.get("output")
    if output:
        print("\n--- Output ---")
        print(json.dumps(output, indent=2, default=str))

    errors = result.get("errors") or []
    if errors:
        print("\n--- Pipeline errors ---")
        for e in errors:
            print(f"  • {e}")

    print("=" * 70)


def get_tool_code(tool_name: str) -> str | None:
    """Fetch and return the source code of a promoted tool."""
    resp = requests.get(f"{SERVER}/api/tools/{tool_name}", timeout=30)
    if resp.status_code == 404:
        print(f"Tool '{tool_name}' not found on server.")
        return None
    resp.raise_for_status()
    return resp.json()["code"]


def save_tool(tool_name: str, destination: str = None) -> None:
    """Download a promoted tool's source code and save it locally."""
    code = get_tool_code(tool_name)
    if code is None:
        return
    dest = destination or f"{tool_name}.py"
    with open(dest, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"Tool saved to: {dest}")


def health() -> dict:
    """Check server health."""
    resp = requests.get(f"{SERVER}/api/health", timeout=10)
    resp.raise_for_status()
    return resp.json()


def list_jobs() -> list:
    """List all jobs on the server."""
    resp = requests.get(f"{SERVER}/api/jobs", timeout=10)
    resp.raise_for_status()
    return resp.json()


# ── Main ───────────────────────────────────────────────────────────────────

def run(query: str, data_path: str) -> dict:
    """Submit a job, wait for it, and print results."""
    job_id  = submit_job(query, data_path)
    job     = poll_job(job_id)
    print_result(job)

    # Offer to download the generated tool
    tool_name = (job.get("result") or {}).get("tool_name")
    if tool_name and job["status"] == "completed":
        answer = input(f"\nDownload tool code '{tool_name}.py'? [y/N] ").strip().lower()
        if answer == "y":
            save_tool(tool_name)

    return job


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        # Interactive mode
        print(f"Server: {SERVER}")
        h = health()
        print(f"Health: {h}\n")
        query     = input("Query     : ").strip()
        data_path = input(f"Data path [{DEFAULT_DATA_PATH}]: ").strip() or DEFAULT_DATA_PATH
    elif len(args) == 1:
        query     = args[0]
        data_path = DEFAULT_DATA_PATH
    else:
        query     = args[0]
        data_path = args[1]

    run(query, data_path)
