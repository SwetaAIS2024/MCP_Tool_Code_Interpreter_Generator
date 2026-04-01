# Deployment Guide — Running with a Remote GPU Model Server

This guide covers two deployment scenarios:

- **Scenario A** — Pipeline and Ollama run on the **same machine** (the GPU machine itself)
- **Scenario B** — Pipeline runs on a **client machine** (Mac, Windows, Linux), Ollama runs on a remote GPU machine

Both scenarios use the same codebase. The only difference is one environment variable.

---

## How the LLM endpoint is resolved

The pipeline always reads `LLM_BASE_URL` to know where Ollama is.
Resolution order (first one that is set wins):

```
1. LLM_BASE_URL in .env           ← client machines set this
2. base_url in config/config.yaml ← defaults to localhost:11434 (GPU machine default)
```

The `.env` file is loaded automatically by `src/llm_client.py` at startup via `python-dotenv`.  
You never need to edit `config/config.yaml` — just set or leave out `LLM_BASE_URL` in `.env`.

---

## Prerequisites — Ollama must be running on the GPU machine

> **This applies to both scenarios.**  
> The pipeline makes HTTP calls to Ollama for every LLM request. If Ollama is not running, every call fails with `Connection error`.

### Scenario A (GPU machine, local access only)

```powershell
# Start Ollama — binds to localhost:11434 by default
ollama serve
```

Ollama on Windows also auto-starts from the system tray after installation.  
Verify it is running:
```powershell
curl http://localhost:11434/api/tags
# Returns JSON with your model list if running correctly
```

### Scenario B (GPU machine, must accept remote connections)

By default Ollama **only listens on `127.0.0.1`** — remote clients will get `Connection refused`.  
You **must** set `OLLAMA_HOST=0.0.0.0` before starting Ollama so it accepts connections from other machines on the network.

```powershell
# Step 1 — Kill ALL running Ollama processes
# Ollama on Windows runs as two processes: "ollama" and "ollama app".
# Stop-Process -Name "ollama" only kills one — use the pipeline below to catch both.
Get-Process | Where-Object { $_.Name -like "*ollama*" } | Stop-Process -Force
Start-Sleep -Seconds 2

# Confirm they are gone (should return nothing):
Get-Process | Where-Object { $_.Name -like "*ollama*" }

# Step 2 — Set OLLAMA_HOST for the current user (no admin rights required)
# "User" scope persists across reboots without needing an admin PowerShell.
[System.Environment]::SetEnvironmentVariable("OLLAMA_HOST", "0.0.0.0", "User")

# Step 3 — Also apply to the current terminal session
$env:OLLAMA_HOST = "0.0.0.0"

# Step 4 — Start Ollama (it now binds to all interfaces)
ollama serve
```

> **Note — "Only one usage of each socket address" error:**  
> Ollama is already running (as system tray `ollama app` + background `ollama`). `Stop-Process -Name "ollama"` only kills one of them. Use the `Get-Process | Where-Object` pipeline above to kill both at once.

> **Note — "Machine" scope vs "User" scope:**  
> The `Machine` scope requires an admin PowerShell (`Requested registry access is not allowed` otherwise). The `User` scope does the same job for a single user and needs no admin rights.

Verify it is now listening on all interfaces:
```powershell
netstat -an | findstr 11434
# Must show:  0.0.0.0:11434
# If it still shows 127.0.0.1:11434 → restart Ollama after setting the env var
```

Open the firewall port so the client machine can reach it (run once):
```powershell
New-NetFirewallRule -DisplayName "Ollama LAN" -Direction Inbound -Protocol TCP -LocalPort 11434 -Action Allow
```

Find the GPU machine's IP to give to clients:
```powershell
ipconfig | findstr "IPv4"
# e.g.  IPv4 Address. . . . . : 10.50.128.61
```

---

## Scenario A — GPU Machine (pipeline + Ollama on the same machine)

### What you need
- A machine with a GPU
- [Ollama](https://ollama.ai/) installed
- Python 3.10+

### Step 1 — Start Ollama

```powershell
ollama serve
```

Ollama on Windows also auto-starts from the system tray. Either way, confirm it is running:
```powershell
curl http://localhost:11434/api/tags
# Returns JSON with your model list
```

### Step 2 — Pull models

```bash
ollama pull deepseek-r1:70b      # reasoning model (intent extraction, spec gen)
ollama pull qwen2.5-coder:32b    # coding model (code generation, repair)
ollama list                      # confirm both appear
```

### Step 3 — Set up the Python environment

```bash
cd MCP_Tool_Code_Interpreter_Generator

python -m venv venv

# Activate:
# Windows:
venv\Scripts\activate
# Mac / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### Step 4 — Create `.env`

```bash
# Mac / Linux:
cp .env.example .env

# Windows:
copy .env.example .env
```

Open `.env`. Leave `LLM_BASE_URL` **commented out** (or set it to `localhost`).  
The default in `config/config.yaml` (`http://localhost:11434/v1`) is used automatically.

```dotenv
# Scenario A — nothing to change here, localhost is the default
# LLM_BASE_URL=http://localhost:11434/v1
```

### Step 5 — Test connectivity

```bash
python scripts/llm_ping.py
```

Expected output:
```
=== LLM connectivity check ===
Both models share the same Ollama endpoint (LLM_BASE_URL).

  [reasoning] (from config/config.yaml) ... PASS  'pong'  (xx.xs)
  [coding]    (from config/config.yaml) ... PASS  'pong'  (xx.xs)

All models OK.
```

### Step 6 — Run the pipeline

```bash
python test.py "Show monthly trend of total injuries."
```

---

## Scenario B — Client Machine (pipeline on Mac/Windows/Linux, Ollama on remote GPU)

### What you need on the client machine
- Python 3.10+
- Network access to the GPU machine on port `11434`
- The data CSV file that the pipeline will analyse

> **GPU machine must be prepared first.**  
> Complete the [Prerequisites — Ollama must be running](#prerequisites--ollama-must-be-running-on-the-gpu-machine) section above before continuing — specifically the Scenario B steps: set `OLLAMA_HOST=0.0.0.0`, open the firewall port, and confirm `netstat` shows `0.0.0.0:11434`.

### Step 1 — Get the code on the client machine

```bash
git clone <your-repo-url>
cd MCP_Tool_Code_Interpreter_Generator
git checkout feature/mac-remote-llm-server
```

Or if already cloned:
```bash
git fetch origin
git checkout feature/mac-remote-llm-server
git pull
```

### Step 2 — Set up the Python environment on the client

```bash
python3 -m venv venv

# Mac / Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

### Step 3 — Create `.env` and set the GPU machine's IP

```bash
cp .env.example .env
```

Open `.env` and uncomment the Scenario B line, replacing `<GPU_MACHINE_IP>` with the actual IP:

```dotenv
# Scenario B — client pointing at remote GPU machine
LLM_BASE_URL=http://10.50.128.61:11434/v1
```

### Step 4 — Test connectivity

```bash
python3 scripts/llm_ping.py
```

Expected output:
```
=== LLM connectivity check ===
Both models share the same Ollama endpoint (LLM_BASE_URL).

  [reasoning] http://10.50.128.61:11434/v1 ... PASS  'pong'  (xx.xs)
  [coding]    http://10.50.128.61:11434/v1 ... PASS  'pong'  (xx.xs)

All models OK.
```

If you get `FAIL  LLM generation failed: Connection error.` see the [Troubleshooting](#troubleshooting) section.

### Step 5 — Run the pipeline

```bash
python3 test.py "Show monthly trend of total injuries."
```

The pipeline sends all LLM calls over the network to the GPU machine.  
No GPU, no Ollama, and no Docker is required on the client machine.

---

## Troubleshooting

### `FAIL  LLM generation failed: Connection error.`

Run this on the client to check raw reachability before blaming the code:

```bash
curl http://<GPU_MACHINE_IP>:11434/api/tags
# Should return JSON with a list of Ollama models
# If it times out or refuses → network / firewall issue on the GPU machine
```

| Symptom | Cause | Fix |
|---|---|---|
| `curl` times out | Firewall blocking port 11434 | Add firewall rule on GPU machine (see above) |
| `curl` refused (connection refused) | Ollama bound to `127.0.0.1` only | Set `OLLAMA_HOST=0.0.0.0` on GPU machine and restart Ollama |
| `FAIL (from config/config.yaml)` | `.env` not created or `LLM_BASE_URL` still commented out | Create `.env` from `.env.example`, set `LLM_BASE_URL` |
| ping passes but `test.py` fails | `python-dotenv` not installed | Run `pip install -r requirements.txt` again |

### VS Code integrated terminal not picking up `.env`

The VS Code terminal may open in a different directory, causing `load_dotenv()` to miss the `.env` file. Fix by exporting the variable in your shell profile so it is always present:

```bash
# Add to ~/.zshrc (Mac) or ~/.bashrc (Linux)
export LLM_BASE_URL=http://<GPU_MACHINE_IP>:11434/v1
source ~/.zshrc
```

Or add it to VS Code's terminal environment settings (`Cmd+,` → search `terminal.integrated.env.osx`):
```json
"terminal.integrated.env.osx": {
    "LLM_BASE_URL": "http://10.50.128.61:11434/v1"
}
```

---

## Quick-reference summary

| | Scenario A (GPU machine) | Scenario B (client machine) |
|---|---|---|
| Ollama installed? | Yes — must be running | No |
| `OLLAMA_HOST` required? | No (localhost is fine) | **Yes — must be `0.0.0.0`** |
| Firewall rule needed? | No | **Yes — open port `11434`** |
| GPU required? | Yes | No |
| Docker required? | No | No |
| `.env` change needed? | None — leave `LLM_BASE_URL` commented out | Set `LLM_BASE_URL=http://<GPU_IP>:11434/v1` |
| Start Ollama | `ollama serve` | Already running on GPU machine |
| Test command | `python scripts/llm_ping.py` | `python3 scripts/llm_ping.py` |
| Run command | `python test.py "<query>"` | `python3 test.py "<query>"` |
