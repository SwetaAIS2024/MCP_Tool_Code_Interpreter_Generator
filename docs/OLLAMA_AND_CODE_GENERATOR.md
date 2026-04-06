# Ollama Model Deployment & Code Generator Pipeline

## 1. How Ollama Is Deployed

**Ollama** is an open-source runtime that serves LLMs locally over an HTTP API compatible with the OpenAI API spec. It is deployed on a **GPU machine** (either the same machine or a remote one).

**Two deployment scenarios** (documented in `config/config.yaml` and `src/llm_client.py`):

| Scenario | Setup | URL |
|---|---|---|
| **Local** (dev on GPU machine) | No env var needed | `http://localhost:11434/v1` |
| **Remote** (client → GPU machine) | Set `LLM_BASE_URL=http://<GPU_IP>:11434/v1` in `.env` | Env var takes priority |

**Two models are running simultaneously in Ollama**, routed by model name:

```yaml
# config/config.yaml
models:
  reasoning: "deepseek-r1:70b"   # intent extraction & planning
  coding:    "qwen2.5-coder:32b" # code generation & repair
```

Key tuning parameters set to override Ollama defaults:

| Parameter | Value | Reason |
|---|---|---|
| `num_ctx` | `16384` | Reduces KV cache from ~40GB to ~3–6GB, cutting latency from 2–3 min → 20–40s |
| `num_predict` | `16384` | Ollama default of 128 tokens is far too small for generated code |
| `read_timeout` | `600s` | Allows up to 10 min for large model load into VRAM on first request |

---

## 2. How the Code Generator Connects to the Model

**Framework stack:**

```
Python App
    └── openai SDK  (OpenAI-compatible client)
            └── httpx  (HTTP transport with custom timeouts)
                    └── Ollama REST API  (/v1/chat/completions)
                            ├── deepseek-r1:70b   (reasoning tasks)
                            └── qwen2.5-coder:32b (coding tasks)
```

The `openai` Python SDK is used **not** against OpenAI's servers, but pointed at Ollama's OpenAI-compatible endpoint. This is the standard pattern for self-hosted LLMs.

In `src/llm_client.py`, the `QwenLLMClient` class initializes this:

```python
self.client = OpenAI(
    base_url=base_url,       # http://localhost:11434/v1 or remote
    api_key="not-needed",    # plain Ollama on LAN requires no auth
    http_client=httpx.Client(timeout=timeout),
    max_retries=0            # retry logic is owned by generate_structured()
)
```

The factory function `create_llm_client(model_type="coding"|"reasoning")` enforces that callers always specify which model they want — no accidental wrong model usage.

### URL Resolution Order

```
LLM_BASE_URL (env var / .env file)
        │ if not set
        ▼
config/config.yaml → llm.base_url
        │ default
        ▼
http://localhost:11434/v1
```

---

## 3. The LangGraph Pipeline — End-to-End Flow

The orchestration framework is **LangGraph** (from LangChain). The pipeline is a compiled `StateGraph` defined in `src/pipeline.py`. Each stage is a node, and conditional edges route between them.

```
User Query
    │
    ▼
[intent_node]  ─────────────── deepseek-r1:70b  (reasoning)
    │  Reads:   dataset schema (CSV preview, dtypes, sample values)
    │  Output:  structured JSON → extracted_intent
    │           (operation, required_columns, implementation_plan, has_gap)
    │
    ▼
[spec_generator_node]  ──────── deepseek-r1:70b  (reasoning)
    │  Input:   extracted_intent
    │  Output:  ToolSpec (tool_name, parameters, return_type, description)
    │
    ▼
[code_generator_node]  ──────── qwen2.5-coder:32b  (coding)
    │  Input:   ToolSpec
    │  Loads:   config/prompts/shared_rules.txt as system message
    │  Output:  Python function, wrapped with @mcp.tool() decorator + imports
    │  Format:  black formatter applied post-generation
    │  Guard:   detects truncated output → injects RuntimeError to force repair
    │
    ▼
[validator_node]  ──────────── (no LLM — static analysis only)
    │  Checks:  AST syntax, schema conformance, return statement presence
    │
    ├── PASS ──────────────────────────────────────────────────────────────┐
    │                                                                      │
    └── FAIL → [repair_node]  ── qwen2.5-coder:32b  (coding)             │
                    │  Sends errors back to LLM to patch code             │
                    │  Max 5 repair attempts (config: max_repair_attempts) │
                    └──────► [validator_node]  (loop back)                │
                                                                          ▼
                                                               [executor_node]
                                                                    │  (no LLM — Docker sandbox)
                                                                    │  Runs generated tool in isolation
                                                                    │
                                                                    ▼
                                                               [promoter_node]
                                                                    │  Moves tool: draft/ → active/
                                                                    │
                                                                    ▼
                                                               [projection_node]
                                                                    │  Packages results into parent-graph fields
                                                                    │  (tool_transcript, artifact_log, errors…)
                                                                    ▼
                                                                   END
```

---

## 4. Two LLM Call Patterns

### `generate()` — Free-form text (used for code generation)

- Sends the prompt as a user message with an optional system message.
- Strips `<think>...</think>` reasoning blocks emitted by deepseek-r1.
- Passes `extra_body={"options": {"num_ctx": ..., "num_predict": ...}}` to override Ollama context/token limits per-request.
- Default temperature: `0.2` for code (low, favours determinism).

### `generate_structured()` — JSON output (used for intent & spec generation)

- Forces `temperature=0.0`.
- Retries up to **3 times** on `JSONDecodeError` with a retry hint appended to the prompt.
- Strips markdown code fences (`\`\`\`json`) and extracts raw JSON between the first `{` and last `}`.
- System message instructs the model to output **pure JSON only**, no commentary.

---

## 5. Per-Stage Model Assignment

| Pipeline Stage | Model Used | Call Type | Reason |
|---|---|---|---|
| Intent extraction | `deepseek-r1:70b` | `generate_structured()` | Needs strong reasoning to interpret ambiguous queries |
| Spec generation | `deepseek-r1:70b` | `generate_structured()` | Needs structured planning to produce a valid ToolSpec |
| Code generation | `qwen2.5-coder:32b` | `generate()` | Specialised coding model → higher quality Python output |
| Code repair | `qwen2.5-coder:32b` | `generate()` | Same model sees its own error to patch it |
| Validation | — | — | Static analysis only, no LLM |
| Execution | — | — | Docker sandbox, no LLM |

---

## 6. Key Design Decisions

1. **Single base URL, two models** — Ollama routes by the `model` field in the request body; no separate client instances are needed for each model.

2. **`max_retries=0` on the SDK** — The SDK's own retry logic is disabled so that `generate_structured()`'s own 3-attempt loop is the sole source of retries. SDK retries on top would cause confusing log spam and double-wait on model load errors.

3. **Truncation detection** — After code generation, the code generator parses the AST to check if the generated function has no `return` statement outside nested defs. If truncated, it injects `raise RuntimeError("TRUNCATED_OUTPUT: ...")` into the function body so sandbox execution fails with a clear signal, triggering the repair loop rather than silently promoting broken code.

4. **`.env` override pattern** — `LLM_BASE_URL` in `.env` takes priority over `config/config.yaml`, so developers on client machines never have to edit the config file directly.

5. **`num_ctx` capped at 16384** — The default Ollama value allocates ~40 GB of KV cache. Each pipeline stage uses fewer than 2000 tokens, so 16384 is more than sufficient and reduces memory pressure and latency significantly.

---

## 7. Key Files Reference

| File | Role |
|---|---|
| `config/config.yaml` | Ollama URL, model names, timeouts, `num_ctx`, `num_predict` |
| `src/llm_client.py` | `QwenLLMClient` — OpenAI-compatible client wrapping Ollama; `generate()` and `generate_structured()` |
| `src/pipeline.py` | LangGraph `StateGraph` — wires all nodes together, defines edges and routing |
| `src/intent_extraction.py` | Uses `reasoning` model → structured JSON intent |
| `src/spec_generator.py` | Uses `reasoning` model → `ToolSpec` Pydantic object |
| `src/code_generator.py` | Uses `coding` model → Python MCP tool code with truncation guard |
| `src/validator.py` | Static AST checks, no LLM |
| `src/executor.py` | Docker sandbox execution, no LLM |
| `src/feedback_handler.py` | Parses human-in-the-loop approval responses |
| `config/prompts/` | Prompt templates loaded at runtime per pipeline stage |
