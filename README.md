# LangGraph Tool Code Generator & Interpreter

> **Autonomous code generation pipeline** using LangGraph orchestration and multi-model LLM approach

A reusable LangGraph workflow that automatically generates, validates, and executes Python data analysis tools from natural language queries. Designed as a composable subgraph for integration into larger agent systems. Built with LangGraph state machine orchestration and powered by specialized LLMs for reasoning and code generation.

---

## 🎯 Overview

**What it does:**
1. Takes a natural language data analysis query
2. Extracts structured intent using reasoning model (DeepSeek-R1)
3. Generates formal tool specifications
4. Generates Python code using specialized coding model (Qwen 2.5-Coder)
5. Validates code in isolated Docker/subprocess sandbox
6. Executes and captures analysis results
7. Promotes validated tools to active registry

**Use as a Child Graph:**
This pipeline is designed to be integrated into larger agent systems as a reusable LangGraph subgraph. Use `build_graph()` to get the compiled graph and integrate it into your parent workflow.

**Example Query:**
```
"Run ANOVA across groups, then perform Tukey HSD post-hoc test with p-values and effect sizes"
```

**Result:**
- Generated tool: `anova_tukeyhsd_traffic_injuries_<timestamp>.py`
- Statistical analysis with validated output
- Automatically added to active tool registry (`tools/active/`)

---

## 🏗️ Architecture

### Tech Stack
- **LangGraph** - StateGraph workflow orchestration and composability
- **DeepSeek-R1 70B** - Reasoning model for intent extraction and spec generation
- **Qwen 2.5-Coder 32B** - Code generation and repair
- **Ollama** - Local LLM inference server
- **Pydantic v2.5+** - Data validation
- **Python 3.10+** - Core runtime
- **Docker** - Sandbox isolation (optional subprocess mode available)

### Pipeline Flow

```
User Query 
    ↓
Intent Extraction (DeepSeek-R1)
    ↓
Spec Generation (DeepSeek-R1)
    ↓
Code Generation (Qwen 2.5-Coder)
    ↓
Validation (syntax + schema + sandbox)
    ├─→ PASS → Executor
    └─→ FAIL → Repair (Qwen 2.5-Coder, max 3 attempts)
              ↓
         Validator
    ↓
Executor (run on actual data)
    ├─→ SUCCESS → Promoter
    └─→ FAIL → END (with error report)
    ↓
Promoter (save to active registry)
    ↓
END
```

### LangGraph Nodes

The pipeline consists of the following nodes:

- **intent_node** - Extracts structured intent from natural language using reasoning model
- **spec_generator_node** - Creates formal tool specification with I/O schemas
- **code_generator_node** - Generates Python code implementing the specification
- **validator_node** - Validates syntax, schema compliance, and sandbox execution
- **repair_node** - Repairs code based on validation errors (max 3 attempts)
- **executor_node** - Executes the tool on actual user data
- **promoter_node** - Promotes successful tool to active registry

**📖 For detailed architecture and module descriptions, see [module_prs/README.md](module_prs/README.md)**

---

## 📁 Project Structure

```
MCP_Tool_Code_Interpreter_Generator/
├── src/                         # Core modules
│   ├── models.py               # Pydantic models and LangGraph state
│   ├── llm_client.py           # Multi-model LLM client
│   ├── intent_extraction.py    # Intent extraction node
│   ├── intent_validator.py     # Intent validation logic
│   ├── spec_generator.py       # Specification generation node
│   ├── code_generator.py       # Code generation + repair nodes
│   ├── validator.py            # Validation node  
│   ├── executor.py             # Execution node
│   ├── promoter.py             # Registry promotion node
│   ├── sandbox.py              # Sandboxed code execution
│   ├── pipeline.py             # LangGraph orchestrator & graph builder
│   └── logger_config.py        # Logging configuration
│
├── tools/                       # Generated tools
│   ├── draft/                  # Initial generated code
│   ├── active/                 # Promoted, production-ready tools
│   └── sandbox/                # Sandbox workspace for execution
│
├── output/                      # Execution results
│   ├── active/                 # Successful execution outputs
│   └── draft/                  # Failed/debug outputs
│
├── config/                      # Configuration files
│   ├── config.yaml             # Main configuration
│   ├── sandbox_policy.yaml     # Sandbox security policy
│   └── prompts/                # LLM prompt templates
│       ├── intent_extraction_v2.txt
│       ├── spec_generation.txt
│       ├── code_generation.txt
│       └── code_repair.txt
│
├── registry/                    # Tool registry
│   └── tools.json              # Active tool metadata
│
├── docker/                      # Docker sandbox
│   ├── Dockerfile.sandbox
│   └── docker-compose.sandbox.yml
│
├── tests/                       # Test suite
├── docs/                        # Documentation
└── test.py  # Interactive pipeline testing
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.ai/) installed and running
- Docker (optional, for Docker sandbox mode)

### 2. Setup Ollama Models

```bash
# Pull the reasoning model (for intent extraction & spec generation)
ollama pull deepseek-r1:70b

# Pull the coding model (for code generation & repair)
ollama pull qwen2.5-coder:32b

# Verify models are available
ollama list
```

### 3. Setup Environment

```bash
# Clone the repository (if applicable)
cd MCP_Tool_Code_Interpreter_Generator

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 4. Configure the System

The default configuration in `config/config.yaml` should work with Ollama:

```yaml
llm:
  base_url: "http://localhost:11434/v1"  # Ollama default endpoint
  
  models:
    reasoning: "deepseek-r1:70b"         # Intent + spec generation
    coding: "qwen2.5-coder:32b"          # Code generation + repair
    
  
  temperature: 0.0

sandbox:
  mode: "docker"  # or "subprocess" for faster testing
  timeout_seconds: 60
  memory_limit_mb: 512
```

### 5. Test the Pipeline

```bash
# Run interactive test with a sample query
python test.py "Calculate average values by group"

# Run with specific query
python test.py "Run ANOVA across groups with Tukey HSD post-hoc test"

# Adjust verbosity
python test.py --verbosity verbose "your query here"
python test.py --verbosity debug "your query here"
```

---

## 🔧 Multi-Model Configuration

The system uses **two specialized models** for optimal performance:

### Reasoning Model (DeepSeek-R1 70B)
- **Used for:** Intent extraction, spec generation
- **Why:** Better at understanding complex requirements and planning
- **Behavior:** May include `<think>` tags in reasoning process (automatically stripped)

### Coding Model (Qwen 2.5-Coder 32B)  
- **Used for:** Code generation, code repair
- **Why:** Specialized for generating clean, efficient Python code
- **Behavior:** Focused output without meta-commentary

See [MULTI_MODEL_SETUP.md](MULTI_MODEL_SETUP.md) for detailed configuration guide.

---

## 🔌 Integration & Usage

### As a Reusable Subgraph (Recommended)

This pipeline is designed as a composable LangGraph component. Integrate it into your parent agent:

```python
from src.pipeline import build_graph, run_pipeline
from langgraph.graph import StateGraph

# Method 1: Use run_pipeline() directly
result = run_pipeline(
    user_query="Calculate summary statistics grouped by category",
    data_path="data/your_data.csv"
)

# Method 2: Integrate the graph as a child node
tool_generator_graph = build_graph()

# Add to your parent workflow
parent = StateGraph(YourParentState)
parent.add_node("tool_generator", tool_generator_graph)
parent.add_edge("planner", "tool_generator")
parent.add_edge("tool_generator", "reviewer")
# ... continue building parent graph
```

### Standalone Testing

For development and testing, use the interactive test script:

```bash
# Test with a specific query
python test.py "your analysis query"

# Adjust verbosity
python test.py --verbosity debug "query here"
```

### Optional: As Standalone MCP Server

If you need to expose this as a standalone MCP server (not recommended for agent integration), you can use the optional `server.py` and `run_server.py` files:

```bash
# This is only needed if NOT using as a child graph
python run_server.py
```

**Note:** For agent integration, `server.py` is not needed - use `build_graph()` or `run_pipeline()` directly.

---

## 📊 Tool Lifecycle

```
DRAFT → Validation → Execution → PROMOTED
  ↓          ↓            ↓
  └─ (repair loop) ─────────→ REJECTED
```

### Tool States

Tools are stored in different directories based on their status:

- **DRAFT** (`tools/draft/`) - Freshly generated code, may have errors
- **ACTIVE** (`tools/active/`) - Validated and executed successfully, production-ready
- **Outputs** (`output/active/`) - Execution results from successful tool runs
- **Registry** (`registry/tools.json`) - Metadata for all active tools

### Output Organization

Generated outputs follow the naming pattern:
```
<operation>_<dataset>_<timestamp>_output.json
```

Example:
```
output/active/anova_tukeyhsd_traffic_injuries_20260210_171102_output.json
```

Each output includes:
- Original query and parameters
- Generated code
- Execution results
- Validation report
- Timestamps and metadata

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific module tests
pytest tests/test_intent_extraction.py -v
pytest tests/test_validator.py -v

# Run with coverage
pytest --cov=src tests/

# Integration tests
pytest tests/test_integration.py

# Interactive pipeline test
python test.py "your analysis query"
```

### Test Verbosity Levels

```bash
# Quiet - minimal output
python test.py --verbosity quiet "query"

# Normal - standard progress (default)
python test.py --verbosity normal "query"

# Verbose - detailed step information
python test.py --verbosity verbose "query"

# Debug - full LLM prompts and responses
python test.py --verbosity debug "query"
```

---

## 🔒 Security

### Sandbox Isolation

All generated code runs in an isolated sandbox with:

- ✅ **No network access** - Prevents data exfiltration
- ✅ **Restricted file system** - Read-only access to data files only
- ✅ **Resource limits** - CPU, memory, and timeout constraints
- ✅ **Import restrictions** - Only allowlisted libraries permitted
- ✅ **Subprocess restrictions** - No shell commands or external processes

### Sandbox Modes

**1. Docker Mode (Recommended for Production)**
```yaml
sandbox:
  mode: "docker"
  timeout_seconds: 60
  memory_limit_mb: 512
```

- Full container isolation
- Complete environment control  
- Higher security guarantees
- Slower startup time

**2. Subprocess Mode (Development)**
```yaml
sandbox:
  mode: "subprocess"
  timeout_seconds: 30
  memory_limit_mb: 512
```

- Faster execution
- Shared host environment
- Lower isolation guarantees
- Good for rapid iteration

### Security Policy

Configure allowed/blocked imports in `config/sandbox_policy.yaml`:

```yaml
allowed_libraries:
  - pandas
  - numpy
  - scipy
  - statsmodels
  - matplotlib
  - seaborn

blocked_imports:
  - os
  - subprocess
  - sys
  - requests
  - urllib
```

See [docs/SANDBOX_SECURITY.md](docs/SANDBOX_SECURITY.md) for comprehensive security documentation.

---

## 📝 Configuration

### Main Config: `config/config.yaml`

```yaml
# LLM Configuration
llm:
  base_url: "http://localhost:11434/v1"
  
  models:
    reasoning: "deepseek-r1:70b"    # Intent extraction & spec generation
    coding: "qwen2.5-coder:32b"     # Code generation & repair
    
  temperature: 0.0

# Directory paths
paths:
  draft_dir: "./tools/draft"
  staged_dir: "./tools/staged"
  active_dir: "./tools/active"
  registry: "./registry/tools.json"
  sandbox_workspace: "./tools/sandbox"

# Validation settings
validation:
  max_repair_attempts: 3            # Code repair retry limit
  sandbox_timeout_seconds: 30

# Sandbox configuration
sandbox:
  mode: "docker"                    # "docker" or "subprocess"
  timeout_seconds: 60               # Execution timeout
  memory_limit_mb: 512              # Memory limit

# Logging
logging:
  level: "INFO"                     # DEBUG, INFO, WARNING, ERROR
  file: "./logs/pipeline.log"
```

### Prompt Templates: `config/prompts/`

- `intent_extraction_v2.txt` - Extract structured intent from queries
- `spec_generation.txt` - Generate tool specifications
- `code_generation.txt` - Generate Python implementation code
- `code_repair.txt` - Repair code based on validation errors

### Sandbox Policy: `config/sandbox_policy.yaml`

Controls security restrictions for code execution:
- Allowed/blocked Python imports
- Resource limits (CPU, memory, timeout)
- Filesystem access restrictions
- Network policies

---

## 🎯 Implementation Status

### ✅ Completed Features

- ✅ Multi-model LLM integration (DeepSeek-R1 + Qwen 2.5-Coder)
- ✅ LangGraph pipeline orchestration
- ✅ Intent extraction with structured output
- ✅ Specification generation with I/O schemas
- ✅ Code generation with FastMCP decorators
- ✅ Multi-stage validation (syntax, schema, sandbox)
- ✅ Automated code repair (up to 3 attempts)
- ✅ Sandboxed execution (Docker + subprocess modes)
- ✅ Tool registry and promotion system
- ✅ Comprehensive logging and debugging
- ✅ Statistical analysis support (ANOVA, Tukey HSD, etc.)
- ✅ Graph visualization (Mermaid diagrams)

### 🔄 Active Development

- 🔄 Enhanced error recovery strategies
- 🔄 Additional statistical operations
- 🔄 Performance optimizations
- 🔄 Extended test coverage

**For detailed implementation specs, see [module_prs/README.md](module_prs/README.md)**

---

## �️ Development

### Setup Dev Environment

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run linter
ruff check src/

# Format code
black src/ tests/

# Type checking
mypy src/
```

### Utility Scripts

```bash
# Clean sandbox temporary files
python scripts/clean_sandbox.py

# Generate graph visualization
python visualize_graph.py
```

### Graph Visualization

The pipeline automatically generates a Mermaid diagram (`pipeline_graph.mmd`) showing the LangGraph workflow. View it at [mermaid.live](https://mermaid.live).

---

## 🐛 Troubleshooting

### Common Issues

**1. Ollama Connection Failed**

```bash
# Check if Ollama is running
ollama list

# Check API endpoint
curl http://localhost:11434/v1/models

# Restart Ollama if needed
# (OS-specific restart command)
```

**2. Models Not Found**

```bash
# Pull required models
ollama pull deepseek-r1:70b
ollama pull qwen2.5-coder:32b

# Verify models are available
ollama list
```

**3. Validation Failures**

- Review `ValidationReport.errors` in output
- Check generated code in `tools/draft/`
- Examine validation details in logs
- Review sandbox execution logs

**4. Docker Sandbox Issues**

```bash
# Check Docker is running
docker ps

# Build sandbox image
cd docker
docker-compose -f docker-compose.sandbox.yml build

# Check sandbox logs
docker logs <container_id>
```

**5. Import Errors in Sandbox**

- Verify library is listed in `config/sandbox_policy.yaml`
- Check library is installed in sandbox environment
- For Docker mode: rebuild sandbox image after adding libraries

**6. Memory/Timeout Errors**

Adjust limits in `config/config.yaml`:
```yaml
sandbox:
  timeout_seconds: 120      # Increase timeout
  memory_limit_mb: 1024     # Increase memory
```

### Debug Mode

Enable detailed logging to troubleshoot issues:

```bash
# Set debug verbosity
python test.py --verbosity debug "query"
```

Or configure in code:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check logs in `logs/pipeline.log` for detailed execution traces.

### Output Inspection

```bash
# Check draft tools
ls -la tools/draft/

# Check active tools
ls -la tools/active/

# Check execution outputs
ls -la output/active/

# View specific output
cat output/active/anova_tukeyhsd_traffic_injuries_<timestamp>_output.json
```

---

## 📚 Documentation

### For Users
- **[Quick Start](#-quick-start)** - Get up and running
- **[Configuration](#-configuration)** - System configuration
- **[Testing](#-testing)** - Run tests and validate
- **[Troubleshooting](#-troubleshooting)** - Common issues and solutions

### For Developers
- **[Module PRs](module_prs/README.md)** - Complete implementation specs
- **[Multi-Model Setup](MULTI_MODEL_SETUP.md)** - LLM configuration guide
- **[Sandbox Security](docs/SANDBOX_SECURITY.md)** - Security policies and implementation
- **[Logging](docs/LOGGING.md)** - Logging configuration and usage
- **[Architecture Docs](docs/)** - Design decisions and diagrams

---

## 🤝 Contributing

1. Review [module_prs/](module_prs/) for implementation specs
2. Create feature branch from `main`
3. Follow coding standards (black, ruff, mypy)
4. Write tests for new functionality
5. Update documentation
6. Submit PR with clear description

---

## 📄 License

[Add license information]

---

## 🙏 Acknowledgments

- **LangGraph** - Workflow orchestration and graph composition
- **Ollama** - Local LLM inference
- **DeepSeek** - Reasoning model
- **Qwen Team** - Code generation model

---

**Version**: 1.0.0  
**Status**: ✅ Production Ready  
**Last Updated**: February 12, 2026

For detailed module documentation and implementation specs, see [module_prs/README.md](module_prs/README.md).
