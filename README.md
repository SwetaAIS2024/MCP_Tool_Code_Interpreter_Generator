# MCP Tool Code Interpreter Generator

> **Autonomous MCP tool generator** using LangGraph orchestration and Qwen LLM

An intelligent system that automatically generates, validates, and deploys MCP (Model Context Protocol) tools from natural language queries. Built with LangGraph state machine orchestration and powered by Qwen 2.5-Coder for code generation.

---

## 🎯 Overview

**What it does:**
1. Takes a natural language data analysis query
2. Extracts structured intent and generates implementation plan
3. Creates formal tool specifications
4. Generates Python code with MCP decorators
5. Validates code in isolated sandbox
6. Executes and captures results
7. Gets human approval (two-stage feedback)
8. Promotes approved tools to active registry

**Example Query:**
```
"Show me the top 10 states by traffic accident count"
```

**Result:**
- Generated MCP tool: `analyze_accidents_by_state.py`
- Validated, tested, and ready to use
- Automatically added to tool registry

---

## 🏗️ Architecture

### Tech Stack
- **FastMCP** - MCP server framework
- **LangGraph** - StateGraph workflow orchestration
- **Qwen 2.5-Coder 32B** - On-premises code generation LLM
- **Pydantic v2.5+** - Data validation
- **Python 3.10+** - Core runtime

### High-Level Flow
```
User Query → Intent → Spec → Code → Validate → Execute → Approve → Promote
```

**📖 For detailed architecture, node descriptions, and implementation specs, see [module_prs/README.md](module_prs/README.md)**

---

## 📁 Project Structure

```
MCP_Tool_Code_Interpreter_Generator/
├── src/                    # Core modules (11 files)
├── tools/                  # Generated tools (draft/staged/active)
├── tests/                  # Test suite
├── config/                 # Configuration & prompts
├── docker/                 # Sandbox container
├── docs/                   # Architecture docs
├── module_prs/             # Implementation PRs
└── reference_files/        # Sample data & examples
```

See complete structure in [module_prs/README.md](module_prs/README.md#-project-file-structure).

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Configure environment
cp .env.example .env
# Edit .env with your LLM endpoint
```

### 2. Configure LLM Endpoint

Update `config/config.yaml`:
```yaml
llm:
  base_url: "http://localhost:8000/v1"
  model: "Qwen/Qwen2.5-Coder-32B-Instruct"
```

### 3. Run the Server

```bash
python run_server.py
```

### 4. Use the Tool

```python
# Via MCP client
analyze_data(
    query="Show top 10 states by accident count",
    file_path="data/traffic_accidents.csv"
)
```

---

## 📚 Documentation

### For Users
- **[Quick Start](#-quick-start)** - Get up and running
- **[Configuration](#-configuration)** - Setup guide
- **[Troubleshooting](#-troubleshooting)** - Common issues

### For Developers
- **[Module PRs](module_prs/README.md)** - Complete implementation specs (10 modules)
- **[Sandbox Security](docs/SANDBOX_SECURITY.md)** - Security policies
- **[Architecture Docs](docs/)** - Design decisions

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific module tests
pytest tests/test_intent_extraction.py -v

# Run with coverage
pytest --cov=src tests/

# Integration tests only
pytest tests/test_integration.py
```

---

## 🔒 Security

### Sandbox Isolation
All generated code runs in an isolated sandbox with:
- ✅ No network access
- ✅ No file system write access
- ✅ Resource limits (CPU, memory, timeout)
- ✅ Import restrictions (only safe libraries)

See [docs/SANDBOX_SECURITY.md](docs/SANDBOX_SECURITY.md) for details.

### Sandbox Modes
- **Subprocess** (default) - Fast, good for development
- **Docker** - Full isolation, production-ready

Configure in `config/config.yaml`:
```yaml
sandbox:
  mode: "subprocess"  # or "docker"
```

---

## 🛠️ Development

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

# List tools by state
python scripts/migrate_tools.py list active

# Migrate tool between states
python scripts/migrate_tools.py migrate my_tool draft staged
```

---

## 📊 Tool Lifecycle

```
DRAFT → STAGED → APPROVED → PROMOTED
  ↓                 ↓
REJECTED        ARCHIVED
```

### States
- **DRAFT** - Freshly generated code
- **STAGED** - Validated, ready for testing
- **APPROVED** - Executed successfully, user approved
- **PROMOTED** - In active registry, production-ready
- **REJECTED** - Failed validation or user rejected

---

## 🎯 Implementation Status

### Current Phase: Foundation
- ✅ Project structure
- ✅ Configuration files
- ✅ Test framework
- ⏳ Module implementation (see [module_prs/README.md](module_prs/README.md#-implementation-roadmap))

**For detailed roadmap and task breakdown, see [module_prs/README.md](module_prs/README.md)**

---

## 🤝 Contributing

1. Review [module_prs/](module_prs/) for implementation specs
2. Create feature branch from `main`
3. Follow coding standards (black, ruff, mypy)
4. Write tests for new functionality
5. Update documentation
6. Submit PR with clear description

---

## 📝 Configuration

### Main Config: `config/config.yaml`
```yaml
llm:
  base_url: "http://localhost:8000/v1"
  model: "Qwen/Qwen2.5-Coder-32B-Instruct"

paths:
  draft_dir: "./tools/draft"
  staged_dir: "./tools/staged"
  active_dir: "./tools/active"

validation:
  max_repair_attempts: 3
  sandbox_timeout_seconds: 30
```

### Sandbox Policy: `config/sandbox_policy.yaml`
- Allowed/blocked imports
- Resource limits
- Filesystem restrictions
- Network policies

### Prompt Templates: `config/prompts/`
- `intent_extraction.txt`
- `spec_generation.txt`
- `code_generation.txt`
- `code_repair.txt`

---

## 🐛 Troubleshooting

### Common Issues

**LLM not responding**
```bash
# Check vLLM server
curl http://localhost:8000/v1/models
```

**Validation failures**
- Review `ValidationReport.errors` for details
- Check sandbox logs in `tools/sandbox/logs/`

**Graph interrupts not working**
- Verify `interrupt_before` in pipeline configuration
- Check LangGraph version compatibility

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 📄 License

[Add license information]

---

## 🙏 Acknowledgments

- **FastMCP** - MCP server framework
- **LangGraph** - Workflow orchestration
- **Qwen Team** - Code generation LLM
- **Anthropic** - MCP protocol specification

---

**Version**: 1.0.0  
**Status**: In Development  
**Last Updated**: January 27, 2026

For detailed module documentation, see [module_prs/README.md](module_prs/README.md).
