# MCP Tool Code Interpreter Generator

**An agentic code-generation pipeline for data analysis MCP tools**

MCP_Tool_Code_Interpreter_Generator is a modular, agentic code-generation pipeline that turns natural-language analysis intents into executable MCP tools for data analysis workflows. It performs intent extraction, tool retrieval and gap detection, contract-first tool specification (schemas), code/package scaffolding, and sandbox validation with iterative repair. Newly generated tools are staged by default and promoted to the active MCP registry only after explicit positive user feedback on the tool's output, ensuring tool quality and preventing registry pollution.

---

## 🎯 Key Features

- **Intent-Driven Generation**: Converts natural language requests into executable data analysis tools
- **Contract-First Design**: Generates formal ToolSpecs with schemas before implementation
- **Multi-Stage Validation**: Schema checking, static analysis, and sandbox testing with auto-repair
- **Staged Execution**: Test tools in isolation before committing to registry
- **Feedback-Gated Promotion**: Tools only registered after explicit user approval
- **Zero Tool Pollution**: Strict acceptance criteria prevent low-quality tools

---

## 🏗️ Architecture Overview

```
User Request → Intent Extraction → Gap Detection → Tool Spec Generation
                                                            ↓
User Approval ← Present Output ← Staged Execution ← Code Generation
      ↓                                                     ↓
   Promote                                            Validation & Repair
      ↓                                                  (sandbox testing)
Active Registry
```

### Tool Lifecycle

```
DRAFT → STAGED → APPROVED → PROMOTED (active registry)
                     ↓
                 REJECTED (archived for learning)
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MCP_Tool_Code_Interpreter_Generator.git
cd MCP_Tool_Code_Interpreter_Generator

# Install dependencies
pip install -r requirements.txt

# Set up environment
export OPENAI_API_KEY="your-api-key"
# or create a .env file

# Initialize registry
python -m src.init_registry
```

### Basic Usage

```python
from src.pipeline import ToolGenerationPipeline

# Initialize pipeline
pipeline = ToolGenerationPipeline()

# Generate tool from natural language request
result = pipeline.process_request(
    "Analyze traffic accidents grouped by severity and location"
)

# Tool is staged, executed, and presented to user
# User approves/rejects → automatic promotion or archival
```

---

## 📋 Example: Generated Tool

**User Request**: "Analyze traffic accidents by severity"

**Generated Tool**:
```python
@mcp.tool()
def group_and_count_by_columns(
    file_path: Annotated[str, Field(description="Path to CSV file")],
    group_by_columns: Annotated[List[str], Field(description="Columns to group by")],
) -> str:
    """
    Group data by specified columns and count occurrences.
    
    WHEN TO USE: When you need to count occurrences across grouping dimensions
    WHAT IT DOES: Loads CSV, groups by columns, counts rows, returns markdown
    RETURNS: Markdown table with grouped counts
    PREREQUISITES: CSV file must exist and contain specified columns
    """
    # Implementation with validation, error handling, and output formatting
    ...
```

**Workflow**:
1. ✅ Tool generated and validated
2. ✅ Executed in staging with user's data
3. ✅ Output presented to user
4. 👤 User approves → tool promoted to active registry
5. ✅ Tool available for future requests

---

## 🔧 Core Components

### 1. Intent Extraction & Gap Detection
- Parse natural language requests
- Query active tool registry
- Identify missing capabilities
- Decision: reuse existing vs. generate new

### 2. Tool Specification Generation
- Contract-first ToolSpec with schemas
- Annotated parameter types
- Comprehensive documentation (WHEN/WHAT/RETURNS/PREREQUISITES)
- Return type definitions

### 3. Code Generation & Scaffolding
- MCP tool decorator and function signature
- Implementation following established patterns
- Error handling and validation
- Machine-readable JSON footers

### 4. Validation & Repair
- **Schema Validation**: Parameter/return type checking
- **Static Analysis**: mypy, pylint, security checks
- **Sandbox Testing**: Isolated execution with synthetic data
- **Iterative Repair**: LLM-driven fixes (max 3 iterations)

### 5. Staged Execution
- Run tool with real user data
- Capture execution metadata and artifacts
- Isolated from active registry
- Record assumptions and limitations

### 6. Feedback Capture
- Present output and tool summary
- Request explicit approval/rejection
- Strict acceptance criteria
- Default to rejection for ambiguity

### 7. Promotion/Archival
- **Approved**: Copy to active registry, update MCP server
- **Rejected**: Archive with feedback for learning
- Version management and duplicate prevention

---

## 📊 Data Models

### ToolCandidate

```python
class ToolCandidate(BaseModel):
    tool_id: str
    version: str
    spec: ToolSpec
    package_path: str
    code_hash: str
    spec_hash: str
    status: ToolStatus  # DRAFT | STAGED | APPROVED | REJECTED | PROMOTED
    validation_report: Optional[ValidationReport]
    run_artifacts: Optional[RunArtifacts]
    user_feedback: Optional[UserFeedback]
    created_at: str
    dependencies: List[str]
```

### Registry Structure

```
registry/
├── active/           # Promoted tools (PROMOTED)
│   ├── metadata.json
│   └── tools/
│       └── <tool_name>/
│           ├── tool.py
│           ├── spec.json
│           └── tests.py
├── staging/          # Validated candidates (STAGED)
│   └── candidates/
└── archive/          # Rejected tools (learning data)
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_spec_generator.py

# Run with coverage
pytest --cov=src tests/
```

### Test Coverage Goals
- Unit tests: >90%
- Integration tests: End-to-end pipeline
- Validation tests: Schema/sandbox isolation

---

## 🛡️ Security & Safety

### Code Generation Safety
- Prohibit dangerous patterns (`eval`, `exec`, `subprocess`)
- Sandbox validation in isolated environments
- File access restricted to allowed paths
- Static analysis for security issues

### Tool Promotion Safety
- Never auto-register tools
- Require explicit user approval
- Version conflict detection
- Promotion audit log

---

## 📈 Metrics & Monitoring

Track key performance indicators:
- **Generation Success Rate**: Tools passing validation (target: >90%)
- **Approval Rate**: User-approved vs. rejected (target: >70%)
- **False Positive Rate**: Approved tools failing in production (target: <5%)
- **Time to Tool**: Request → staged execution (target: <60s)

---

## 🗺️ Roadmap

### Version 1.0 (MVP)
- [x] Intent extraction and gap detection
- [x] Contract-first spec generation
- [x] Code generation with validation
- [x] Staged execution pipeline
- [x] Feedback capture and promotion

### Version 1.1
- [ ] Enhanced auto-repair with feedback loop
- [ ] Support for JSON, Parquet, SQL data sources
- [ ] Tool versioning and upgrade paths
- [ ] Performance metrics and caching

### Version 2.0
- [ ] Multi-tool composition (tool chaining)
- [ ] Visual tool builder UI
- [ ] A/B testing for tool variants
- [ ] Automatic deprecation based on usage

### Version 2.1
- [ ] Visualization tools (matplotlib, plotly)
- [ ] Export tools as standalone packages
- [ ] Collaborative team registry

---

## 📚 Documentation

- **[Project Requirements](ProjectRequirements.instructions.md)**: Comprehensive specification
- **[Reference Examples](reference_files/)**: Sample tools and outputs
- **API Documentation**: Coming soon

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Implement changes with tests
4. Run validation (`pytest tests/`)
5. Submit a pull request with test coverage report

---

## 📝 License

[MIT License](LICENSE) - see LICENSE file for details

---

## 🙏 Acknowledgments

- Built on [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
- Powered by [FastMCP](https://github.com/jlowin/fastmcp)
- Validation with [Pydantic](https://docs.pydantic.dev/)

---

## 📧 Contact

For questions, issues, or feature requests, please open an issue on GitHub.

---

**Status**: 🚧 Active Development  
**Version**: 1.0.0-alpha  
**Last Updated**: 2026-01-22
