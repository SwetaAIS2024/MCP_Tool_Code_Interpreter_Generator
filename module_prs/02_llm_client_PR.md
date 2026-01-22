# LLM Client Module - Project Requirements

**Module**: `src/llm_client.py`  
**Version**: 1.0.0  
**Dependencies**: `openai`, `yaml`, `pydantic`  
**Parent PR**: Main ProjectRequirements.instructions.md - Section 5.1.1

---

## 1. Module Purpose

Provide a unified interface for LLM interactions supporting:
- **Cloud providers**: OpenAI, Anthropic
- **On-premises**: Qwen2.5-Coder via vLLM/llama.cpp
- Task-specific model routing
- Prompt template management
- Response parsing and validation

---

## 2. Core Architecture

### 2.1 Unified Client Interface

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from enum import Enum

class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"  # vLLM, llama.cpp, TGI

class LLMTask(str, Enum):
    INTENT_EXTRACTION = "intent_extraction"
    SPEC_GENERATION = "spec_generation"
    CODE_GENERATION = "code_generation"
    CODE_REPAIR = "code_repair"
    VALIDATION = "validation"

class BaseLLMClient(ABC):
    """Abstract base for LLM clients"""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        task: LLMTask,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        response_schema: Dict[str, Any],
        task: LLMTask
    ) -> Dict[str, Any]:
        """Generate structured JSON response"""
        pass
```

---

## 3. Implementation

### 3.1 Qwen Local Client (Primary for On-Prem)

```python
from openai import OpenAI
import yaml
from pathlib import Path

class QwenLLMClient(BaseLLMClient):
    """Client for Qwen models via vLLM/llama.cpp (OpenAI-compatible API)"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with local vLLM server"""
        
        # Load configuration
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Connect to local server (OpenAI-compatible endpoint)
        self.client = OpenAI(
            base_url=self.config["llm"]["base_url"],
            api_key="not-needed"  # Local server
        )
        
        # Model configurations
        self.models = {
            "primary": self.config["llm"]["code_generation_model"],
            "fallback": self.config["llm"]["fallback_model"]
        }
        
        # Load prompt templates
        self.prompts = self._load_prompts()
    
    def generate(
        self,
        prompt: str,
        task: LLMTask,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_fallback: bool = False
    ) -> str:
        """Generate text response"""
        
        # Select model
        model_config = self.models["fallback" if use_fallback else "primary"]
        model_name = model_config["model_name"]
        
        # Get task-specific defaults
        temp = temperature or self._get_task_temperature(task)
        tokens = max_tokens or model_config.get("max_tokens", 4096)
        
        # System prompt based on task
        system_prompt = self._get_system_prompt(task)
        
        # Call model
        response = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temp,
            max_tokens=tokens,
            top_p=model_config.get("top_p", 0.95),
        )
        
        return response.choices[0].message.content
    
    def generate_structured(
        self,
        prompt: str,
        response_schema: Dict[str, Any],
        task: LLMTask
    ) -> Dict[str, Any]:
        """Generate JSON response matching schema"""
        
        # Enhanced prompt with JSON schema
        enhanced_prompt = f"""{prompt}

You must respond with valid JSON matching this schema:
{json.dumps(response_schema, indent=2)}

Respond only with the JSON object, no additional text."""
        
        response_text = self.generate(enhanced_prompt, task, temperature=0.1)
        
        # Parse JSON
        import json
        try:
            # Extract JSON from markdown code blocks if present
            if "```json" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_text = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_text = response_text
            
            result = json.loads(json_text)
            return result
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}\n{response_text}")
    
    def generate_tool_spec(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ToolSpec from user intent (high-level API)"""
        
        prompt = self.prompts["spec_generation"].format(**intent)
        
        spec_schema = {
            "type": "object",
            "properties": {
                "tool_name": {"type": "string"},
                "description": {"type": "string"},
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
                "when_to_use": {"type": "string"},
                "what_it_does": {"type": "string"},
                "returns": {"type": "string"},
                "prerequisites": {"type": "string"}
            },
            "required": ["tool_name", "description", "input_schema", "output_schema"]
        }
        
        return self.generate_structured(
            prompt,
            spec_schema,
            LLMTask.SPEC_GENERATION
        )
    
    def generate_code(self, spec: Dict[str, Any], reference_tool: Optional[str] = None) -> str:
        """Generate Python code from ToolSpec"""
        
        # Build prompt from template
        prompt_data = {
            "spec": spec,
            "reference_tool": reference_tool or self._load_reference_tool()
        }
        prompt = self.prompts["code_generation"].format(**prompt_data)
        
        return self.generate(
            prompt,
            LLMTask.CODE_GENERATION,
            temperature=0.2  # Low temp for precise code
        )
    
    def repair_code(
        self,
        code: str,
        errors: Dict[str, Any],
        spec: Dict[str, Any]
    ) -> str:
        """Fix code based on validation errors"""
        
        prompt_data = {
            "code": code,
            "errors": errors,
            "spec": spec
        }
        prompt = self.prompts["code_repair"].format(**prompt_data)
        
        return self.generate(
            prompt,
            LLMTask.CODE_REPAIR,
            temperature=0.3  # Slightly higher for creative fixes
        )
    
    def _get_task_temperature(self, task: LLMTask) -> float:
        """Get default temperature for task"""
        temps = {
            LLMTask.INTENT_EXTRACTION: 0.1,
            LLMTask.SPEC_GENERATION: 0.2,
            LLMTask.CODE_GENERATION: 0.2,
            LLMTask.CODE_REPAIR: 0.3,
            LLMTask.VALIDATION: 0.0
        }
        return temps.get(task, 0.2)
    
    def _get_system_prompt(self, task: LLMTask) -> str:
        """Get system prompt for task"""
        prompts = {
            LLMTask.SPEC_GENERATION: (
                "You are an expert at designing clean, contract-first tool specifications. "
                "Generate precise JSON schemas and clear documentation."
            ),
            LLMTask.CODE_GENERATION: (
                "You are an expert Python developer specializing in data analysis tools. "
                "Generate clean, idiomatic, type-safe code with comprehensive error handling. "
                "Follow PEP 8 and use pandas best practices."
            ),
            LLMTask.CODE_REPAIR: (
                "You are an expert at debugging and fixing Python code. "
                "Analyze errors carefully and provide minimal, targeted fixes."
            ),
        }
        return prompts.get(task, "You are a helpful AI assistant.")
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load prompt templates from files"""
        prompts = {}
        prompt_dir = Path("prompts")
        
        for template_file in ["spec_generation.txt", "code_generation.txt", "code_repair.txt"]:
            path = prompt_dir / template_file
            if path.exists():
                with open(path) as f:
                    prompts[template_file.replace(".txt", "")] = f.read()
        
        return prompts
    
    def _load_reference_tool(self) -> str:
        """Load reference tool example"""
        ref_path = Path("reference_files/sample_mcp_tools/load_and_analyze_csv.py")
        if ref_path.exists():
            with open(ref_path) as f:
                return f.read()
        return ""
```

---

### 3.2 Cloud Provider Clients (Optional)

```python
class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude models"""
    
    def __init__(self, api_key: str):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)
    
    def generate(self, prompt: str, task: LLMTask, **kwargs) -> str:
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature", 0.2),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

class OpenAIClient(BaseLLMClient):
    """Client for OpenAI GPT models"""
    
    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
    
    def generate(self, prompt: str, task: LLMTask, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens", 4096)
        )
        return response.choices[0].message.content
```

---

### 3.3 Client Factory

```python
import os

def create_llm_client(config_path: str = "config.yaml") -> BaseLLMClient:
    """Factory to create appropriate LLM client"""
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    provider = config["llm"]["provider"]
    
    if provider == "local":
        return QwenLLMClient(config_path)
    elif provider == "anthropic":
        return AnthropicClient(os.getenv("ANTHROPIC_API_KEY"))
    elif provider == "openai":
        return OpenAIClient(os.getenv("OPENAI_API_KEY"))
    else:
        raise ValueError(f"Unknown provider: {provider}")
```

---

## 4. Configuration

### 4.1 config.yaml Structure

```yaml
llm:
  provider: "local"  # "local" | "openai" | "anthropic"
  base_url: "http://localhost:8000/v1"  # For local vLLM
  
  # Primary model (Qwen2.5-Coder-32B)
  code_generation_model:
    model_name: "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"
    quantization: "awq"
    max_tokens: 4096
    temperature: 0.2
    top_p: 0.95
    gpu_memory_utilization: 0.6
  
  # Fallback model (Qwen2.5-Coder-7B)
  fallback_model:
    model_name: "Qwen/Qwen2.5-Coder-7B-Instruct"
    quantization: "fp16"
    max_tokens: 2048
    temperature: 0.2
    gpu_memory_utilization: 0.3
  
  # Task routing
  task_routing:
    use_fallback_for:
      - "simple_groupby"
      - "basic_filter"
      - "standard_aggregation"
```

---

## 5. Prompt Templates

### 5.1 prompts/spec_generation.txt

```
You are designing a ToolSpec for an MCP data analysis tool.

USER INTENT:
Operation: {operation}
Required Columns: {columns}
Metrics: {metrics}
Input File: {input_file}
Expected Output: {expected_output}

DATASET SCHEMA:
{dataset_schema}

Generate a complete ToolSpec following this pattern:

REFERENCE TOOL EXAMPLE:
{reference_tool_spec}

Your ToolSpec must include:
1. tool_name: descriptive snake_case name
2. description: one-sentence summary
3. input_schema: JSON Schema with required parameters
4. output_schema: JSON Schema for return value
5. when_to_use: trigger conditions
6. what_it_does: step-by-step logic
7. returns: output format specification
8. prerequisites: required prior steps

Output only valid JSON matching the ToolSpec schema.
```

### 5.2 prompts/code_generation.txt

```
Generate a complete Python MCP tool implementation from this ToolSpec:

{spec}

REQUIREMENTS:
1. Use @mcp.tool() decorator
2. Use Annotated types with Field descriptions
3. Implement error handling with try/except
4. Follow this structure:
   - Load data with load_csv_data_with_types()
   - Validate with validate_file_and_columns()
   - Perform transformation (pandas operations)
   - Format output as markdown
   - Add JSON footer: <!--output_json:{{...}}-->

5. Match the style of this reference tool:

{reference_tool}

6. Type hints for all function parameters and return values
7. Comprehensive docstring with WHEN/WHAT/RETURNS/PREREQUISITES
8. Handle edge cases: empty data, missing columns, type mismatches

Output only the complete Python code, no explanations.
```

### 5.3 prompts/code_repair.txt

```
Fix this Python tool based on validation errors:

ORIGINAL CODE:
{code}

VALIDATION ERRORS:
{errors}

TOOLSPEC (MUST PRESERVE):
{spec}

INSTRUCTIONS:
1. Fix all validation errors while preserving the ToolSpec contract
2. Do NOT change function signature or return type
3. Maintain all error handling
4. Keep the same logic flow
5. Fix only what's broken

If errors include:
- Schema errors: Fix parameter types/annotations
- Runtime errors: Add validation, handle edge cases
- Test failures: Fix logic to match expected behavior

Output only the corrected Python code, no explanations.
```

---

## 6. Testing Requirements

### 6.1 Unit Tests

```python
# tests/test_llm_client.py

def test_qwen_client_initialization():
    client = QwenLLMClient("config.yaml")
    assert client.client is not None
    assert "primary" in client.models

def test_generate_basic():
    client = QwenLLMClient()
    response = client.generate(
        "Write a function to add two numbers",
        LLMTask.CODE_GENERATION
    )
    assert "def" in response
    assert len(response) > 0

def test_generate_structured():
    client = QwenLLMClient()
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    
    response = client.generate_structured(
        "Generate a person with name John",
        schema,
        LLMTask.SPEC_GENERATION
    )
    
    assert "name" in response
    assert response["name"] == "John"

def test_generate_tool_spec():
    client = QwenLLMClient()
    intent = {
        "operation": "groupby",
        "columns": ["region"],
        "metrics": ["count"],
        "input_file": "data.csv"
    }
    
    spec = client.generate_tool_spec(intent)
    
    assert "tool_name" in spec
    assert "input_schema" in spec
    assert spec["input_schema"]["type"] == "object"

def test_task_temperature():
    client = QwenLLMClient()
    
    assert client._get_task_temperature(LLMTask.CODE_GENERATION) == 0.2
    assert client._get_task_temperature(LLMTask.CODE_REPAIR) == 0.3
    assert client._get_task_temperature(LLMTask.VALIDATION) == 0.0

def test_client_factory():
    client = create_llm_client("config.yaml")
    assert isinstance(client, BaseLLMClient)
```

### 6.2 Integration Tests

```python
def test_end_to_end_generation(qwen_server_running):
    """Test full pipeline: intent -> spec -> code"""
    
    client = QwenLLMClient()
    
    # Step 1: Generate spec
    intent = {...}
    spec = client.generate_tool_spec(intent)
    
    # Step 2: Generate code
    code = client.generate_code(spec)
    
    # Verify code is valid Python
    import ast
    ast.parse(code)
    
    # Verify contains required patterns
    assert "@mcp.tool()" in code
    assert "def " in code
```

---

## 7. Implementation Checklist

- [ ] Implement BaseLLMClient abstract class
- [ ] Implement QwenLLMClient with vLLM support
- [ ] Add OpenAIClient and AnthropicClient (optional)
- [ ] Create client factory
- [ ] Implement prompt template loading
- [ ] Implement structured JSON generation
- [ ] Add task-specific temperature defaults
- [ ] Add system prompt selection
- [ ] Create prompt templates (spec, code, repair)
- [ ] Add error handling and retries
- [ ] Write unit tests (>90% coverage)
- [ ] Write integration tests with mock LLM server
- [ ] Add configuration validation
- [ ] Document API with examples

---

## 8. Dependencies

```python
# requirements.txt (for this module)
openai>=1.12.0          # OpenAI-compatible client (works with vLLM)
anthropic>=0.18.0       # Optional: Anthropic Claude
pyyaml>=6.0
pydantic>=2.5.0
```

---

## 9. Performance Optimization

### 9.1 Response Caching

```python
from functools import lru_cache

class QwenLLMClient:
    @lru_cache(maxsize=128)
    def _get_cached_response(self, prompt_hash: str, task: str) -> str:
        """Cache repeated prompts"""
        pass
```

### 9.2 Async Support

```python
async def generate_async(
    self,
    prompt: str,
    task: LLMTask
) -> str:
    """Async generation for parallel requests"""
    pass
```

---

**Status**: Ready for Implementation  
**Priority**: P0 (Required by spec_generator, code_generator)  
**Estimated Effort**: 2-3 days  
**Prerequisites**: vLLM server running with Qwen2.5-Coder-32B
