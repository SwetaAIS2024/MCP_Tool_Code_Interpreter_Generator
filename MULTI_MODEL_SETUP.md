# Multi-Model Configuration

## Overview
The pipeline now uses **two specialized models** for different tasks:
- **Reasoning Model** (`deepseek-r1:70b`) - For intent extraction and specification generation
- **Coding Model** (`qwen2.5-coder:32b`) - For code generation and repair

This approach provides:
- Better quality implementation plans from the reasoning model
- Faster, more focused code generation from the coder model
- Controlled response format for each task type

## Configuration

### Config File ([config.yaml](config/config.yaml))
```yaml
llm:
  base_url: "http://localhost:11434/v1"
  
  models:
    reasoning: "deepseek-r1:70b"  # For intent & spec generation
    coding: "qwen2.5-coder:32b"   # For code generation & repair
    default: "qwen2.5-coder:32b"
  
  model: "qwen2.5-coder:32b"  # Fallback
  temperature: 0.3
```

## Implementation Details

### 1. LLM Client Updates ([src/llm_client.py](src/llm_client.py))
- Added `model_override` parameter to `QwenLLMClient.__init__`
- Supports model type aliases: `"reasoning"`, `"coding"`, `"default"`
- Added system message support to control model behavior
- Enhanced JSON extraction to handle `<think>` tags from reasoning models
- Temperature set to 0.0 for structured output

### 2. Node Updates

#### Intent Extraction Node ([src/intent_extraction.py](src/intent_extraction.py))
```python
# Uses reasoning model for better planning
llm_client = create_llm_client(model_type="reasoning")
```

#### Spec Generator Node ([src/spec_generator.py](src/spec_generator.py))
```python
# Uses reasoning model for spec planning
llm_client = create_llm_client(model_type="reasoning")
```

#### Code Generator Node ([src/code_generator.py](src/code_generator.py))
```python
# Uses coding model for code generation
llm_client = create_llm_client(model_type="coding")
```

#### Repair Node ([src/code_generator.py](src/code_generator.py))
```python
# Uses coding model for code repair
llm_client = create_llm_client(model_type="coding")
```

### 3. Prompt Improvements ([config/prompts/intent_extraction.txt](config/prompts/intent_extraction.txt))

**Key changes to prevent reasoning model from including meta-text:**

```
CRITICAL INSTRUCTIONS:
- Return ONLY valid JSON conforming to the schema below
- DO NOT include any explanatory text, thinking process, or commentary
- DO NOT use <think> tags or similar meta-text
- DO NOT add markdown code fences (```) around the JSON
- Output must be pure JSON starting with opening brace and ending with closing brace
```

**Added operation selection guide:**
```
OPERATION SELECTION GUIDE:
 - "top N X by Y" or "most common X" → use "groupby_aggregate"
 - "filter by X" → use "filter"
 - "summary statistics" → use "describe_summary"
 - etc.
```

**Better column mapping examples:**
```
- "accident types" → "crash_type" or "first_crash_type" (NOT injury columns)
- "top 5 X" → operation="sort_limit", limit=5, group_by=[X]
```

### 4. JSON Parsing Improvements

System message enforces strict JSON output:
```python
system_message = (
    "You are a JSON generation system. "
    "Return ONLY valid JSON conforming to the schema. "
    "DO NOT include any explanatory text, thinking process, commentary, or meta-text. "
    "DO NOT use <think> tags or similar reasoning markers. "
    "Output must be pure JSON starting with { and ending with }."
)
```

Enhanced extraction handles:
- `<think>` tags removal
- Markdown code fence extraction
- Smart brace boundary detection

### 5. Bug Fixes

**Spec Generator** - Fixed TypeError when LLM returns lists with dicts:
```python
# Convert all items to strings before joining
spec_dict[field] = " ".join(str(item) for item in spec_dict[field])
```

## Benefits

1. **Quality**: Reasoning model produces better implementation plans with detailed steps
2. **Speed**: Coding model generates code faster without unnecessary reasoning
3. **Control**: Strict system messages prevent unwanted output formats
4. **Flexibility**: Easy to swap models by updating config.yaml
5. **Cost-effective**: Use expensive reasoning model only where needed

## Testing

Run the test pipeline:
```bash
python test_pipeline_with_feedback.py
```

Expected behavior:
- Intent extraction uses `deepseek-r1:70b` (reasoning)
- Spec generation uses `deepseek-r1:70b` (reasoning)
- Code generation uses `qwen2.5-coder:32b` (coding)
- Code repair uses `qwen2.5-coder:32b` (coding)

## Future Enhancements

1. **Semantic validation** - Ensure generated code matches intent (e.g., if intent says limit=5, code must contain `.head(5)`)
2. **Model routing logic** - Auto-select model based on task complexity
3. **Response format constraints** - Use JSON schema validation for strict adherence
4. **Performance monitoring** - Track model selection effectiveness
