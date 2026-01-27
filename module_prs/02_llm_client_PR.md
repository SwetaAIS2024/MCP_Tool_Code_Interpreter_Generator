# LLM Client Module

**Module**: `src/llm_client.py`  
**Priority**: P0  
**Effort**: 2-3 days

---

## Core Interface

```python
class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        pass
    
    @abstractmethod
    def generate_structured(self, prompt: str, schema: Dict) -> Dict:
        pass
```

---

## Qwen Implementation

```python
class QwenLLMClient(BaseLLMClient):
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.client = OpenAI(
            base_url=self.config["llm"]["base_url"],
            api_key="not-needed"
        )
        self.model = self.config["llm"]["model"]
    
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content
    
    def generate_structured(self, prompt: str, schema: Dict) -> Dict:
        response = self.generate(prompt)
        return json.loads(response)
```

---

## Usage

```python
llm = QwenLLMClient()
result = llm.generate("Generate tool spec for...", temperature=0.3)
```

---

## Implementation Checklist

- [ ] Implement base client interface
- [ ] Implement Qwen client
- [ ] Add prompt templates
- [ ] Write tests
- [ ] Add error handling
