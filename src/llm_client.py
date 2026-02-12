"""LLM Client Module - Abstraction layer for LLM interactions."""

import json
import yaml
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
from openai import OpenAI


# ============================================================================
# Base Interface
# ============================================================================

class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.7, system_message: str = None) -> str:
        """Generate free-form text response.
        
        Args:
            prompt: Input prompt for the LLM
            temperature: Sampling temperature (0.0-1.0)
            system_message: Optional system message to guide behavior
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def generate_structured(self, prompt: str, schema: Dict, system_message: str = None) -> Dict:
        """Generate structured JSON response.
        
        Args:
            prompt: Input prompt with instructions for JSON output
            schema: Expected JSON schema (for documentation)
            system_message: Optional system message to enforce JSON output
            
        Returns:
            Parsed JSON dictionary
        """
        pass


# ============================================================================
# Qwen Implementation
# ============================================================================

class QwenLLMClient(BaseLLMClient):
    """LLM client for Qwen 2.5-Coder via vLLM server."""
    
    def __init__(self, config_path: str = "config/config.yaml", model_override: str = None):
        """Initialize Qwen client with configuration.
        
        Args:
            config_path: Path to YAML configuration file
            model_override: Optional model name to override config (e.g., 'reasoning', 'coding', or full model name)
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_file) as f:
            self.config = yaml.safe_load(f)
        
        # Initialize OpenAI-compatible client
        self.client = OpenAI(
            base_url=self.config["llm"]["base_url"],
            api_key="not-needed"  # vLLM doesn't require API key
        )
        
        # Determine which model to use - must be 'reasoning' or 'coding'
        if not model_override:
            raise ValueError("model_override is required - must specify 'reasoning' or 'coding'")
        
        if model_override not in ["reasoning", "coding"]:
            raise ValueError(f"model_override must be 'reasoning' or 'coding', got: {model_override}")
        
        # Get the specific model for this task type
        models_dict = self.config["llm"].get("models", {})
        if model_override not in models_dict:
            raise KeyError(f"Model type '{model_override}' not found in config. Available: {list(models_dict.keys())}")
        
        self.model = models_dict[model_override]
        
        self.default_temperature = self.config["llm"].get("temperature", 0.3)
    
    def generate(self, prompt: str, temperature: float = None, system_message: str = None) -> str:
        """Generate free-form text response.
        
        Args:
            prompt: Input prompt for the LLM
            temperature: Sampling temperature (uses config default if None)
            system_message: Optional system message to guide model behavior
            
        Returns:
            Generated text response
        """
        if temperature is None:
            temperature = self.default_temperature
        
        try:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {e}")
    
    def generate_structured(self, prompt: str, schema: Optional[Dict] = None, system_message: str = None) -> Dict:
        """Generate structured JSON response.
        
        Args:
            prompt: Input prompt requesting JSON output
            schema: Expected JSON schema (for documentation, not enforced)
            system_message: Optional system message to enforce JSON-only output
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            ValueError: If response is not valid JSON
        """
        # Use strict system message for structured output if not provided
        if system_message is None:
            system_message = (
                "You are a JSON generation system. "
                "Return ONLY valid JSON conforming to the schema. "
                "DO NOT include any explanatory text, thinking process, commentary, or meta-text. "
                "DO NOT use <think> tags or similar reasoning markers. "
                "DO NOT add markdown code fences around the JSON. "
                "Output must be pure JSON starting with { and ending with }."
            )
        
        # Use very low temperature for structured output
        response = self.generate(prompt, temperature=0.0, system_message=system_message)
        
        try:
            # Remove <think> tags if present (common with reasoning models)
            if "<think>" in response and "</think>" in response:
                # Extract content after </think>
                response = response.split("</think>", 1)[1].strip()
            
            # Try to extract JSON from markdown code blocks if present
            if "```json" in response:
                # Extract JSON from markdown code block
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                # Generic code block
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
            else:
                # Look for JSON object boundaries
                # Find first { and last }
                first_brace = response.find("{")
                last_brace = response.rfind("}")
                
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    json_str = response[first_brace:last_brace+1].strip()
                else:
                    json_str = response.strip()
            
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # Enhanced error message showing what we tried to parse
            raise ValueError(
                f"Failed to parse JSON response: {e}\n"
                f"Extracted JSON string (first 500 chars): {json_str[:500]}\n"
                f"Full response (first 1000 chars): {response[:1000]}"
            )


# ============================================================================
# Factory Function
# ============================================================================

def create_llm_client(config_path: str = "config/config.yaml", model_type: str = "coding") -> BaseLLMClient:
    """Factory function to create LLM client.
    
    Args:
        config_path: Path to configuration file
        model_type: Model type to use ('reasoning' or 'coding') - REQUIRED
        
    Returns:
        Configured LLM client instance
    """
    if model_type not in ["reasoning", "coding"]:
        raise ValueError(f"model_type must be 'reasoning' or 'coding', got: {model_type}")
    return QwenLLMClient(config_path, model_override=model_type)
