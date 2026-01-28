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
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate free-form text response.
        
        Args:
            prompt: Input prompt for the LLM
            temperature: Sampling temperature (0.0-1.0)
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def generate_structured(self, prompt: str, schema: Dict) -> Dict:
        """Generate structured JSON response.
        
        Args:
            prompt: Input prompt with instructions for JSON output
            schema: Expected JSON schema (for documentation)
            
        Returns:
            Parsed JSON dictionary
        """
        pass


# ============================================================================
# Qwen Implementation
# ============================================================================

class QwenLLMClient(BaseLLMClient):
    """LLM client for Qwen 2.5-Coder via vLLM server."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize Qwen client with configuration.
        
        Args:
            config_path: Path to YAML configuration file
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
        self.model = self.config["llm"]["model"]
        self.default_temperature = self.config["llm"].get("temperature", 0.3)
    
    def generate(self, prompt: str, temperature: float = None) -> str:
        """Generate free-form text response.
        
        Args:
            prompt: Input prompt for the LLM
            temperature: Sampling temperature (uses config default if None)
            
        Returns:
            Generated text response
        """
        if temperature is None:
            temperature = self.default_temperature
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {e}")
    
    def generate_structured(self, prompt: str, schema: Optional[Dict] = None) -> Dict:
        """Generate structured JSON response.
        
        Args:
            prompt: Input prompt requesting JSON output
            schema: Expected JSON schema (for documentation, not enforced)
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            ValueError: If response is not valid JSON
        """
        # Use low temperature for structured output
        response = self.generate(prompt, temperature=0.2)
        
        try:
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
                json_str = response.strip()
            
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}\nResponse: {response}")


# ============================================================================
# Factory Function
# ============================================================================

def create_llm_client(config_path: str = "config/config.yaml") -> BaseLLMClient:
    """Factory function to create LLM client.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured LLM client instance
    """
    return QwenLLMClient(config_path)
