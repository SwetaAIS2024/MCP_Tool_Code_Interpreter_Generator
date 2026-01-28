"""Intent Extraction Module - Extract structured intent from natural language queries."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from src.models import ToolGeneratorState
from src.llm_client import QwenLLMClient


# ============================================================================
# Intent Extractor
# ============================================================================

class IntentExtractor:
    """Extract structured intent and implementation plan from user queries."""
    
    def __init__(self, llm_client: QwenLLMClient):
        """Initialize with LLM client.
        
        Args:
            llm_client: Configured Qwen LLM client
        """
        self.llm = llm_client
        self.prompt_template_path = Path("config/prompts/intent_extraction.txt")
    
    def extract(self, query: str, data_path: str) -> Dict[str, Any]:
        """Extract structured intent from user query.
        
        Args:
            query: Natural language data analysis request
            data_path: Path to the dataset
            
        Returns:
            Dictionary with operation, columns, metrics, implementation_plan, etc.
        """
        # Load dataset schema for context
        df_preview = pd.read_csv(data_path, nrows=5)
        columns = list(df_preview.columns)
        dtypes = {col: str(dtype) for col, dtype in df_preview.dtypes.to_dict().items()}
        sample_values = {col: df_preview[col].head(3).tolist() for col in columns}
        
        # Build comprehensive analysis prompt
        prompt = self._build_prompt(query, data_path, columns, dtypes, sample_values)
        
        # Define expected JSON schema
        schema = {
            "type": "object",
            "properties": {
                "operation": {"type": "string"},
                "columns": {"type": "array", "items": {"type": "string"}},
                "metrics": {"type": "array", "items": {"type": "string"}},
                "filters": {"type": "array"},
                "sort_by": {"type": "array"},
                "sort_order": {"type": "string"},
                "limit": {"type": "integer"},
                "output_format": {"type": "string"},
                "implementation_plan": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {"type": "integer"},
                            "action": {"type": "string"},
                            "details": {"type": "string"}
                        }
                    }
                },
                "expected_output": {"type": "object"},
                "edge_cases": {"type": "array"},
                "validation_rules": {"type": "array"}
            },
            "required": ["operation", "columns", "implementation_plan"]
        }
        
        # Use Qwen LLM for detailed extraction
        return self.llm.generate_structured(prompt, schema)
    
    def _build_prompt(self, query: str, data_path: str, columns: list, 
                     dtypes: dict, sample_values: dict) -> str:
        """Build comprehensive prompt for intent extraction.
        
        Args:
            query: User query
            data_path: Dataset path
            columns: List of column names
            dtypes: Dictionary of column data types
            sample_values: Sample values for each column
            
        Returns:
            Formatted prompt string
        """
        # Try to load template, fallback to inline if not found
        if self.prompt_template_path.exists():
            with open(self.prompt_template_path) as f:
                template = f.read()
            return template.format(
                query=query,
                data_path=data_path,
                columns=columns,
                dtypes=dtypes,
                sample_values=sample_values
            )
        
        # Fallback inline prompt
        return f"""Analyze this data analysis request and create a detailed implementation plan.

USER QUERY: {query}
DATASET: {data_path}
AVAILABLE COLUMNS: {columns}
COLUMN TYPES: {dtypes}
SAMPLE VALUES: {sample_values}

Provide a thorough analysis with:

1. INTENT BREAKDOWN:
   - Primary operation (groupby, filter, aggregate, join, pivot, transform, etc.)
   - Required columns and their roles
   - Metrics/calculations needed (count, sum, mean, median, std, etc.)
   - Filter conditions (if any)
   - Sorting requirements
   - Output format (table, chart, summary, json)

2. IMPLEMENTATION PLAN (Step-by-step todo list):
   - List each discrete step needed to accomplish the task
   - Include data loading, transformations, calculations, formatting
   - Specify order of operations
   - Note any edge cases or validations needed

3. EXPECTED OUTPUT:
   - Describe what the final result should look like
   - Include column names, data types, format

Return JSON with this structure:
{{
  "operation": "groupby_aggregate",
  "columns": ["col1", "col2"],
  "metrics": ["count", "mean"],
  "filters": [{{"column": "date", "operator": ">", "value": "2023-01-01"}}],
  "sort_by": ["count"],
  "sort_order": "descending",
  "limit": 10,
  "output_format": "table",
  "implementation_plan": [
    {{"step": 1, "action": "Load CSV file", "details": "Read data"}},
    {{"step": 2, "action": "Validate columns", "details": "Check required columns exist"}}
  ],
  "expected_output": {{
    "columns": ["col1", "col2", "count"],
    "format": "markdown_table"
  }},
  "edge_cases": ["empty dataset", "missing columns"],
  "validation_rules": ["columns must exist", "numeric columns for aggregations"]
}}
"""


# ============================================================================
# Gap Detector
# ============================================================================

class GapDetector:
    """Detect if a new tool is needed or existing tool can handle the request."""
    
    def __init__(self, registry_path: str = "registry/tools.json"):
        """Initialize gap detector.
        
        Args:
            registry_path: Path to tools registry
        """
        self.registry_path = Path(registry_path)
    
    def detect(self, intent: Dict[str, Any]) -> bool:
        """Detect if there's a capability gap requiring a new tool.
        
        Args:
            intent: Extracted intent dictionary
            
        Returns:
            True if new tool is needed, False if existing tool can handle it
        """
        existing_tools = self._load_registry()
        
        if not existing_tools:
            # No tools in registry, always need new tool
            return True
        
        # Calculate overlap with each existing tool
        overlap_scores = [
            self._calculate_overlap(intent, tool) 
            for tool in existing_tools.values()
        ]
        
        # If best overlap is < 85%, we need a new tool
        max_overlap = max(overlap_scores, default=0)
        return max_overlap < 0.85
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load tools from registry.
        
        Returns:
            Dictionary of registered tools
        """
        if not self.registry_path.exists():
            return {}
        
        try:
            with open(self.registry_path) as f:
                registry = json.load(f)
            return registry.get("tools", {})
        except Exception:
            return {}
    
    def _calculate_overlap(self, intent: Dict, tool: Dict) -> float:
        """Calculate overlap score between intent and existing tool.
        
        Args:
            intent: Extracted intent
            tool: Existing tool metadata
            
        Returns:
            Overlap score between 0.0 and 1.0
        """
        score = 0.0
        weights = {"operation": 0.4, "columns": 0.3, "metrics": 0.3}
        
        # Compare operation
        if intent.get("operation") == tool.get("operation"):
            score += weights["operation"]
        
        # Compare columns (set intersection)
        intent_cols = set(intent.get("columns", []))
        tool_cols = set(tool.get("columns", []))
        if intent_cols and tool_cols:
            col_overlap = len(intent_cols & tool_cols) / len(intent_cols | tool_cols)
            score += weights["columns"] * col_overlap
        
        # Compare metrics (set intersection)
        intent_metrics = set(intent.get("metrics", []))
        tool_metrics = set(tool.get("metrics", []))
        if intent_metrics and tool_metrics:
            metric_overlap = len(intent_metrics & tool_metrics) / len(intent_metrics | tool_metrics)
            score += weights["metrics"] * metric_overlap
        
        return score


# ============================================================================
# Helper Functions
# ============================================================================

def extract_intent(query: str, data_path: str, llm_client: Optional[QwenLLMClient] = None) -> Dict:
    """Extract intent from user query.
    
    Args:
        query: Natural language query
        data_path: Path to dataset
        llm_client: Optional LLM client (creates new one if None)
        
    Returns:
        Extracted intent dictionary
    """
    if llm_client is None:
        from src.llm_client import create_llm_client
        llm_client = create_llm_client()
    
    extractor = IntentExtractor(llm_client)
    return extractor.extract(query, data_path)


def detect_capability_gap(intent: Dict) -> bool:
    """Detect if new tool is needed.
    
    Args:
        intent: Extracted intent dictionary
        
    Returns:
        True if new tool needed, False if existing tool can handle it
    """
    detector = GapDetector()
    return detector.detect(intent)


# ============================================================================
# LangGraph Nodes
# ============================================================================

def intent_node(state: ToolGeneratorState) -> ToolGeneratorState:
    """LangGraph node: Extract intent and detect capability gap.
    
    Args:
        state: Current generator state
        
    Returns:
        Updated state with extracted_intent and has_gap
    """
    from src.llm_client import create_llm_client
    
    llm_client = create_llm_client()
    intent = extract_intent(state["user_query"], state["data_path"], llm_client)
    gap_detected = detect_capability_gap(intent)
    
    return {
        **state,
        "extracted_intent": intent,
        "has_gap": gap_detected
    }


def route_after_intent(state: ToolGeneratorState) -> str:
    """Route after intent extraction.
    
    Args:
        state: Current generator state
        
    Returns:
        Next node name
    """
    return "spec_generator_node" if state["has_gap"] else "executor_node"
