"""Intent Extraction Module - Extract structured intent from natural language queries."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from difflib import SequenceMatcher
from src.models import ToolGeneratorState
from src.llm_client import QwenLLMClient
from src.logger_config import get_logger, log_section

logger = get_logger(__name__)


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
        self.prompt_template_path = Path("config/prompts/intent_extraction_v2.txt")
    
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
        
        # Log available columns
        log_section(logger, "INTENT EXTRACTION - AVAILABLE COLUMNS")
        logger.debug(f"Columns: {columns}")
        
        # Build comprehensive analysis prompt
        prompt = self._build_prompt(query, data_path, columns, dtypes, sample_values)
        
        # Define expected JSON schema
        schema = {
            "type": "object",
            "properties": {
                "has_gap": {"type": "boolean"},
                "gap_reason": {"type": "string"},
                "operation": {"type": "string"},
                "required_columns": {"type": "array", "items": {"type": "string"}},
                "missing_columns": {"type": "array", "items": {"type": "string"}},
                "filters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string"},
                            "operator": {"type": "string"},
                            "value": {}
                        }
                    }
                },
                "group_by": {"type": "array", "items": {"type": "string"}},
                "metrics": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "column": {"type": ["string", "null"]},
                            "alias": {"type": ["string", "null"]}
                        }
                    }
                },
                "sort_by": {"type": "array", "items": {"type": "string"}},
                "sort_order": {"type": "string"},
                "limit": {"type": ["integer", "null"]},
                "output_format": {"type": "string"},
                "implementation_plan": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {"type": "integer"},
                            "action": {"type": "string"},
                            "details": {"type": "string"},
                            "validations": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "edge_cases": {"type": "array", "items": {"type": "string"}},
                "validation_rules": {"type": "array", "items": {"type": "string"}},
                "assumptions": {"type": "array", "items": {"type": "string"}},
                "clarifications_needed": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["has_gap", "operation", "required_columns", "implementation_plan"]
        }
        
        # System message to enforce following the user's requested analysis
        system_message = """You are an intent extraction system. Your job is to understand the user's query and create an implementation plan for the EXACT analysis they requested.

CRITICAL RULES:
1. If the user asks for "ANOVA", your implementation_plan MUST describe ANOVA analysis (scipy.stats.f_oneway), NOT correlation, chi-square, or any other test
2. If the user asks for "correlation", your implementation_plan MUST describe correlation analysis (scipy.stats.pearsonr), NOT ANOVA
3. If the user asks for "Tukey HSD", your plan MUST include statsmodels.stats.multicomp.pairwise_tukeyhsd
4. DO NOT substitute different statistical methods than what the user explicitly requested
5. Output ONLY valid JSON - no thinking process, no explanations, no markdown
6. The 'operation' field is REQUIRED - never return null/None for it"""
        
        # Use Qwen LLM for detailed extraction
        intent = self.llm.generate_structured(prompt, schema, system_message=system_message)
        
        # CRITICAL: Validate and ground column names to prevent hallucination
        required_cols = intent.get('required_columns', [])
        missing_cols = intent.get('missing_columns', [])
        
        hallucinated_cols = [col for col in required_cols if col not in columns]
        if hallucinated_cols:
            log_section(logger, "COLUMN GROUNDING - Resolving hallucinated columns")
            logger.warning(f"Hallucinated columns: {hallucinated_cols}")
            logger.info(f"Available columns: {columns}")
            logger.info("Attempting fuzzy matching...")
            
            # Try to ground hallucinated columns to actual columns
            grounded, unresolved = self._ground_columns(query, columns, hallucinated_cols)
            
            # Keep non-hallucinated columns that were correct
            valid_cols = [col for col in required_cols if col in columns]
            
            # Add successfully grounded columns
            intent['required_columns'] = valid_cols + grounded
            
            # Add unresolved to missing_columns
            for col in unresolved:
                if col not in missing_cols:
                    missing_cols.append(col)
            intent['missing_columns'] = missing_cols
            
            logger.info("GROUNDING RESULTS:")
            logger.info(f"  Valid columns (already existed): {valid_cols}")
            logger.info(f"  Grounded columns (fuzzy matched): {grounded}")
            logger.info(f"  Unresolved columns (moved to missing): {unresolved}")
        
        # Log extracted intent
        log_section(logger, "EXTRACTED INTENT")
        logger.info(f"Required columns: {intent.get('required_columns', [])}")
        logger.info(f"Missing columns: {intent.get('missing_columns', [])}")
        logger.info(f"Operation: {intent.get('operation')}")
        
        return intent
    
    def _ground_columns(self, query: str, available_columns: List[str], 
                       required_columns: List[str]) -> Tuple[List[str], List[str]]:
        """Ground hallucinated column names to actual dataset columns using fuzzy matching.
        
        Args:
            query: User's natural language query
            available_columns: Actual columns in the dataset
            required_columns: Columns extracted by LLM (may be hallucinated)
            
        Returns:
            Tuple of (grounded_columns, unresolved_columns)
        """
        grounded = []
        unresolved = []
        
        # Common concept mappings for traffic accident data
        concept_mappings = {
            'accident_type': ['crash_type', 'first_crash_type', 'trafficway_type'],
            'crash_type': ['crash_type', 'first_crash_type'],
            'type': ['crash_type', 'first_crash_type', 'trafficway_type'],
            'severity': ['most_severe_injury', 'damage', 'injuries_fatal'],
            'injury': ['most_severe_injury', 'injuries_total', 'injuries_fatal'],
            'date': ['crash_date'],
            'time': ['crash_hour', 'crash_date'],
            'year': ['crash_date', 'crash_month'],
            'month': ['crash_month', 'crash_date'],
            'day': ['crash_day_of_week', 'crash_date'],
            'hour': ['crash_hour'],
            'weather': ['weather_condition'],
            'location': ['alignment', 'trafficway_type'],
            'cause': ['prim_contributory_cause'],
            'control': ['traffic_control_device'],
            'lighting': ['lighting_condition'],
            'surface': ['roadway_surface_cond'],
            'defect': ['road_defect']
        }
        
        for req_col in required_columns:
            # Check if column exists exactly
            if req_col in available_columns:
                grounded.append(req_col)
                continue
            
            # Try concept mapping first
            req_col_lower = req_col.lower()
            candidates = []
            for concept, col_list in concept_mappings.items():
                if concept in req_col_lower or req_col_lower in concept:
                    candidates.extend([c for c in col_list if c in available_columns])
            
            # If no concept match, do fuzzy matching
            if not candidates:
                # Calculate similarity scores
                similarities = []
                for avail_col in available_columns:
                    # Token-based matching (split on underscores)
                    req_tokens = set(req_col_lower.split('_'))
                    avail_tokens = set(avail_col.lower().split('_'))
                    
                    # Jaccard similarity (token overlap)
                    intersection = len(req_tokens & avail_tokens)
                    union = len(req_tokens | avail_tokens)
                    token_score = intersection / union if union > 0 else 0
                    
                    # Levenshtein-like similarity (entire string)
                    string_score = SequenceMatcher(None, req_col_lower, avail_col.lower()).ratio()
                    
                    # Combined score (weighted)
                    combined_score = 0.6 * token_score + 0.4 * string_score
                    similarities.append((avail_col, combined_score))
                
                # Get best match above threshold
                similarities.sort(key=lambda x: x[1], reverse=True)
                if similarities and similarities[0][1] >= 0.3:
                    candidates = [similarities[0][0]]
            
            # Use first candidate if found
            if candidates:
                best_match = candidates[0]
                print(f"  ✓ Grounded '{req_col}' → '{best_match}'")
                grounded.append(best_match)
            else:
                print(f"  ✗ Could not ground '{req_col}' (no match in dataset)")
                unresolved.append(req_col)
        
        return grounded, unresolved
    
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
        # Check if query contains statistical keywords - use simpler focused prompt
        statistical_keywords = [
            'anova', 'f-test', 'f-statistic', 'analysis of variance',
            't-test', 'tukey', 'bonferroni', 'post-hoc', 'multiple comparison',
            'chi-square', 'chi2', 'pearson', 'spearman', 'correlation',
            'regression', 'effect size', "cohen's d", 'eta-squared'
        ]
        
        query_lower = query.lower()
        has_statistical_keywords = any(keyword in query_lower for keyword in statistical_keywords)
        
        if has_statistical_keywords:
            # Use simpler ANOVA-focused prompt
            anova_prompt_path = Path("config/prompts/intent_extraction_anova.txt")
            if anova_prompt_path.exists():
                with open(anova_prompt_path, encoding='utf-8') as f:
                    template = f.read()
                return template.format(
                    query=query,
                    columns=columns
                )
        
        # Try to load standard template
        if self.prompt_template_path.exists():
            with open(self.prompt_template_path, encoding='utf-8') as f:
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
        # Note: intent uses "required_columns" in new schema
        intent_cols = set(intent.get("required_columns", intent.get("columns", [])))
        tool_cols = set(tool.get("required_columns", tool.get("columns", [])))
        if intent_cols and tool_cols:
            col_overlap = len(intent_cols & tool_cols) / len(intent_cols | tool_cols)
            score += weights["columns"] * col_overlap
        
        # Compare metrics (set intersection)
        # Note: metrics is now array of objects, extract names
        intent_metrics_raw = intent.get("metrics", [])
        if isinstance(intent_metrics_raw, list) and intent_metrics_raw and isinstance(intent_metrics_raw[0], dict):
            # New schema: extract metric names
            intent_metrics = set(m.get("name") for m in intent_metrics_raw if m.get("name"))
        else:
            # Old schema or simple strings
            intent_metrics = set(intent_metrics_raw) if isinstance(intent_metrics_raw, list) else set()
        
        tool_metrics_raw = tool.get("metrics", [])
        if isinstance(tool_metrics_raw, list) and tool_metrics_raw and isinstance(tool_metrics_raw[0], dict):
            tool_metrics = set(m.get("name") for m in tool_metrics_raw if m.get("name"))
        else:
            tool_metrics = set(tool_metrics_raw) if isinstance(tool_metrics_raw, list) else set()
        
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
    
    # Use reasoning model for intent extraction and planning
    llm_client = create_llm_client(model_type="reasoning")
    intent = extract_intent(state["user_query"], state["data_path"], llm_client)
    gap_detected = detect_capability_gap(intent)
    
    return {
        **state,
        "extracted_intent": intent,
        "has_gap": gap_detected
    }


def route_after_intent(state: ToolGeneratorState) -> str:
    """Route after intent extraction with validation gates.
    
    Args:
        state: Current generator state
        
    Returns:
        Next node name
    """
    from langgraph.graph import END
    
    # HARD STOP GATE: Check if columns are properly grounded
    required_cols = state.get("required_columns", [])
    missing_cols = state.get("missing_columns", [])
    operation = state.get("operation", "")
    
    # Gate 1: For groupby/aggregation operations, must have required columns
    groupby_operations = ["groupby_aggregate", "group_by", "pivot", "time_series_aggregate"]
    if operation in groupby_operations and len(required_cols) == 0:
        print("\n" + "🛑"*40)
        print("ROUTING GATE: BLOCKED")
        print(f"Operation '{operation}' requires columns, but required_columns is empty")
        print("This indicates column grounding failed")
        print("🛑"*40 + "\n")
        state["errors"] = state.get("errors", []) + [
            "Column grounding failed: no valid columns found for groupby operation"
        ]
        return END  # Stop pipeline - needs clarification
    
    # Gate 2: If critical columns are missing and not resolved, stop
    if len(missing_cols) > 0 and len(required_cols) == 0:
        print("\n" + "🛑"*40)
        print("ROUTING GATE: BLOCKED")
        print(f"All required columns are missing: {missing_cols}")
        print("Cannot generate tool without any valid columns")
        print("🛑"*40 + "\n")
        state["errors"] = state.get("errors", []) + [
            f"Cannot ground columns: {missing_cols} not found in dataset"
        ]
        return END  # Stop pipeline - needs clarification
    
    # Gate 3: Warn if partial resolution (some columns grounded, some missing)
    if len(missing_cols) > 0:
        print(f"\n⚠️ WARNING: Proceeding with partial column resolution")
        print(f"   Grounded: {required_cols}")
        print(f"   Missing: {missing_cols}\n")
    
    # Proceed to spec generation if gates passed
    return "spec_generator_node" if not state["has_gap"] else "spec_generator_node"
