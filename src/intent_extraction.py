"""Intent Extraction Module - Extract structured intent from natural language queries."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
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
            logger.info("Attempting LLM-based grounding...")
            
            # Try to ground hallucinated columns to actual columns
            grounded, unresolved = self._ground_columns(query, columns, hallucinated_cols,
                                                        dtypes=dtypes,
                                                        sample_values=sample_values)
            
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
            logger.info(f"  Grounded columns (LLM matched): {grounded}")
            logger.info(f"  Unresolved columns (moved to missing): {unresolved}")
        
        # GROUPBY COLUMN FALLBACK: If groupby_aggregate has only 1 column (metric only),
        # the user said "by group" without specifying which column.
        # Pick the first categorical column as a default grouping column.
        operation = intent.get('operation', '')
        required_cols = intent.get('required_columns', [])
        if operation == 'groupby_aggregate' and len(required_cols) < 2:
            categorical_cols = [col for col, dtype in dtypes.items()
                                if dtype == 'object' and col not in required_cols
                                and col != 'crash_date']
            if categorical_cols:
                default_group_col = categorical_cols[0]
                required_cols = [default_group_col] + required_cols
                intent['required_columns'] = required_cols
                group_by = intent.get('group_by', [])
                if default_group_col not in group_by:
                    group_by.insert(0, default_group_col)
                intent['group_by'] = group_by
                clarifications = intent.get('clarifications_needed', [])
                clarifications.append(
                    f"Grouping column defaulted to '{default_group_col}' because the query "
                    f"did not specify which column to group by. Specify the grouping column for precise results."
                )
                intent['clarifications_needed'] = clarifications
                logger.info(f"GROUPBY FALLBACK: added default grouping column '{default_group_col}'")

        # Log extracted intent
        log_section(logger, "EXTRACTED INTENT")
        logger.info(f"Required columns: {intent.get('required_columns', [])}")
        logger.info(f"Missing columns: {intent.get('missing_columns', [])}")
        logger.info(f"Operation: {intent.get('operation')}")
        
        return intent
    
    def _ground_columns(self, query: str, available_columns: List[str],
                       required_columns: List[str],
                       dtypes: Dict[str, str] = None,
                       sample_values: Dict[str, list] = None) -> Tuple[List[str], List[str]]:
        """Ground hallucinated column names to actual dataset columns using the LLM.

        The LLM receives the user query, all available columns with their dtypes
        and sample values, and each hallucinated name — and picks the semantically
        correct real column (or NONE).  This prevents numeric/text confusion that
        pure string-similarity matching causes (e.g. mapping 'injury_count' →
        'most_severe_injury' instead of 'injuries_total').

        Args:
            query: User's natural language query
            available_columns: Actual columns in the dataset
            required_columns: Hallucinated column names to resolve
            dtypes: Optional column dtype map
            sample_values: Optional sample values per column

        Returns:
            Tuple of (grounded_columns, unresolved_columns)
        """
        grounded = []
        unresolved = []

        # Build a concise schema description for the LLM
        schema_lines = []
        for col in available_columns:
            dtype = dtypes.get(col, "unknown") if dtypes else "unknown"
            samples = sample_values.get(col, [])[:3] if sample_values else []
            schema_lines.append(f"  - {col}  (dtype: {dtype},  samples: {samples})")
        schema_text = "\n".join(schema_lines)

        for req_col in required_columns:
            prompt = f"""Dataset columns and their dtypes/samples:
{schema_text}

User query: "{query}"
Hallucinated column name that does NOT exist: "{req_col}"

Pick the single best matching column from the dataset for "{req_col}".
- Prefer numeric (int64/float64) columns if the name implies a count or total.
- Prefer object columns if the name implies a category or label.
- Reply with ONLY the exact column name from the list above, nothing else.
- If nothing matches, reply: NONE"""

            try:
                # Use generate (not generate_structured) so the reasoning model
                # can use its <think> block freely — we scan the full raw response
                # (thinking + answer) for any real column name.
                raw_response = self.llm.generate(prompt, temperature=0.0)

                # 1. Try the final answer line first (after </think> if present)
                if "</think>" in raw_response:
                    answer_part = raw_response.split("</think>", 1)[1].strip()
                else:
                    answer_part = raw_response.strip()

                cleaned = answer_part.strip('`\'".,: \n').split('\n')[0].strip()

                if cleaned and cleaned != "NONE" and cleaned in available_columns:
                    print(f"  [OK] Grounded '{req_col}' -> '{cleaned}'")
                    grounded.append(cleaned)
                else:
                    # 2. Scan ENTIRE raw response (incl. <think> reasoning) for any column name
                    matched = next((col for col in available_columns if col in raw_response), None)
                    if matched:
                        print(f"  [OK] Grounded '{req_col}' -> '{matched}' (found in reasoning)")
                        grounded.append(matched)
                    else:
                        print(f"  [FAIL] LLM could not ground '{req_col}' (replied: '{cleaned}')")
                        unresolved.append(req_col)
            except Exception as e:
                logger.warning(f"LLM grounding failed for '{req_col}': {e}")
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
    
    def __init__(self, registry_path: str = "tools/registry.json"):
        """Initialize gap detector.
        
        Args:
            registry_path: Path to tools registry (default: tools/registry.json written by ToolPromoter)
        """
        self.registry_path = Path(registry_path)
    
    def detect(self, intent: Dict[str, Any], user_query: str = "") -> tuple:
        """Detect if there's a capability gap requiring a new tool.
        
        Args:
            intent: Extracted intent dictionary
            user_query: Original natural language query (used for similarity)
            
        Returns:
            Tuple of (has_gap: bool, best_match: Optional[Dict]).
            has_gap is True if a new tool is needed.
            best_match is the registry entry with the highest overlap score
            (or None when has_gap is True meaning no adequate match was found).
        """
        existing_tools = self._load_registry()
        
        if not existing_tools:
            # No tools in registry, always need new tool
            return True, None
        
        # Find the best-matching tool and its score
        best_tool = None
        best_score = 0.0
        for tool in existing_tools.values():
            score = self._calculate_overlap(intent, tool, user_query)
            if score > best_score:
                best_score = score
                best_tool = tool
        
        threshold = 0.5
        logger.info(f"🔍 Gap detection: best registry match score = {best_score:.3f} (threshold {threshold})")
        has_gap = best_score < threshold
        return has_gap, (best_tool if not has_gap else None)
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load tools from registry.

        tools/registry.json stores tools as a LIST of dicts (written by ToolPromoter).
        Normalise to a name-keyed dict so the rest of the logic is uniform.

        Only returns tools whose active file actually exists on disk — draft tools
        and stale entries (file deleted after promotion) are excluded.

        Returns:
            Dict mapping tool name -> tool metadata
        """
        if not self.registry_path.exists():
            return {}

        try:
            with open(self.registry_path) as f:
                registry = json.load(f)
            tools_raw = registry.get("tools", {})

            # Normalise to list
            if isinstance(tools_raw, list):
                tools_list = [t for t in tools_raw if isinstance(t, dict) and "name" in t]
            elif isinstance(tools_raw, dict):
                tools_list = list(tools_raw.values())
            else:
                return {}

            # Filter: keep only tools whose active file exists and is NOT in draft/
            valid = {}
            for t in tools_list:
                tool_path = t.get("tool_path") or t.get("path", "")
                if not tool_path:
                    continue
                p = Path(tool_path)
                # Must exist on disk and must be under an active directory (not draft)
                if p.exists() and "draft" not in p.parts:
                    valid[t["name"]] = t

            return valid
        except Exception:
            return {}
    
    def _calculate_overlap(self, intent: Dict, tool: Dict, user_query: str = "") -> float:
        """Calculate overlap score between intent and existing tool.

        Scoring:
          - user_query semantic similarity (simple word-overlap): 0.5 weight
          - required_columns Jaccard similarity:                  0.3 weight
          - operation match:                                      0.2 weight

        Args:
            intent: Extracted intent
            tool: Existing tool metadata (from tools/registry.json)
            user_query: Original natural language query

        Returns:
            Overlap score between 0.0 and 1.0
        """
        score = 0.0

        # --- 1. Operation match (0.2) ---
        intent_op = intent.get("operation", "")
        tool_op = tool.get("operation", "")
        # registry entries may not store 'operation' — fall back gracefully
        if intent_op and tool_op and intent_op == tool_op:
            score += 0.2
        elif not tool_op:
            # Old registry entries don't record operation — don't penalise
            score += 0.1

        # --- 2. Required columns Jaccard (0.3) ---
        intent_cols = set(intent.get("required_columns", intent.get("columns", [])))
        tool_cols = set(tool.get("required_columns", tool.get("columns", [])))
        if not tool_cols:
            # Try to infer from tool name tokens (e.g. weather_injuries_correlation)
            tool_name_tokens = set(tool.get("name", "").lower().split("_"))
            tool_cols = {c for c in intent_cols if c.lower() in tool_name_tokens}
        if intent_cols and tool_cols:
            col_overlap = len(intent_cols & tool_cols) / len(intent_cols | tool_cols)
            score += 0.3 * col_overlap

        # --- 3. User-query word overlap (0.5) ---
        tool_query = tool.get("user_query", "").lower()
        if user_query and tool_query:
            stop = {"the", "a", "an", "of", "in", "and", "or", "by", "for",
                    "with", "to", "from", "is", "are", "was", "were", "be"}
            iw = set(user_query.lower().split()) - stop
            tw = set(tool_query.split()) - stop
            if iw and tw:
                query_overlap = len(iw & tw) / len(iw | tw)
                score += 0.5 * query_overlap

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
        llm_client = create_llm_client(model_type="reasoning")
    
    extractor = IntentExtractor(llm_client)
    return extractor.extract(query, data_path)


def detect_capability_gap(intent: Dict, user_query: str = "") -> tuple:
    """Detect if new tool is needed.
    
    Args:
        intent: Extracted intent dictionary
        user_query: Original natural language query (used for similarity matching)
        
    Returns:
        Tuple of (has_gap: bool, best_match: Optional[Dict]).
        has_gap is True if a new tool is needed.
        best_match is the registry entry that best covers the request (None when has_gap is True).
    """
    detector = GapDetector()
    return detector.detect(intent, user_query=user_query)


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
    from src.intent_validator import validate_intent, log_validation_results
    import pandas as pd
    
    # Use reasoning model for intent extraction and planning
    llm_client = create_llm_client(model_type="reasoning")
    intent = extract_intent(state["user_query"], state["data_path"], llm_client)
    gap_detected, best_match = detect_capability_gap(intent, user_query=state["user_query"])
    
    # Validate intent before proceeding
    # Load dataset to get available columns
    try:
        df = pd.read_csv(state["data_path"])
        available_columns = df.columns.tolist()
        
        is_valid, errors, warnings = validate_intent(intent, available_columns)
        log_validation_results(is_valid, errors, warnings)
        
        # Store validation results in intent
        intent["validation"] = {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings
        }
        
        if not is_valid:
            logger.error("❌ Intent validation failed - cannot proceed to code generation")
            logger.error(f"Errors: {errors}")
            # Still return the intent but mark it as invalid
            # The routing logic will handle this
    except Exception as e:
        logger.warning(f"⚠️ Intent validation skipped: {e}")
        intent["validation"] = {
            "is_valid": True,  # Assume valid if validation fails
            "errors": [],
            "warnings": [f"Validation skipped: {str(e)}"]
        }
    
    return {
        **state,
        "extracted_intent": intent,
        "has_gap": gap_detected,
        "matched_tool": best_match
    }


def route_after_intent(state: ToolGeneratorState) -> str:
    """Route after intent extraction with validation gates.
    
    Args:
        state: Current generator state
        
    Returns:
        Next node name
    """
    from langgraph.graph import END
    
    # NEW: Check intent validation results
    intent = state.get("extracted_intent", {})
    validation = intent.get("validation", {})
    
    if not validation.get("is_valid", True):
        logger.error("❌ Intent validation failed - stopping pipeline")
        errors = validation.get("errors", [])
        for i, err in enumerate(errors, 1):
            logger.error(f"  {i}. {err}")
        logger.info("💡 Suggestion: Review the query and ensure it maps to available dataset columns")
        return END
    
    # Log warnings if any
    warnings = validation.get("warnings", [])
    if warnings:
        logger.warning("⚠️ Intent validation warnings:")
        for i, warn in enumerate(warnings, 1):
            logger.warning(f"  {i}. {warn}")
    
    # LEGACY GATES (kept for backward compatibility but mostly redundant now)
    required_cols = intent.get("required_columns", [])
    missing_cols = intent.get("missing_columns", [])
    operation = intent.get("operation", "")
    
    # Gate 1: For groupby/aggregation operations, must have required columns
    groupby_operations = ["groupby_aggregate", "group_by", "pivot", "time_series_aggregate"]
    if operation in groupby_operations and len(required_cols) == 0:
        print("\n" + "[BLOCKED]"*10)
        print("ROUTING GATE: BLOCKED")
        print(f"Operation '{operation}' requires columns, but required_columns is empty")
        print("This indicates column grounding failed")
        print("[BLOCKED]"*10 + "\n")
        state["errors"] = state.get("errors", []) + [
            "Column grounding failed: no valid columns found for groupby operation"
        ]
        return END  # Stop pipeline - needs clarification
    
    # Gate 2: If critical columns are missing and not resolved, stop
    if len(missing_cols) > 0 and len(required_cols) == 0:
        print("\n" + "[BLOCKED]"*10)
        print("ROUTING GATE: BLOCKED")
        print(f"All required columns are missing: {missing_cols}")
        print("Cannot generate tool without any valid columns")
        print("[BLOCKED]"*10 + "\n")
        state["errors"] = state.get("errors", []) + [
            f"Cannot ground columns: {missing_cols} not found in dataset"
        ]
        return END  # Stop pipeline - needs clarification
    
    # Gate 3: Warn if partial resolution (some columns grounded, some missing)
    if len(missing_cols) > 0:
        print(f"\n[WARN] Proceeding with partial column resolution")
        print(f"   Grounded: {required_cols}")
        print(f"   Missing: {missing_cols}\n")
    
    # Route based on gap detection result
    # has_gap=True  → a new tool is needed → generate spec → code → ...
    # has_gap=False → an existing tool already covers this request → done
    if not state["has_gap"]:
        logger.info("✅ Existing tool covers this request — skipping generation")
        matched = state.get("matched_tool") or {}
        tool_name = matched.get("name", "unknown")
        tool_path = matched.get("tool_path", "")
        output_file = matched.get("output_file", "")
        logger.info(f"   📦 Matched tool : {tool_name}")
        if tool_path:
            logger.info(f"   📄 Tool path    : {tool_path}")
        if output_file:
            logger.info(f"   📊 Last output  : {output_file}")
        return END
    return "spec_generator_node"
