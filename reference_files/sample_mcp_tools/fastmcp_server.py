"""
CODING STANDARDS FOR THIS FILE:
================================
1. NEVER USE UNICODE EMOJIS OR SPECIAL CHARACTERS
   - Windows console (cp1252 codec) cannot display Unicode emojis
   - This causes UnicodeEncodeError and server crashes
   - All emojis have been completely removed from this file

2. STICK TO ASCII TEXT ONLY:
   - Use plain text descriptions instead of emojis
   - Use status indicators like "OK", "ERROR", "WARNING"
   - Use ASCII symbols if needed: *, +, -, =, |, etc.

3. ALWAYS TEST ON WINDOWS CONSOLE BEFORE DEPLOYMENT
================================
"""

# Essential imports (optimized)
import gc
import hashlib
import json
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Annotated, List, Literal, Union, Dict, Any, Tuple

# Optimized matplotlib configuration
import matplotlib
import numpy as np
import pandas as pd
from pydantic import Field

matplotlib.use("Agg")  # Use non-interactive backend for better performance
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# Data science imports
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# PDF generation imports
try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print(
        "Warning: ReportLab not available. PDF generation will not work. Install with: pip install reportlab"
    )

# Initialize FastMCP at the top level
from fastmcp import FastMCP

mcp = FastMCP("AdvancedAnalysis")


# Global cache for datasets (OPTIMIZATION)
_dataset_cache = {}


def get_file_hash(file_path: str) -> str:
    """Get hash of file for caching"""
    try:
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return str(hash(file_path))  # Fallback to path hash


def validate_file_and_columns(
    file_path: str, required_columns: list = None, numeric_columns: list = None
) -> tuple:
    """OPTIMIZED: Centralized validation function with caching"""
    if not Path(file_path).exists():
        return False, f"File '{file_path}' does not exist"

    try:
        # Use cached loading for validation too
        df = load_csv_data_cached(file_path)

        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                return False, f"Columns not found: {missing_cols}"

        if numeric_columns:
            non_numeric = [
                col
                for col in numeric_columns
                if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])
            ]
            if non_numeric:
                return False, f"Non-numeric columns: {non_numeric}"

        return True, df
    except Exception as e:
        return False, f"Error loading data: {str(e)}"


def fast_file_check(file_path: str) -> bool:
    """OPTIMIZED: Fast file existence check without loading data"""
    return Path(file_path).exists()


def detect_column_types(df: pd.DataFrame) -> dict:
    """Intelligently detect column types and purposes."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = [
        col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])
    ]

    # Remove datetime columns from categorical overview to avoid duplication
    categorical_cols = [col for col in categorical_cols if col not in datetime_cols]

    high_cardinality: List[str] = []
    potential_ids: List[str] = []
    potential_targets: List[str] = []

    total_rows = max(len(df), 1)
    for col in categorical_cols:
        unique_ratio = df[col].nunique(dropna=False) / total_rows
        if unique_ratio > 0.5:
            high_cardinality.append(col)
        if "id" in col.lower() or unique_ratio > 0.95:
            potential_ids.append(col)
        if df[col].nunique() <= 10 and any(
            keyword in col.lower()
            for keyword in ("target", "class", "label", "category", "type")
        ):
            potential_targets.append(col)

    column_types: Dict[str, str] = {}
    for col in df.columns:
        if col in datetime_cols:
            column_types[col] = "datetime"
        elif col in numeric_cols:
            column_types[col] = "numeric"
        elif col in categorical_cols:
            column_types[col] = "categorical"
        else:
            column_types[col] = str(df[col].dtype)

    column_types["_metadata"] = {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
        "high_cardinality": high_cardinality,
        "potential_ids": potential_ids,
        "potential_targets": potential_targets,
    }

    return column_types


def is_meaningful_correlation(col1: str, col2: str) -> bool:
    """
    Filter out semantically meaningless correlations between columns.
    
    Args:
        col1, col2: Column names to check
        
    Returns:
        False if correlation is meaningless (should be skipped), True otherwise
    """
    col1_lower = col1.lower()
    col2_lower = col2.lower()
    
    # Skip correlation between coordinate pairs (lat/lon)
    coordinate_terms = ['lat', 'latitude', 'lon', 'longitude', 'lng', 'coord']
    is_col1_coordinate = any(term in col1_lower for term in coordinate_terms)
    is_col2_coordinate = any(term in col2_lower for term in coordinate_terms)
    
    if is_col1_coordinate and is_col2_coordinate:
        return False
    
    # Skip correlations with ID or index-like columns
    id_terms = ['id', 'index', 'idx', 'key', 'pk', 'uuid']
    is_col1_id = any(term in col1_lower for term in id_terms)
    is_col2_id = any(term in col2_lower for term in id_terms)
    
    if is_col1_id or is_col2_id:
        return False
    
    # Skip self-correlation (should be 1.0 anyway)
    if col1 == col2:
        return False
        
    return True


def filter_correlation_matrix(corr_matrix) -> str:
    """
    Filter correlation matrix to show only meaningful correlations.
    
    Args:
        corr_matrix: Pandas correlation matrix
        
    Returns:
        Formatted string with meaningful correlations only
    """
    meaningful_correlations = []
    processed_pairs = set()
    
    for col1 in corr_matrix.columns:
        for col2 in corr_matrix.columns:
            # Skip if already processed this pair
            pair = tuple(sorted([col1, col2]))
            if pair in processed_pairs:
                continue
            processed_pairs.add(pair)
            
            # Check if correlation is meaningful
            if not is_meaningful_correlation(col1, col2):
                continue
                
            corr_value = corr_matrix.loc[col1, col2]
            
            # Only show correlations with |r| >= 0.2 (avoid noise)
            if abs(corr_value) >= 0.2:
                meaningful_correlations.append(f"- {col1} ↔ {col2}: {corr_value:.3f}")
    
    if meaningful_correlations:
        return "**Meaningful Correlations (|r| ≥ 0.2):**\n" + "\n".join(meaningful_correlations)
    else:
        return "**No strong meaningful correlations found** (all |r| < 0.2 or semantically irrelevant)"


def optimize_correlation_analysis(
    df: pd.DataFrame, max_features: int = 20
) -> pd.DataFrame:
    """Optimize correlation analysis for large datasets"""
    numeric_cols = df.select_dtypes(include=["number"]).columns

    # If too many numeric columns, select most important ones
    if len(numeric_cols) > max_features:
        # Use variance as importance metric
        variances = df[numeric_cols].var().sort_values(ascending=False)
        selected_cols = variances.head(max_features).index
        return df[selected_cols].corr()

    return df[numeric_cols].corr()


def create_optimized_plot():
    """Context manager for memory-efficient plotting"""
    plt.figure(figsize=(10, 6))
    return plt


def cleanup_plot():
    """Clean up matplotlib resources efficiently"""
    # Only close the current figure, not all figures
    plt.close()
    # Optional garbage collection only when needed
    if hasattr(plt, '_pylab_helpers') and len(plt._pylab_helpers.Gcf.figs) > 5:
        gc.collect()


@lru_cache()
def load_csv_data_cached(file_path: str) -> pd.DataFrame:
    """Load CSV with caching and optimized type detection based on file path"""
    file_hash = get_file_hash(file_path)
    cache_key = f"{file_path}_{file_hash}"

    if cache_key not in _dataset_cache:
        # Load and perform expensive type detection only once
        df = load_csv_data_with_types(file_path)
        _dataset_cache[cache_key] = df

    return _dataset_cache[cache_key]


# === ROBUST JSON-IN-CELL HANDLING FUNCTIONS ===

def analyze_json_schema_from_samples(sample_data: List[Any]) -> Dict[str, Any]:
    """Analyze the schema of parsed JSON samples"""
    if not sample_data:
        return {}
    
    all_keys = set()
    common_keys = None
    
    for item in sample_data[:10]:  # Analyze first 10 samples
        if isinstance(item, dict):
            item_keys = set(item.keys())
            all_keys.update(item_keys)
            if common_keys is None:
                common_keys = item_keys
            else:
                common_keys = common_keys.intersection(item_keys)
    
    return {
        'common_keys': list(common_keys) if common_keys else [],
        'all_keys': list(all_keys),
        'schema_consistency': len(common_keys) / len(all_keys) if all_keys else 0
    }

def get_optimal_parsing_strategy(pattern_counts: Dict[str, int], dominant_pattern: str) -> str:
    """Determine the best parsing strategy based on pattern analysis"""
    if not dominant_pattern:
        return 'standard'
    
    # Return the dominant pattern as the strategy
    return dominant_pattern

def clean_malformed_json(val: str) -> str:
    """Attempt to clean malformed JSON strings"""
    if not val:
        return val
    
    # Remove extra brackets at the beginning/end
    if val.count('{') > val.count('}'):
        val = val.rstrip('{')
    if val.count('}') > val.count('{'):
        val = val.lstrip('}')
    
    # Fix common escaping issues
    val = re.sub(r'\\{2,}', '\\', val)  # Multiple backslashes
    val = re.sub(r'"{2,}', '"', val)    # Multiple quotes
    
    # Try to balance brackets
    open_braces = val.count('{')
    close_braces = val.count('}')
    if open_braces > close_braces:
        val += '}' * (open_braces - close_braces)
    elif close_braces > open_braces:
        val = '{' * (close_braces - open_braces) + val
    
    return val

def try_parse_json_multiple_strategies(val: str) -> Dict[str, Any]:
    """Try multiple strategies to parse a JSON string"""
    
    strategies = [
        ('standard', lambda x: json.loads(x)),
        ('triple_bracket', lambda x: json.loads(x[3:-3]) if x.startswith('{{{') and x.endswith('}}}') else None),
        ('escaped', lambda x: json.loads(x.replace('\\"', '"'))),
        ('double_escaped', lambda x: json.loads(x.replace('\\\\"', '"'))),
        ('single_quotes', lambda x: json.loads(x.replace("'", '"'))),
        ('array', lambda x: json.loads(x) if x.startswith('[') else None),
    ]
    
    for pattern_name, strategy in strategies:
        try:
            result = strategy(val)
            if result is not None:
                return {
                    'success': True,
                    'pattern': pattern_name,
                    'data': result
                }
        except (json.JSONDecodeError, ValueError, AttributeError):
            continue
    
    # Final attempt: try to clean and parse
    try:
        cleaned = clean_malformed_json(val)
        result = json.loads(cleaned)
        return {
            'success': True,
            'pattern': 'cleaned',
            'data': result
        }
    except:
        pass
    
    return {
        'success': False,
        'pattern': None,
        'data': None
    }

def analyze_json_patterns(sample_values: pd.Series) -> Dict[str, Any]:
    """Analyze a sample of values to determine JSON patterns"""
    
    json_patterns = {
        'standard': 0,      # {"key": "value"}
        'triple_bracket': 0, # {{{...}}}
        'escaped': 0,       # {\"key\": \"value\"}
        'array': 0,         # [{"key": "value"}]
        'malformed': 0,     # Partial or invalid JSON
        'not_json': 0       # Clearly not JSON
    }
    
    successful_parses = 0
    detected_pattern = None
    sample_parsed_data = []
    
    for val in sample_values:
        if not isinstance(val, str):
            json_patterns['not_json'] += 1
            continue
            
        val_stripped = val.strip()
        
        # Try multiple parsing strategies
        parsed_data = try_parse_json_multiple_strategies(val_stripped)
        
        if parsed_data['success']:
            successful_parses += 1
            json_patterns[parsed_data['pattern']] += 1
            sample_parsed_data.append(parsed_data['data'])
            if not detected_pattern:
                detected_pattern = parsed_data['pattern']
        else:
            if val_stripped.startswith(('{', '[')) and val_stripped.endswith(('}', ']')):
                json_patterns['malformed'] += 1
            else:
                json_patterns['not_json'] += 1
    
    total_samples = len(sample_values)
    json_confidence = successful_parses / total_samples if total_samples > 0 else 0
    
    return {
        'is_json_column': json_confidence > 0.7,  # 70% threshold
        'confidence': json_confidence,
        'pattern_counts': json_patterns,
        'dominant_pattern': detected_pattern,
        'sample_schema': analyze_json_schema_from_samples(sample_parsed_data),
        'parsing_strategy': get_optimal_parsing_strategy(json_patterns, detected_pattern)
    }

def detect_json_columns_robust(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Robustly detect JSON columns with detailed analysis of the JSON format
    Returns dict with column names as keys and metadata about the JSON format
    """
    json_columns = {}
    
    for col in df.select_dtypes(include=['object']).columns:
        sample_values = df[col].dropna().head(20)  # Larger sample
        if len(sample_values) == 0:
            continue
            
        json_analysis = analyze_json_patterns(sample_values)
        
        if json_analysis['is_json_column']:
            json_columns[col] = json_analysis
            
    return json_columns

def safe_json_loads(val):
    """Safely parse JSON with fallback to original value"""
    if pd.isna(val) or not isinstance(val, str):
        return val
    try:
        return json.loads(val)
    except:
        return val  # Return original if parsing fails

def expand_json_column_robust(df: pd.DataFrame, json_col: str, json_metadata: Dict[str, Any]) -> pd.DataFrame:
    """Expand JSON column using the optimal strategy based on detected patterns"""
    
    parsing_strategy = json_metadata['parsing_strategy']
    
    try:
        # Apply the optimal parsing strategy
        if parsing_strategy == 'standard':
            df[json_col] = df[json_col].apply(safe_json_loads)
        elif parsing_strategy == 'triple_bracket':
            df[json_col] = df[json_col].apply(lambda x: safe_json_loads(x[3:-3]) if isinstance(x, str) and x.startswith('{{{') and x.endswith('}}}') else safe_json_loads(x))
        elif parsing_strategy == 'escaped':
            df[json_col] = df[json_col].apply(lambda x: safe_json_loads(x.replace('\\"', '"')) if isinstance(x, str) else x)
        elif parsing_strategy == 'cleaned':
            df[json_col] = df[json_col].apply(lambda x: safe_json_loads(clean_malformed_json(x)) if isinstance(x, str) else x)
        else:
            # Default to standard parsing
            df[json_col] = df[json_col].apply(safe_json_loads)
        
        # Use pd.json_normalize for expansion
        valid_json_mask = df[json_col].apply(lambda x: isinstance(x, (dict, list)))
        
        if valid_json_mask.any():
            json_df = pd.json_normalize(df.loc[valid_json_mask, json_col])
            json_df.columns = [f"{json_col}_{col}" for col in json_df.columns]
            
            # Reindex to match original dataframe
            json_df = json_df.reindex(df.index)
            
            # Concatenate with original (minus JSON column)
            result_df = pd.concat([df.drop(json_col, axis=1), json_df], axis=1)
            
            success_rate = valid_json_mask.sum() / len(df)
            print(f"[INFO] Expanded JSON column '{json_col}' with {success_rate:.1%} success rate using {parsing_strategy} strategy")
            
            return result_df
        else:
            print(f"[WARN] No valid JSON found in column '{json_col}' after parsing")
            return df
            
    except Exception as e:
        print(f"[ERROR] Failed to expand JSON column '{json_col}': {e}")
        return df

# === END JSON-IN-CELL HANDLING FUNCTIONS ===


def load_csv_data_with_types(file_path: str, auto_expand_json: bool = True) -> pd.DataFrame:
    """Load and prepare CSV data with optimized type detection and JSON expansion (internal function)"""
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)

        # STEP 1: Handle JSON columns if requested
        if auto_expand_json:
            json_columns_metadata = detect_json_columns_robust(df)
            
            if json_columns_metadata:
                print(f"[INFO] Detected JSON columns with patterns: {[(col, meta['dominant_pattern']) for col, meta in json_columns_metadata.items()]}")
                
                for col, metadata in json_columns_metadata.items():
                    if metadata['confidence'] > 0.7:  # Only expand high-confidence JSON columns
                        df = expand_json_column_robust(df, col, metadata)

        # STEP 2: OPTIMIZED: Try to identify and convert common column types (only once)
        for col in df.columns:
            # Try to convert to numeric
            if df[col].dtype == "object":
                # Try numeric conversion first
                try:
                    numeric_series = pd.to_numeric(df[col], errors="coerce")
                    # If more than 50% of values are successfully converted, use numeric
                    if numeric_series.notna().sum() > len(df) * 0.5:
                        df[col] = numeric_series
                    # Otherwise try datetime conversion
                    else:
                        try:
                            df[col] = pd.to_datetime(
                                df[col],
                                errors="raise",
                            )
                        except:
                            pass  # Keep as object/string
                except:
                    pass  # Keep as object/string

        return df
    except Exception as e:
        raise ValueError(f"Error loading CSV file: {str(e)}")


# Loads a CSV from disk using pandas. Tries to convert string columns to numeric if possible. Returns a cleaned DataFrame.
def load_csv_data(file_path: str) -> pd.DataFrame:
    """DEPRECATED: Use load_csv_data_cached() for better performance"""
    return load_csv_data_with_types(file_path)


# Loads a CSV file and outputs:
# Basic metadata (rows, columns).
# Data types.
# Basic numeric stats (mean, std, min, max).
# Missing values summary.
# Returns a Markdown-formatted string report.
@mcp.tool()
def load_and_analyze_csv(
    file_path: Annotated[str, Field(description="Path to the CSV file to analyze")],
    auto_expand_json: Annotated[bool, Field(description="Automatically detect and expand JSON columns into separate columns for analysis")] = True,
) -> str:
    """
    Load a CSV file and provide a detailed summary of its structure and contents.

    WHEN TO USE THIS TOOL:
    - ALWAYS use this as the FIRST tool when starting any data analysis
    - Use when you need to understand what columns exist in the dataset
    - Use to identify data types (numeric, categorical, datetime)
    - Use to detect data quality issues (missing values, duplicates)

    WHAT THIS TOOL DOES:
    - Loads CSV and returns comprehensive dataset overview
    - Identifies all column names and their data types
    - Provides descriptive statistics for numeric columns (mean, std, min, max, quartiles)
    - Reports missing values per column with percentages
    - Detects and expands JSON columns automatically (if enabled)
    - Identifies high-cardinality categorical columns

    RETURNS: Markdown-formatted report with:
    - Dataset dimensions (rows, columns, memory usage)
    - Complete column list with detected types
    - Summary statistics for all numeric columns
    - Missing value analysis
    - Data quality warnings

    PREREQUISITES: None - this is always the first tool to use
    """
    try:
        # Validate file exists and columns if needed
        validate_file_and_columns(file_path)

        # Load the data using enhanced loader with JSON processing
        df = load_csv_data_with_types(file_path, auto_expand_json=auto_expand_json)

        # Get column type information
        column_types = detect_column_types(df)
        type_metadata = column_types.get("_metadata", {})
        types_by_column = {
            key: value for key, value in column_types.items() if key != "_metadata"
        }

        report = []
        report.append("## CSV Data Analysis")
        report.append(f"- File: {file_path}")
        report.append(f"- Total rows: {len(df):,}")
        report.append(f"- Total columns: {len(df.columns)}")
        report.append(
            f"- Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        )
        report.append(f"- Columns: {', '.join(df.columns)}")

        # Enhanced data types with auto-detected types
        report.append("\n### Data Types:")
        for col in df.columns:
            dtype = df[col].dtype
            detected_type = types_by_column.get(col, "unknown")
            report.append(f"- {col}: {dtype} (detected as: {detected_type})")

        high_cardinality_cols = type_metadata.get("high_cardinality", [])
        if high_cardinality_cols:
            report.append(
                "High-cardinality categorical columns: "
                + ", ".join(high_cardinality_cols)
            )

        # Basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            report.append("\n### Numeric Columns Summary:")
            for col in numeric_cols:
                stats = df[col].describe()
                report.append(
                    f"- {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
                    f"min={stats['min']}, max={stats['max']}, "
                    f"25%={stats['25%']:.2f}, 75%={stats['75%']:.2f}"
                )

        # Missing values with percentages
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            report.append("\n### Missing Values:")
            for col, missing_count in missing_data.items():
                if missing_count > 0:
                    missing_pct = (missing_count / len(df)) * 100
                    report.append(
                        f"- {col}: {missing_count} missing values ({missing_pct:.1f}%)"
                    )

        # Data quality summary
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            report.append("\n### Data Quality:")
            report.append(f"- Duplicate rows: {duplicates}")

        # Append a machine-readable JSON footer to help downstream argument inference
        try:
            footer_payload = {
                "path": file_path,
                # Provide both a flat summary and nested metadata for flexibility
                "numeric_columns": type_metadata.get("numeric", []),
                # Server uses 'categorical' to denote text-like columns
                "text_columns": type_metadata.get("categorical", []),
                "datetime_columns": type_metadata.get("datetime", []),
                "column_types": column_types,  # includes _metadata with lists
            }
            report.append("\n<!--output_json:" + json.dumps(footer_payload) + "-->")
        except Exception:
            # If footer generation fails, proceed without blocking
            pass

        return "\n".join(report)

    except Exception as e:
        return f"Error analyzing CSV: {str(e)}"


# Full EDA pipeline with:
# Descriptive stats.
# Correlation matrix.
# Missing/duplicate data summary.
# Optional KMeans clustering on numeric columns.
# Optional Linear Regression if valid x_column and y_column are given.
# Returns Markdown report summarizing the above.


@mcp.tool()
def perform_advanced_eda_on_csv(
    file_path: Annotated[str, Field(description="Path to the CSV file to analyze")],
    auto_detect_analysis: Annotated[bool, Field(description="Automatically detect and perform appropriate analysis based on column types - enhances analysis quality with intelligent pattern detection")] = False,
    auto_expand_json: Annotated[bool, Field(description="Automatically detect and expand JSON columns into separate columns for comprehensive analysis")] = True,
    n_clusters: Annotated[
        int,
        Field(
            description="Number of clusters for KMeans (default=min(3, len(df)//10))"
        ),
    ] = None,
    random_state: Annotated[int, Field(description="Random state for clustering")] = 0,
    n_init: Annotated[
        Union[str, int], Field(description="n_init parameter for KMeans")
    ] = "auto",
    x_column: Annotated[
        str, Field(description="Independent variable column for regression")
    ] = None,
    y_column: Annotated[
        str, Field(description="Dependent variable column for regression")
    ] = None,
) -> str:
    """
    Perform comprehensive exploratory data analysis (EDA) on a CSV dataset.

    WHEN TO USE THIS TOOL:
    - Use AFTER load_and_analyze_csv to get deeper statistical insights
    - Use when you need correlation analysis between numeric columns
    - Use when you want to identify patterns in temporal data (with auto_detect_analysis=True)
    - Use when you need categorical distribution analysis (with auto_detect_analysis=True)
    - Use for geographical analysis if lat/lon columns exist (with auto_detect_analysis=True)

    WHAT THIS TOOL DOES:
    Core analysis (always performed):
    - Descriptive statistics for all columns (numeric and categorical)
    - Correlation matrix for numeric columns with significance filtering
    - Missing data patterns and duplicate detection
    - Clustering analysis (KMeans) on numeric data
    - Regression analysis if x_column and y_column specified

    Enhanced intelligent analysis (when auto_detect_analysis=True):
    - TEMPORAL: Creates hourly/daily/weekly patterns if datetime columns found
    - CATEGORICAL: Distribution analysis for incident/event type columns  
    - GEOGRAPHICAL: Spatial analysis if latitude/longitude columns detected

    JSON handling (when auto_expand_json=True):
    - Detects columns containing JSON strings (handles various formats)
    - Automatically expands JSON into separate columns
    - Supports standard JSON, escaped JSON, and malformed JSON cleaning

    RETURNS: Markdown report with:
    - Complete descriptive statistics
    - Filtered correlation matrix (only meaningful correlations shown)
    - Data quality assessment
    - Clustering insights (if applicable)
    - Temporal/categorical/spatial patterns (if auto_detect_analysis=True)

    PREREQUISITES: Dataset must exist and be loadable
    RECOMMENDED PARAMETERS: Set auto_detect_analysis=True for comprehensive insights
    """
    try:
        if isinstance(n_init, str):
            if n_init.lower() == "auto":
                n_init = "auto"  # Keep as string
            elif n_init.isdigit():
                n_init = int(n_init)  # Convert string numbers to int
            else:
                raise ValueError("n_init must be 'auto' or an integer")
        elif isinstance(n_init, int):
            if n_init < 1:
                raise ValueError("n_init as integer must be >= 1")
        else:
            raise ValueError("n_init must be 'auto' or an integer")

        if not Path(file_path).exists():
            return f"Error: File '{file_path}' does not exist."

        # Use enhanced data loading with JSON processing
        df = load_csv_data_with_types(file_path, auto_expand_json=auto_expand_json)

        # Get column type information
        column_types = detect_column_types(df)

        eda_report = ["## Comprehensive CSV Data Analysis"]
        eda_report.append("\n### Dataset Overview:")
        eda_report.append(f"- File: {file_path}")
        eda_report.append(f"- Total records: {len(df)}")
        eda_report.append(f"- Columns: {', '.join(df.columns)}")

        eda_report.append("\n### Descriptive Statistics:")
        eda_report.append(df.describe(include="all").to_string())

        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 1:
            eda_report.append("\n\n### Correlation Analysis:")
            # Use optimized correlation analysis for large datasets
            if len(numeric_cols) > 20:
                corr_df = optimize_correlation_analysis(df[numeric_cols])
                eda_report.append("(Showing top correlated features for performance)")
                corr = corr_df.corr()
            else:
                corr = df[numeric_cols].corr()
            
            # Use semantic filter to show only meaningful correlations
            meaningful_corr_text = filter_correlation_matrix(corr)
            eda_report.append(meaningful_corr_text)
            
            # Only show full matrix for small datasets with meaningful correlations
            if len(numeric_cols) <= 5 and any(abs(corr.iloc[i, j]) >= 0.2 and is_meaningful_correlation(corr.columns[i], corr.columns[j]) 
                                             for i in range(len(corr)) for j in range(i+1, len(corr))):
                eda_report.append("\n**Full Correlation Matrix:**")
                eda_report.append(corr.round(3).to_string())

        eda_report.append("\n\n### Data Quality:")
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            eda_report.append("Missing Values:")
            for col, missing_count in missing_data.items():
                if missing_count > 0:
                    pct = (missing_count / len(df)) * 100
                    eda_report.append(f"- {col}: {missing_count} ({pct:.1f}%)")
        else:
            eda_report.append("No missing values found.")
        eda_report.append(f"Duplicate rows: {df.duplicated().sum()}")

        if len(numeric_cols) >= 2:
            eda_report.append("\n\n## Advanced Analysis")

            clean_numeric_df = df[numeric_cols].dropna()
            if len(clean_numeric_df) > 2:
                try:
                    cluster_count = n_clusters or min(3, len(clean_numeric_df) // 10)
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(clean_numeric_df)
                    kmeans = KMeans(
                        n_clusters=cluster_count,
                        random_state=random_state,
                        n_init=n_init,
                    )
                    clusters = kmeans.fit_predict(scaled_data)

                    df_with_clusters = clean_numeric_df.copy()
                    df_with_clusters["cluster"] = clusters

                    eda_report.append(
                        f"\n### KMeans Clustering ({cluster_count} clusters):"
                    )
                    cluster_summary = (
                        df_with_clusters.groupby("cluster")[numeric_cols]
                        .agg(["mean", "count"])
                        .round(2)
                    )
                    eda_report.append(cluster_summary.to_string())

                except Exception as e:
                    eda_report.append(f"\nClustering analysis failed: {e}")

            # Regression
            col1 = x_column or (numeric_cols[0] if len(numeric_cols) > 1 else None)
            col2 = y_column or (numeric_cols[1] if len(numeric_cols) > 1 else None)

            if col1 and col2 and col1 in df.columns and col2 in df.columns:
                try:
                    if pd.api.types.is_numeric_dtype(
                        df[col1]
                    ) and pd.api.types.is_numeric_dtype(df[col2]):
                        clean_data = df[[col1, col2]].dropna()
                        if len(clean_data) > 1:
                            model = LinearRegression()
                            X = clean_data[[col1]].values
                            y = clean_data[col2].values
                            model.fit(X, y)
                            r2 = model.score(X, y)
                            eda_report.append(
                                f"\n### Linear Regression: {col2} ~ {col1}"
                            )
                            eda_report.append(f"- R² Score: {r2:.4f}")
                            eda_report.append(f"- Coefficient: {model.coef_[0]:.4f}")
                            eda_report.append(f"- Intercept: {model.intercept_:.4f}")
                        else:
                            eda_report.append(
                                f"\nRegression skipped: not enough data for {col1} and {col2}."
                            )
                    else:
                        eda_report.append(
                            f"\nRegression skipped: {col1} or {col2} is not numeric."
                        )
                except Exception as e:
                    eda_report.append(f"\nRegression analysis failed: {e}")
            else:
                eda_report.append(
                    "\nRegression skipped: no valid column pair selected."
                )

        # INTELLIGENT ANALYSIS: Auto-detect common patterns and add specialized analysis
        if auto_detect_analysis:
            eda_report.append("\n## Intelligent Pattern Detection")
            
            # 1. TEMPORAL ANALYSIS: Auto-detect datetime columns
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            if not datetime_cols:
                # Try to detect datetime-like string columns
                for col in df.columns:
                    if any(word in col.lower() for word in ['time', 'date', 'timestamp', 'created', 'occurred']):
                        try:
                            pd.to_datetime(df[col].head(100))  # Test with sample
                            datetime_cols.append(col)
                            break
                        except:
                            continue
            
            if datetime_cols:
                eda_report.append(f"\n### 📅 TEMPORAL ANALYSIS DETECTED")
                datetime_col = datetime_cols[0]
                eda_report.append(f"**Temporal column identified**: {datetime_col}")
                
                try:
                    # Convert to datetime if needed
                    if df[datetime_col].dtype == 'object':
                        df[datetime_col] = pd.to_datetime(df[datetime_col])
                    
                    # Extract temporal components
                    df['_hour'] = df[datetime_col].dt.hour
                    df['_day_of_week'] = df[datetime_col].dt.day_name()
                    df['_month'] = df[datetime_col].dt.month
                    
                    # Hourly distribution
                    hourly_counts = df['_hour'].value_counts().sort_index()
                    peak_hour = hourly_counts.idxmax()
                    peak_count = hourly_counts.max()
                    eda_report.append(f"- **Peak Hour**: {peak_hour}:00 with {peak_count:,} records ({peak_count/len(df)*100:.1f}%)")
                    
                    # Daily distribution
                    daily_counts = df['_day_of_week'].value_counts()
                    peak_day = daily_counts.idxmax()
                    peak_day_count = daily_counts.max()
                    eda_report.append(f"- **Peak Day**: {peak_day} with {peak_day_count:,} records ({peak_day_count/len(df)*100:.1f}%)")
                    
                    # Date range
                    date_range = df[datetime_col].max() - df[datetime_col].min()
                    eda_report.append(f"- **Date Range**: {df[datetime_col].min().strftime('%Y-%m-%d')} to {df[datetime_col].max().strftime('%Y-%m-%d')} ({date_range.days} days)")
                    
                    # Optional: Save temporal analysis files (only for detailed analysis workflows)
                    # Note: CSV export removed to prevent automatic file generation
                    # Users can use dedicated export tools if CSV output is needed
                    
                    # Clean up temporary columns
                    df.drop(['_hour', '_day_of_week', '_month'], axis=1, inplace=True, errors='ignore')
                    
                except Exception as e:
                    eda_report.append(f"- Temporal analysis failed: {e}")
            
            # 2. CATEGORICAL ANALYSIS: Auto-detect incident/event type columns
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            category_col = None
            
            for col in categorical_cols:
                col_lower = col.lower()
                if any(word in col_lower for word in ['type', 'category', 'class', 'kind', 'incident', 'event']):
                    unique_ratio = df[col].nunique() / len(df)
                    if 0.001 < unique_ratio < 0.5:  # Good categorical ratio
                        category_col = col
                        break
            
            if category_col:
                eda_report.append(f"\n### 📊 CATEGORICAL ANALYSIS DETECTED")
                eda_report.append(f"**Category column identified**: {category_col}")
                
                try:
                    type_counts = df[category_col].value_counts()
                    total = len(df)
                    
                    eda_report.append(f"- **Total categories**: {len(type_counts)}")
                    eda_report.append(f"- **Most common**: {type_counts.index[0]} ({type_counts.iloc[0]:,} records, {type_counts.iloc[0]/total*100:.1f}%)")
                    if len(type_counts) > 1:
                        eda_report.append(f"- **Second most**: {type_counts.index[1]} ({type_counts.iloc[1]:,} records, {type_counts.iloc[1]/total*100:.1f}%)")
                    
                    # Distribution analysis
                    eda_report.append(f"\n**Full distribution:**")
                    for category, count in type_counts.head(10).items():
                        percentage = count / total * 100
                        eda_report.append(f"  - {category}: {count:,} ({percentage:.1f}%)")
                    
                    if len(type_counts) > 10:
                        eda_report.append(f"  - ... and {len(type_counts) - 10} more categories")
                        
                except Exception as e:
                    eda_report.append(f"- Categorical analysis failed: {e}")
            
            # 3. GEOGRAPHICAL ANALYSIS: Auto-detect lat/lon columns
            geo_cols = []
            for col in df.columns:
                col_lower = col.lower()
                if any(word in col_lower for word in ['lat', 'longitude', 'lng', 'coord']):
                    if pd.api.types.is_numeric_dtype(df[col]):
                        geo_cols.append(col)
            
            if len(geo_cols) >= 2:
                eda_report.append(f"\n### 🗺️ GEOGRAPHICAL ANALYSIS DETECTED")
                eda_report.append(f"**Geographic columns identified**: {', '.join(geo_cols[:2])}")
                
                try:
                    lat_col, lon_col = geo_cols[0], geo_cols[1]
                    # Basic geographic statistics
                    lat_range = df[lat_col].max() - df[lat_col].min()
                    lon_range = df[lon_col].max() - df[lon_col].min()
                    center_lat = df[lat_col].mean()
                    center_lon = df[lon_col].mean()
                    
                    eda_report.append(f"- **Geographic center**: ({center_lat:.4f}, {center_lon:.4f})")
                    eda_report.append(f"- **Latitude range**: {lat_range:.4f} degrees")
                    eda_report.append(f"- **Longitude range**: {lon_range:.4f} degrees")
                    eda_report.append(f"- **Spatial spread**: {'Wide' if lat_range > 1 or lon_range > 1 else 'Localized'}")
                    
                    # Check for actual spatial clustering or patterns (meaningful analysis)
                    if lat_range > 0.001 and lon_range > 0.001:  # Only if there's actual variation
                        eda_report.append(f"- **Data distribution**: Geographic points show spatial variation suitable for mapping")
                    else:
                        eda_report.append(f"- **Data distribution**: Points are highly concentrated (minimal spatial variation)")
                    
                except Exception as e:
                    eda_report.append(f"- Geographic analysis failed: {e}")

        # Append a machine-readable JSON footer so the orchestrator can capture schema/profile
        try:
            type_metadata = column_types.get("_metadata", {})
            footer_payload = {
                "path": file_path,
                "numeric_columns": type_metadata.get("numeric", []),
                "text_columns": type_metadata.get("categorical", []),
                "datetime_columns": type_metadata.get("datetime", []),
                "column_types": column_types,
            }
            eda_report.append("\n<!--output_json:" + json.dumps(footer_payload) + "-->")
        except Exception:
            pass

        return "\n".join(eda_report)

    except Exception as e:
        return f"Error performing EDA: {str(e)}"


# Focused on pattern and anomaly detection, including:
# If a target column is specified:
# Summary stats.
# Outlier detection using IQR.
# High value identification.
# For categorical: frequency and distribution.
# Overall column-type breakdown.
# Skewness of numeric columns.
# Uniqueness ratio for categorical columns.
# Returns Markdown-style pattern analysis.
@mcp.tool()
def analyze_csv_patterns(
    file_path: Annotated[str, Field(description="Path to the CSV file to analyze")],
    target_column: Annotated[
        str,
        Field(description="Name of the target column to analyze patterns for"),
    ] = None,
    iqr_multiplier: Annotated[
        float, Field(description="Multiplier for IQR to detect outliers")
    ] = 1.5,
) -> str:
    """
    Detect patterns, anomalies, and outliers in dataset columns using statistical methods.

    WHEN TO USE THIS TOOL:
    - Use to identify unusual data points and anomalies
    - Use to detect outliers in numeric columns
    - Use to find data quality issues
    - Use when you need distribution analysis for specific columns

    WHAT THIS TOOL DOES:
    If target_column specified:
    - Provides detailed summary statistics for that column
    - Detects outliers using IQR method (Interquartile Range)
    - Identifies high/low value records
    - For categorical: frequency distribution and top categories
    - For numeric: skewness, outlier bounds, extreme values

    If NO target_column specified:
    - Analyzes ALL columns for patterns
    - Reports numeric vs categorical breakdown
    - Calculates skewness for numeric columns
    - Reports uniqueness ratio for categorical columns

    RETURNS: Markdown report with:
    - Pattern analysis summary
    - Outlier detection results (with IQR bounds)
    - Distribution statistics
    - Data quality warnings

    PREREQUISITES: Dataset must be loaded
    PARAMETERS: Adjust iqr_multiplier (default 1.5) to control outlier sensitivity
    """
    try:
        # OPTIMIZED: Use fast file check instead of full validation
        if not fast_file_check(file_path):
            return f"Error: File '{file_path}' does not exist."

        df = load_csv_data_cached(file_path)

        patterns = ["## CSV Pattern Analysis"]
        patterns.append(f"- File: {file_path}")

        if target_column and target_column in df.columns:
            col_data = df[target_column]
            patterns.append(f"- Target column: {target_column}")

            if pd.api.types.is_numeric_dtype(col_data):
                patterns.append(f"\n### {target_column} Analysis (Numeric):")
                patterns.append(f"- Mean: {col_data.mean():.2f}")
                patterns.append(f"- Median: {col_data.median():.2f}")
                patterns.append(f"- Standard deviation: {col_data.std():.2f}")

                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - iqr_multiplier * IQR
                upper = Q3 + iqr_multiplier * IQR
                outliers = col_data[(col_data < lower) | (col_data > upper)]

                patterns.append(
                    f"- Outliers detected: {len(outliers)} ({len(outliers) / len(col_data) * 100:.1f}%)"
                )
                high_values = col_data[col_data > col_data.quantile(0.9)]
                patterns.append(f"- High values (top 10%): {len(high_values)}")
                patterns.append(
                    f"- High value range: {high_values.min():.2f} to {high_values.max():.2f}"
                )
            else:
                patterns.append(f"\n### {target_column} Analysis (Categorical):")
                value_counts = col_data.value_counts()
                patterns.append(f"- Unique values: {col_data.nunique()}")
                patterns.append(
                    f"- Most common: {value_counts.index[0]} ({value_counts.iloc[0]})"
                )
                patterns.append(f"- Value distribution: {dict(value_counts.head())}")

        numeric_cols = df.select_dtypes(include=["number"]).columns
        categorical_cols = df.select_dtypes(include=["object"]).columns

        patterns.append("\n### AI-Generated Pattern Analysis:")
        patterns.append(f"- Numeric columns: {len(numeric_cols)}")
        patterns.append(f"- Categorical columns: {len(categorical_cols)}")

        # OPTIMIZED: Single AI call instead of multiple separate calls
        pattern_context = (
            f"Pattern analysis for dataset with target column: {target_column}"
            if target_column
            else "General pattern analysis"
        )
        
        return "\n".join(patterns)

    except Exception as e:
        return f"Error analyzing patterns: {str(e)}"


# --- New Functions Start Here ---


@mcp.tool()
def handle_missing_values(
    file_path: Annotated[str, Field(description="Path to the CSV file to process")],
    strategy: Annotated[
        Literal["mean", "median", "mode", "drop_rows", "drop_columns", "constant"],
        Field(
            description="Strategy for handling missing values: 'mean', 'median', 'mode' (impute), 'drop_rows', 'drop_columns', or 'constant'"
        ),
    ],
    columns: Annotated[
        List[str],
        Field(
            description="List of columns to apply the strategy to. If empty, applies to all relevant columns."
        ),
    ] = None,
    fill_value: Annotated[
        str,
        Field(description="Value to fill missing data with if strategy is 'constant'"),
    ] = None,
) -> str:
    """
    Clean missing values in a CSV file using flexible imputation or removal strategies.

    Supported strategies:
    - 'mean', 'median', 'mode': Impute missing values with the respective statistic
    - 'drop_rows', 'drop_columns': Remove rows or columns containing missing values
    - 'constant': Fill missing values with a specified constant

    Returns a Markdown report of changes and saves the cleaned data to a new CSV file.
    """
    try:
        # OPTIMIZED: Use fast file check
        if not fast_file_check(file_path):
            return f"Error: File '{file_path}' does not exist."

        df = load_csv_data_cached(file_path)
        original_rows = len(df)
        original_cols = len(df.columns)
        report = [f"## Handling Missing Values in {file_path}"]

        cols_to_process = columns if columns else df.columns.tolist()

        if strategy == "drop_rows":
            initial_missing_rows = df.isnull().any(axis=1).sum()
            df.dropna(subset=cols_to_process, inplace=True)
            rows_removed = original_rows - len(df)
            report.append(
                f"Strategy: Dropped rows with missing values in specified columns ({', '.join(cols_to_process)} if specified, else any)."
            )
            report.append(
                f"Rows removed: {rows_removed} (out of {original_rows} total)."
            )

        elif strategy == "drop_columns":
            cols_before_drop = set(df.columns)
            df.dropna(
                axis=1, how="any", subset=cols_to_process, inplace=True
            )  # Drops columns if any NA in subset
            cols_removed = cols_before_drop - set(df.columns)
            report.append(
                f"Strategy: Dropped columns with any missing values in specified columns ({', '.join(cols_to_process)} if specified, else any)."
            )
            report.append(
                f"Columns removed: {', '.join(cols_removed) if cols_removed else 'None'}."
            )

        elif strategy in ["mean", "median", "mode", "constant"]:
            report.append(f"Strategy: Imputing missing values using '{strategy}'.")
            imputed_count = 0

            for col in cols_to_process:
                if col not in df.columns:
                    report.append(
                        f"Warning: Column '{col}' not found. Skipping imputation for this column."
                    )
                    continue

                if df[col].isnull().sum() > 0:
                    missing_before_col = df[col].isnull().sum()
                    if strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
                        df[col].fillna(df[col].mean(), inplace=True)
                        report.append(
                            f"- Imputed {missing_before_col} missing values in numeric column '{col}' with its mean ({df[col].mean():.2f})."
                        )
                        imputed_count += missing_before_col
                    elif strategy == "median" and pd.api.types.is_numeric_dtype(
                        df[col]
                    ):
                        df[col].fillna(df[col].median(), inplace=True)
                        report.append(
                            f"- Imputed {missing_before_col} missing values in numeric column '{col}' with its median ({df[col].median():.2f})."
                        )
                        imputed_count += missing_before_col
                    elif strategy == "mode":
                        # Mode can return multiple values, pick the first
                        mode_val = df[col].mode()[0]
                        df[col].fillna(mode_val, inplace=True)
                        report.append(
                            f"- Imputed {missing_before_col} missing values in column '{col}' with its mode ('{mode_val}')."
                        )
                        imputed_count += missing_before_col
                    elif strategy == "constant":
                        if fill_value is not None:
                            df[col].fillna(fill_value, inplace=True)
                            report.append(
                                f"- Imputed {missing_before_col} missing values in column '{col}' with constant value '{fill_value}'."
                            )
                            imputed_count += missing_before_col
                        else:
                            report.append(
                                f"- Skipped '{col}': 'constant' strategy requires 'fill_value'."
                            )
                    else:
                        report.append(
                            f"- Skipped '{col}': '{strategy}' strategy is only applicable to numeric columns for mean/median."
                        )

            if (
                imputed_count == 0 and strategy != "constant"
            ):  # Only if fill_value wasn't provided or no columns matched
                report.append(
                    "No missing values were imputed based on the strategy and column types."
                )

        else:
            return f"Error: Invalid strategy '{strategy}'. Choose from 'mean', 'median', 'mode', 'drop_rows', 'drop_columns', 'constant'."

        output_file = Path(file_path).stem + "_cleaned.csv"
        df.to_csv(output_file, index=False)
        report.append(f"\nProcessed data saved to '{output_file}'.")
        report.append(
            f"Final dataset shape: {len(df)} rows, {len(df.columns)} columns."
        )

        return "\n".join(report)

    except Exception as e:
        return f"Error handling missing values: {str(e)}"


@mcp.tool()
def handle_outliers(
    file_path: Annotated[str, Field(description="Path to the CSV file to process")],
    column: Annotated[
        str,
        Field(description="Column to detect and handle outliers in (must be numeric)"),
    ],
    strategy: Annotated[
        Literal["remove", "cap"],
        Field(
            description="Strategy for handling outliers: 'remove' rows or 'cap' values"
        ),
    ],
    iqr_multiplier: Annotated[
        float,
        Field(
            description="Multiplier for IQR to determine outlier bounds (default=1.5)"
        ),
    ] = 1.5,
) -> str:
    """
    Detect and handle outliers in a numeric column using the IQR method.

    - 'remove': Exclude rows containing outliers
    - 'cap': Replace outlier values with the nearest non-outlier bound

    Returns a Markdown report and saves the processed data to a new CSV file.
    """
    try:
        if not Path(file_path).exists():
            return f"Error: File '{file_path}' does not exist."

        df = load_csv_data_cached(file_path)

        if column not in df.columns:
            return f"Error: Column '{column}' not found in the dataset."
        if not pd.api.types.is_numeric_dtype(df[column]):
            return f"Error: Column '{column}' is not numeric. Outlier detection is for numeric columns only."

        report = [f"## Handling Outliers in '{column}' of {file_path}"]

        col_data = df[column].dropna()  # Perform calculation on non-null data
        if len(col_data) == 0:
            return f"No non-missing data in column '{column}' to detect outliers."

        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR

        outliers_count = len(
            df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        )

        if outliers_count == 0:
            report.append(
                f"No outliers detected in column '{column}' using IQR multiplier {iqr_multiplier}."
            )
            return "\n".join(report)

        report.append(
            f"Outliers detected: {outliers_count} ({outliers_count / len(df) * 100:.2f}%) in column '{column}'."
        )
        report.append(f"IQR Bounds: Lower={lower_bound:.2f}, Upper={upper_bound:.2f}")

        if strategy == "remove":
            original_rows = len(df)
            df = df[~((df[column] < lower_bound) | (df[column] > upper_bound))]
            rows_removed = original_rows - len(df)
            report.append(f"Strategy: Removed {rows_removed} rows containing outliers.")
        elif strategy == "cap":
            df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
            df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
            report.append(
                f"Strategy: Capped outlier values in '{column}' to be within [{lower_bound:.2f}, {upper_bound:.2f}]."
            )
        else:
            return f"Error: Invalid strategy '{strategy}'. Choose 'remove' or 'cap'."

        output_file = Path(file_path).stem + "_outliers_handled.csv"
        df.to_csv(output_file, index=False)
        report.append(f"\nProcessed data saved to '{output_file}'.")
        report.append(
            f"Final dataset shape: {len(df)} rows, {len(df.columns)} columns."
        )

        return "\n".join(report)

    except Exception as e:
        return f"Error handling outliers: {str(e)}"


@mcp.tool()
def perform_t_test(
    file_path: Annotated[str, Field(description="Path to the CSV file to analyze")],
    group_column: Annotated[
        str, Field(description="Categorical column defining the two groups")
    ],
    value_column: Annotated[
        str, Field(description="Numeric column whose means will be compared")
    ],
    group1_name: Annotated[
        str, Field(description="Name of the first group for comparison")
    ],
    group2_name: Annotated[
        str, Field(description="Name of the second group for comparison")
    ],
    independent_samples: Annotated[
        bool,
        Field(
            description="True for independent samples t-test, False for paired samples t-test"
        ),
    ] = True,
) -> str:
    """
    Perform statistical t-test to compare means between two groups.

    WHEN TO USE THIS TOOL:
    - Use when you need to compare average values between two groups
    - Use to test if observed differences are statistically significant
    - Use AFTER exploratory analysis identifies potential group differences

    WHAT THIS TOOL DOES:
    - Performs Welch's t-test (independent samples, default) or paired t-test
    - Calculates t-statistic and p-value
    - Tests null hypothesis: "means are equal between groups"
    - Determines if difference is statistically significant (typically p < 0.05)

    USE CASE EXAMPLES:
    - Compare average incident duration between weekdays vs weekends
    - Compare average severity scores between morning vs evening hours
    - Compare average values between two incident types

    RETURNS: Markdown report with:
    - Test type (independent or paired)
    - T-statistic value
    - P-value and significance interpretation
    - Group means and difference

    PREREQUISITES: 
    - Requires categorical column with exactly 2 groups
    - Requires numeric column for comparison
    - Minimum 2 data points per group
    """
    try:
        if not Path(file_path).exists():
            return f"Error: File '{file_path}' does not exist."

        df = load_csv_data_cached(file_path)

        if group_column not in df.columns:
            return f"Error: Group column '{group_column}' not found."
        if value_column not in df.columns:
            return f"Error: Value column '{value_column}' not found."
        if not pd.api.types.is_numeric_dtype(df[value_column]):
            return f"Error: Value column '{value_column}' is not numeric."

        group1_data = df[df[group_column] == group1_name][value_column].dropna()
        group2_data = df[df[group_column] == group2_name][value_column].dropna()

        if len(group1_data) < 2 or len(group2_data) < 2:
            return "Error: Not enough data points in one or both groups for a t-test (at least 2 required per group)."

        report = [
            f"## T-Test Analysis: Comparing '{value_column}' between '{group1_name}' and '{group2_name}'"
        ]

        if independent_samples:
            t_stat, p_value = stats.ttest_ind(
                group1_data, group2_data, equal_var=False
            )  # Welch's t-test by default
            test_type = "Independent Samples (Welch's)"
        else:
            if len(group1_data) != len(group2_data):
                return "Error: For paired samples t-test, both groups must have the same number of observations after dropping NaNs."
            t_stat, p_value = stats.ttest_rel(group1_data, group2_data)
            test_type = "Paired Samples"

        report.append(f"- Test Type: {test_type}")
        report.append(f"- T-statistic: {t_stat:.4f}")
        report.append(f"- P-value: {p_value:.4f}")

        return "\n".join(report)

    except Exception as e:
        return f"Error performing t-test: {str(e)}"


@mcp.tool()
def perform_anova(
    file_path: Annotated[str, Field(description="Path to the CSV file to analyze")],
    dependent_column: Annotated[
        str, Field(description="Numeric column (dependent variable)")
    ],
    group_column: Annotated[
        str,
        Field(description="Categorical column defining groups (independent variable)"),
    ],
) -> str:
    """
    Perform one-way ANOVA to compare means across multiple groups (3+).

    WHEN TO USE THIS TOOL:
    - Use when comparing averages across THREE OR MORE groups
    - Use when you have one categorical variable and one numeric variable
    - Use INSTEAD of t-test when you have more than 2 groups
    - Use AFTER exploratory analysis identifies group differences

    WHAT THIS TOOL DOES:
    - Performs one-way Analysis of Variance (ANOVA)
    - Tests null hypothesis: "all group means are equal"
    - Calculates F-statistic and p-value
    - Reports mean values for each group
    - Determines if at least one group differs significantly

    USE CASE EXAMPLES:
    - Compare average severity across multiple incident types
    - Compare average duration across different days of week
    - Compare average values across multiple road segments

    RETURNS: Markdown report with:
    - F-statistic value
    - P-value and significance interpretation
    - Mean value for each group
    - Group count information

    PREREQUISITES:
    - Requires categorical column with 3+ groups
    - Requires numeric column for comparison
    - Each group needs at least 2 data points
    
    NOTE: ANOVA only tells you groups differ - use post-hoc tests to find which pairs differ
    """
    try:
        if not Path(file_path).exists():
            return f"Error: File '{file_path}' does not exist."

        df = load_csv_data_cached(file_path)

        if dependent_column not in df.columns:
            return f"Error: Dependent column '{dependent_column}' not found."
        if group_column not in df.columns:
            return f"Error: Group column '{group_column}' not found."
        if not pd.api.types.is_numeric_dtype(df[dependent_column]):
            return f"Error: Dependent column '{dependent_column}' is not numeric."

        # Ensure group column is treated as categorical
        df[group_column] = df[group_column].astype("category")

        # Filter out NaN values from both columns
        clean_df = df[[dependent_column, group_column]].dropna()

        # Get unique groups
        groups = [
            clean_df[clean_df[group_column] == g][dependent_column]
            for g in clean_df[group_column].unique()
        ]

        if len(groups) < 2:
            return "Error: ANOVA requires at least two groups for comparison."

        # Check if each group has enough data
        for i, group_data in enumerate(groups):
            if len(group_data) < 2:
                return f"Error: Group '{clean_df[group_column].unique()[i]}' has less than 2 data points. ANOVA requires at least 2 per group."

        f_stat, p_value = stats.f_oneway(*groups)

        report = [
            f"## ANOVA Test: Comparing '{dependent_column}' across groups defined by '{group_column}'"
        ]
        report.append(f"- F-statistic: {f_stat:.4f}")
        report.append(f"- P-value: {p_value:.4f}")

        # AI-Generated Statistical Interpretation
        group_means = {
            str(clean_df[group_column].unique()[i]): float(group_data.mean())
            for i, group_data in enumerate(groups)
        }

        return "\n".join(report)

    except Exception as e:
        return f"Error performing ANOVA: {str(e)}"


@mcp.tool()
def generate_plot(
    file_path: Annotated[str, Field(description="Path to the CSV file to plot")],
    plot_type: Annotated[
        Literal[
            "histogram",
            "scatterplot",
            "boxplot",
            "pairplot",
            "bar_chart",
            "auto_temporal",
            "auto_categorical",
            "temporal_line",
            "temporal_bar",
            "temporal_heatmap",
            "temporal_matrix",
            "temporal_distribution",
        ],
        Field(
            description="Type of plot to generate. Standard: histogram, scatterplot, boxplot, pairplot, bar_chart. Smart: auto_temporal (detects datetime and chooses a sensible temporal plot), auto_categorical. Temporal modes: temporal_line, temporal_bar, temporal_heatmap, temporal_matrix, temporal_distribution."
        ),
    ],
    x_column: Annotated[
        str,
        Field(
            description="X-axis column for scatterplot, bar_chart, or column for histogram/boxplot"
        ),
    ] = None,
    y_column: Annotated[
        str, Field(description="Y-axis column for scatterplot or bar_chart")
    ] = None,
    hue_column: Annotated[
        str,
        Field(
            description="Optional column for color encoding (e.g., for scatterplot or boxplot)"
        ),
    ] = None,
    columns_for_pairplot: Annotated[
        List[str],
        Field(
            description="List of numeric columns for pairplot. Max 5 columns recommended."
        ),
    ] = None,
    title: Annotated[str, Field(description="Title of the plot")] = "Generated Plot",
    output_dir: Annotated[
        str,
        Field(
            description="Directory to save the plot image file. If not provided, plot is not saved locally."
        ),
    ] = "src/agents/analysis/reportDemo",
) -> str:
    """
        Generate and save statistical and temporal plots from a CSV dataset.

        WHEN TO USE THIS TOOL:
        - You want a quick visualization without writing plotting code
        - Categorical counts: use bar_chart with x_column only (y optional) to plot frequencies
        - Numeric distributions: use histogram or boxplot
        - Relationships between two numeric columns: use scatterplot
    - Correlation overview: use pairplot (up to ~5 numeric columns)
        - Time-based views: use temporal_* modes or auto_temporal when a datetime column exists
        - Not sure which plot to pick: use auto_temporal or auto_categorical for sensible defaults

        WHAT THIS TOOL DOES:
        - Loads the CSV using cached reading and basic type inference
        - Validates that requested columns exist; adapts to common plot/column mismatches
        - bar_chart: if y_column is omitted or non-numeric, plots counts of x_column
        - Day-of-week intelligence: maps 0–6 or 1–7 codes to weekday names and orders Monday→Sunday
        - Temporal modes materialize features (hour, date, weekday, month, year_week) and aggregate appropriately
        - Saves a PNG to output_dir and returns a Markdown image reference with a short relative path

        RETURNS:
        - Markdown string containing:
            - An inline image embed (![title](relative_filename.png))
            - A "Saved:" line with the full saved path
            - A brief "Summary:" of what was plotted

        OUTPUT FILES CREATED:
        - <sanitized_title>_<plot_type>_<timestamp>.png
        - Saved under output_dir (default: src/agents/analysis/reportDemo)

        PREREQUISITES:
        - file_path must exist and be readable as CSV
        - scatterplot requires both x_column and y_column, typically numeric
        - histogram/boxplot require a valid x_column (numeric preferred)
    - pairplot operates on numeric columns (provide columns_for_pairplot as needed)
        - temporal_* modes require a parseable datetime column (auto-detected when temporal tokens are used)

        Supported plot types:
    - histogram, scatterplot, boxplot, pairplot, bar_chart
    - auto_temporal, auto_categorical, temporal_line, temporal_bar, temporal_matrix, temporal_distribution
    """
    try:
        # Validate file and load data with caching
        validate_file_and_columns(file_path)
        df = load_csv_data_cached(file_path)

        # ----------------------
        # Helpers (internal only)
        # ----------------------
        def _detect_datetime_column(local_df: pd.DataFrame) -> str:
            # Prefer actual datetime dtype
            dt_cols = local_df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]', 'datetimetz']).columns.tolist()
            if dt_cols:
                return dt_cols[0]
            # Try heuristic detection via name + parsability
            for col in local_df.columns:
                col_l = str(col).lower()
                if any(k in col_l for k in ["time", "date", "timestamp", "created", "occurred"]):
                    try:
                        pd.to_datetime(local_df[col].head(100), errors='raise')
                        return col
                    except Exception:
                        continue
            # Fallback: attempt first object-like column that parses
            for col in local_df.columns:
                if local_df[col].dtype == 'object':
                    try:
                        pd.to_datetime(local_df[col].head(50), errors='raise')
                        return col
                    except Exception:
                        continue
            return None

        def _ensure_datetime(local_df: pd.DataFrame, col: str) -> bool:
            if col is None or col not in local_df.columns:
                return False
            if not pd.api.types.is_datetime64_any_dtype(local_df[col]):
                try:
                    local_df[col] = pd.to_datetime(local_df[col], errors='coerce')
                except Exception:
                    return False
            return not local_df[col].isna().all()

        def _add_temporal_features(local_df: pd.DataFrame, col: str) -> None:
            # Create common temporal features if not present
            if 'hour' not in local_df.columns:
                local_df['hour'] = local_df[col].dt.hour
            if 'date' not in local_df.columns:
                local_df['date'] = local_df[col].dt.date
            if 'day' not in local_df.columns:
                local_df['day'] = local_df[col].dt.day
            if 'weekday' not in local_df.columns:
                local_df['weekday'] = local_df[col].dt.day_name()
            if 'weekday_num' not in local_df.columns:
                local_df['weekday_num'] = local_df[col].dt.dayofweek
            # ISO week and year for correct cross-year handling
            if 'iso_year' not in local_df.columns:
                try:
                    local_df['iso_year'] = local_df[col].dt.isocalendar().year
                except Exception:
                    # Older pandas: fallback
                    local_df['iso_year'] = local_df[col].dt.year
            if 'week' not in local_df.columns:
                try:
                    local_df['week'] = local_df[col].dt.isocalendar().week
                except Exception:
                    local_df['week'] = local_df[col].dt.week
            if 'year_week' not in local_df.columns:
                local_df['year_week'] = local_df['iso_year'].astype(str) + '-W' + local_df['week'].astype(str)
            if 'month' not in local_df.columns:
                local_df['month'] = local_df[col].dt.month

        def _materialize_temporal_if_requested(local_df: pd.DataFrame, dt_col: str, maybe_cols: list) -> list:
            # If user passed derived names (hour, weekday, week, month, date), ensure present
            derived = {"hour", "weekday", "weekday_num", "week", "month", "date", "day", "year_week"}
            out = []
            for c in maybe_cols:
                if c is None:
                    out.append(None)
                elif c in local_df.columns:
                    out.append(c)
                elif c in derived:
                    _add_temporal_features(local_df, dt_col)
                    out.append(c)
                else:
                    out.append(c)
            return out

        def _pivot_counts(local_df: pd.DataFrame, index_col: str, col_col: str) -> pd.DataFrame:
            try:
                pvt = local_df.pivot_table(index=index_col, columns=col_col, values=col_col, aggfunc='count', fill_value=0)
            except Exception:
                # Fallback to crosstab
                pvt = pd.crosstab(local_df[index_col], local_df[col_col])
            return pvt

        def _agg_series(local_df: pd.DataFrame, by_col: str, val_col: str = None, how: str = 'count') -> pd.Series:
            if val_col is not None and val_col in local_df.columns and pd.api.types.is_numeric_dtype(local_df[val_col]) and how in ('sum', 'mean', 'median'):
                return getattr(local_df.groupby(by_col)[val_col], how)()
            # default to counts
            return local_df.groupby(by_col).size()

        def _save_and_return(_title: str, _plot_type: str, _desc: str) -> str:
            plt.tight_layout()
            plot_file_path = None
            if output_dir:
                if not Path(output_dir).is_absolute():
                    project_root = Path(__file__).parent.parent.parent.parent
                    output_path = project_root / output_dir
                else:
                    output_path = Path(output_dir)

                output_path.mkdir(parents=True, exist_ok=True)
                sanitized_title = "".join(c for c in _title if c.isalnum() or c in (" ", ".", "_")).rstrip()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_file_name = f"{sanitized_title.replace(' ', '_').replace('.', '')}_{_plot_type}_{timestamp}.png"
                plot_file_path = output_path / plot_file_name
                plt.savefig(plot_file_path, dpi=150, bbox_inches="tight")

            cleanup_plot()

            if plot_file_path:
                # Backward compatible markdown + explicit saved path + one-line summary
                # Use "Saved to:" pattern to align with pipeline artifact parser
                return (
                    f"![{_title}]({plot_file_path.name})\n"
                    f"Saved to: {plot_file_path}\n"
                    f"Artifact: {plot_file_path.name}\n"
                    f"Summary: {_desc}"
                )
            else:
                return f"Plot generated but not saved to disk.\nSummary: {_desc}"

        # DYNAMIC PLOT TYPE COMPATIBILITY CHECK
        # Intelligently detect plot type mismatches and suggest better alternatives
        if plot_type in ["histogram", "boxplot"] and x_column:
            if x_column in df.columns and not pd.api.types.is_numeric_dtype(df[x_column]):
                # Categorical column detected for numeric plot - intelligently fallback
                return generate_plot(
                    file_path=file_path,
                    plot_type="auto_categorical",
                    x_column=None,  # Let auto_categorical detect best column
                    y_column=None,
                    hue_column=hue_column,
                    title=f"{title} (Auto-converted from {plot_type})",
                    output_dir=output_dir
                )
        
        elif plot_type == "scatterplot" and x_column and y_column:
            x_numeric = pd.api.types.is_numeric_dtype(df[x_column]) if x_column in df.columns else False
            y_numeric = pd.api.types.is_numeric_dtype(df[y_column]) if y_column in df.columns else False
            
            if not x_numeric or not y_numeric:
                # Mixed or categorical data for scatterplot - suggest bar chart
                return generate_plot(
                    file_path=file_path,
                    plot_type="bar_chart",
                    x_column=x_column if not x_numeric else y_column,  # Use categorical column as x
                    y_column=y_column if y_numeric else x_column,      # Use numeric column as y if available
                    hue_column=hue_column,
                    title=f"{title} (Auto-converted from scatterplot)",
                    output_dir=output_dir
                )

        # Use optimized plot creation
        create_optimized_plot()

        # AUTO-PLOT MODES: Intelligent plot generation and new temporal modes
        if plot_type == "auto_temporal":
            # Auto-detect temporal patterns and create appropriate plots
            dt_col = _detect_datetime_column(df)
            if not _ensure_datetime(df, dt_col):
                cleanup_plot()
                return "Error: No temporal columns detected for auto_temporal plot."

            _add_temporal_features(df, dt_col)
            # Heuristic: if data spans > 2 days, use daily bar; else hourly bar
            try:
                date_span = pd.to_datetime(pd.Series(list(set(df['date'])))).sort_values()
                many_days = len(date_span) >= 3
            except Exception:
                many_days = False

            if many_days:
                # Daily counts as bar
                ser = _agg_series(df, by_col='date', how='count')
                plt.figure(figsize=(12, 6))
                ser.sort_index().plot(kind='bar', color='steelblue', alpha=0.85)
                plt.title('Incidents per Day')
                plt.xlabel('Date')
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                return _save_and_return("Incidents per Day", plot_type, "Daily count distribution over time.")
            else:
                # Hourly counts as bar
                ser = _agg_series(df, by_col='hour', how='count')
                plt.figure(figsize=(12, 6))
                ser.sort_index().plot(kind='bar', color='skyblue', alpha=0.85)
                plt.title('Incidents by Hour of Day')
                plt.xlabel('Hour of Day')
                plt.ylabel('Count')
                plt.xticks(rotation=0)
                return _save_and_return("Incidents by Hour of Day", plot_type, "Hourly count distribution across the day.")

        elif plot_type == "auto_categorical":
            # Auto-detect categorical columns and create distribution plots
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            category_col = None
            
            for col in categorical_cols:
                col_lower = col.lower()
                if any(word in col_lower for word in ['type', 'category', 'class', 'kind', 'incident', 'event']):
                    unique_ratio = df[col].nunique() / len(df)
                    if 0.001 < unique_ratio < 0.5:  # Good categorical ratio
                        category_col = col
                        break
            
            if not category_col:
                cleanup_plot()
                return "Error: No categorical columns detected for auto_categorical plot."
            
            type_counts = df[category_col].value_counts()
            total = len(df)
            
            plt.figure(figsize=(12, 6))
            type_counts.plot(kind='bar', color='lightcoral', alpha=0.8)
            plt.title(f'Distribution by {category_col.title()}')
            plt.xlabel(category_col.title())
            plt.ylabel('Number of Records')
            plt.xticks(rotation=45)
            plt.grid(axis='y', alpha=0.3)
            
            # Add percentages on bars
            for i, (cat_name, count) in enumerate(type_counts.items()):
                plt.text(i, count + count*0.01, f'{count:,}\n({count/total*100:.1f}%)', 
                        ha='center', va='bottom', fontweight='bold')
            
            title = f"Categorical Distribution - {category_col}"

        elif plot_type == "histogram":
            if not x_column:
                cleanup_plot()
                return "Error: 'x_column' is required for histogram."
            if x_column not in df.columns:
                cleanup_plot()
                return f"Error: Column '{x_column}' not found."
            if not pd.api.types.is_numeric_dtype(df[x_column]):
                cleanup_plot()
                return f"Error: Column '{x_column}' is not numeric for histogram."

            sns.histplot(df[x_column].dropna(), kde=True)
            plt.title(
                f"Histogram of {x_column}" if title == "Generated Plot" else title
            )
            plt.xlabel(x_column)
            plt.ylabel("Frequency")

        elif plot_type == "scatterplot":
            if not x_column or not y_column:
                cleanup_plot()
                return "Error: 'x_column' and 'y_column' are required for scatterplot."
            if x_column not in df.columns or y_column not in df.columns:
                cleanup_plot()
                return f"Error: One or both columns ('{x_column}', '{y_column}') not found."
            if not pd.api.types.is_numeric_dtype(
                df[x_column]
            ) or not pd.api.types.is_numeric_dtype(df[y_column]):
                cleanup_plot()
                return "Error: Both 'x_column' and 'y_column' must be numeric for scatterplot."

            sns.scatterplot(
                x=df[x_column],
                y=df[y_column],
                hue=df[hue_column] if hue_column else None,
            )
            plt.title(
                f"Scatter Plot of {y_column} vs {x_column}"
                if title == "Generated Plot"
                else title
            )
            plt.xlabel(x_column)
            plt.ylabel(y_column)

        elif plot_type == "boxplot":
            if not x_column:
                cleanup_plot()
                return "Error: 'x_column' is required for boxplot (the column to plot)."
            if x_column not in df.columns:
                cleanup_plot()
                return f"Error: Column '{x_column}' not found."
            # Note: Categorical data fallback is now handled by the dynamic compatibility check above
            if not pd.api.types.is_numeric_dtype(df[x_column]):
                cleanup_plot()
                return f"Error: Column '{x_column}' is not numeric for boxplot."

            sns.boxplot(
                y=df[x_column], x=df[hue_column] if hue_column else None
            )  # x is for grouping, y is the value
            plt.title(f"Box Plot of {x_column}" if title == "Generated Plot" else title)
            plt.ylabel(x_column)
            if hue_column:
                plt.xlabel(hue_column)

        

        elif plot_type == "pairplot":
            if not columns_for_pairplot:
                cleanup_plot()
                return "Error: 'columns_for_pairplot' is required for pairplot."
            if len(columns_for_pairplot) > 5:
                cleanup_plot()
                return "Warning: Pairplot with more than 5 columns can be very slow. Please select fewer columns."

            subset_df = df[columns_for_pairplot].select_dtypes(include=["number"])
            if subset_df.empty:
                cleanup_plot()
                return "Error: No numeric columns found in the specified list for pairplot."

            sns.pairplot(
                subset_df.dropna(),
                hue=hue_column if hue_column in df.columns else None,
            )
            plt.suptitle(
                "Pair Plot" if title == "Generated Plot" else title, y=1.02
            )  # Adjust suptitle position

        elif plot_type == "bar_chart":
            # Allow y_column to be optional. If missing or non-numeric, plot counts of x.
            if not x_column:
                cleanup_plot()
                return "Error: 'x_column' is required for bar chart."
            if x_column not in df.columns:
                cleanup_plot()
                return f"Error: Column '{x_column}' not found."

            use_counts = True
            if y_column and (y_column in df.columns) and pd.api.types.is_numeric_dtype(df[y_column]):
                use_counts = False

            # Helper: map day-of-week codes to names when x looks like a DOW code
            def _maybe_map_day_names(series: pd.Series, col_name: str) -> Tuple[pd.Series, List[str]]:
                try:
                    low = str(col_name).lower()
                    order: List[str] = []
                    if any(tok in low for tok in ["day_of_week", "dayofweek", "weekday", "dow"]):
                        s = pd.to_numeric(series, errors='coerce')
                        unique = sorted([int(v) for v in s.dropna().unique().tolist()])
                        # Heuristics: 0-6 => Monday..Sunday; 1-7 => Sunday..Saturday (common in crash datasets)
                        if set(unique).issubset(set(range(0,7))):
                            names = {0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"}
                            mapped = s.map(names).fillna(series.astype(str))
                            order = [names[i] for i in range(0,7)]
                            return mapped, order
                        if set(unique).issubset(set(range(1,8))):
                            # Assume 1=Sunday..7=Saturday (TxDOT/CRIS convention)
                            names = {1:"Sunday",2:"Monday",3:"Tuesday",4:"Wednesday",5:"Thursday",6:"Friday",7:"Saturday"}
                            mapped = s.map(names).fillna(series.astype(str))
                            order = [names[i] for i in range(1,8)]
                            return mapped, order
                    return series.astype(str), order
                except Exception:
                    return series.astype(str), []

            if use_counts:
                # Count occurrences of x_column
                x_series = df[x_column]
                x_series, category_order = _maybe_map_day_names(x_series, x_column)
                value_counts = x_series.value_counts().reindex(category_order or None)
                # value_counts may become all NaN if order provided; handle default ordering
                if value_counts.isna().all():
                    value_counts = x_series.value_counts()
                vc_df = value_counts.reset_index()
                vc_df.columns = [x_column, "Count"]
                plt.figure(figsize=(12, 6))
                sns.barplot(x=vc_df[x_column], y=vc_df["Count"], order=category_order if category_order else None, color='steelblue', alpha=0.85)
                plt.title(
                    f"Bar Chart of Counts for {x_column}"
                    if title == "Generated Plot"
                    else title
                )
                plt.xlabel(x_column)
                plt.ylabel("Count")
                plt.xticks(rotation=45, ha="right")
            else:
                # Numeric y: direct bar heights by y values (not recommended for categorical x without aggregation)
                plt.figure(figsize=(12, 6))
                x_series = df[x_column]
                x_series, category_order = _maybe_map_day_names(x_series, x_column)
                sns.barplot(x=x_series, y=df[y_column], hue=df[hue_column] if hue_column else None)
                plt.title(
                    f"Bar Chart of {y_column} by {x_column}"
                    if title == "Generated Plot"
                    else title
                )
                plt.xlabel(x_column)
                plt.ylabel(y_column)
                plt.xticks(rotation=45, ha="right")

        elif plot_type == "temporal_distribution":
            # 1D temporal distribution as bars or line depending on density
            dt_col = _detect_datetime_column(df)
            if not _ensure_datetime(df, dt_col):
                cleanup_plot()
                return "Error: temporal_distribution requires a datetime column."
            _add_temporal_features(df, dt_col)

            # Choose bucket: prefer explicit x_column if it is a temporal feature
            candidates = ['hour', 'date', 'weekday', 'month', 'year_week']
            x_choice = x_column if x_column in candidates else ('date' if df.get('date', pd.Series()).nunique() >= 3 else 'hour')
            ser = _agg_series(df, by_col=x_choice, how='count')
            plt.figure(figsize=(12, 6))
            kind = 'line' if len(ser) > 30 else 'bar'
            if kind == 'bar':
                ser.sort_index().plot(kind='bar', color='steelblue', alpha=0.85)
                plt.xticks(rotation=45 if x_choice in ('date', 'year_week') else 0, ha='right')
            else:
                ser.sort_index().plot(kind='line', color='steelblue', marker='o', alpha=0.85)
                plt.xticks(rotation=45 if x_choice in ('date', 'year_week') else 0, ha='right')
            ttl = f"Temporal Distribution by {x_choice.title()}"
            plt.title(ttl)
            plt.xlabel(x_choice.title())
            plt.ylabel('Count')
            return _save_and_return(ttl, plot_type, f"Counts aggregated by {x_choice}.")

        elif plot_type == "temporal_line":
            dt_col = _detect_datetime_column(df)
            if not _ensure_datetime(df, dt_col):
                cleanup_plot()
                return "Error: temporal_line requires a datetime column."
            _add_temporal_features(df, dt_col)

            # Determine x bucket (default: date) and y aggregation
            x_choice = x_column if x_column else 'date'
            [x_choice] = _materialize_temporal_if_requested(df, dt_col, [x_choice])
            agg_how = 'mean' if (y_column and y_column in df.columns and pd.api.types.is_numeric_dtype(df[y_column])) else 'count'
            ser = _agg_series(df, by_col=x_choice, val_col=(y_column if agg_how != 'count' else None), how=agg_how)
            plt.figure(figsize=(12, 6))
            ser.sort_index().plot(kind='line', marker='o', color='teal', alpha=0.9)
            ttl = f"Temporal Line by {x_choice.title()}"
            plt.title(ttl)
            plt.xlabel(x_choice.title())
            plt.ylabel('Value' if agg_how != 'count' else 'Count')
            plt.xticks(rotation=45 if x_choice in ('date', 'year_week') else 0, ha='right')
            return _save_and_return(ttl, plot_type, f"Time series by {x_choice} using {'mean of ' + y_column if agg_how != 'count' else 'counts'}.")

        elif plot_type == "temporal_bar":
            dt_col = _detect_datetime_column(df)
            if not _ensure_datetime(df, dt_col):
                cleanup_plot()
                return "Error: temporal_bar requires a datetime column."
            _add_temporal_features(df, dt_col)

            x_choice = x_column if x_column else ('date' if df.get('date', pd.Series()).nunique() >= 3 else 'hour')
            [x_choice] = _materialize_temporal_if_requested(df, dt_col, [x_choice])
            agg_how = 'mean' if (y_column and y_column in df.columns and pd.api.types.is_numeric_dtype(df[y_column])) else 'count'
            ser = _agg_series(df, by_col=x_choice, val_col=(y_column if agg_how != 'count' else None), how=agg_how)
            plt.figure(figsize=(12, 6))
            ser.sort_index().plot(kind='bar', color='slateblue', alpha=0.9)
            ttl = f"Temporal Bar by {x_choice.title()}"
            plt.title(ttl)
            plt.xlabel(x_choice.title())
            plt.ylabel('Value' if agg_how != 'count' else 'Count')
            plt.xticks(rotation=45 if x_choice in ('date', 'year_week') else 0, ha='right')
            return _save_and_return(ttl, plot_type, f"Bar aggregation by {x_choice} using {'mean of ' + y_column if agg_how != 'count' else 'counts'}.")

        elif plot_type == "temporal_heatmap":
            # Build 2D matrix from time bucket vs category/time bucket
            dt_col = _detect_datetime_column(df)
            have_dt = _ensure_datetime(df, dt_col)
            if have_dt:
                _add_temporal_features(df, dt_col)

            # Determine axes (prefer provided args; else sensible defaults)
            x_choice = x_column
            y_choice = y_column
            if have_dt:
                x_choice, y_choice = _materialize_temporal_if_requested(df, dt_col, [x_choice, y_choice])

            # If still missing, try to infer from existing columns without requiring datetime
            def _find_col_by_keywords(local_df: pd.DataFrame, keywords: list) -> str:
                cols = list(local_df.columns)
                lower = {c.lower(): c for c in cols}
                for kw in keywords:
                    for c in cols:
                        if kw in c.lower():
                            return c
                return None

            if not x_choice:
                x_choice = 'hour' if 'hour' in df.columns else _find_col_by_keywords(df, ['hour'])
            if not y_choice:
                # Prefer weekday/ day_of_week style if available
                y_choice = 'weekday' if 'weekday' in df.columns else (
                    _find_col_by_keywords(df, ['weekday', 'day_of_week', 'dayofweek', 'dow', 'wday'])
                )

            # If datetime was not available, materialize derived columns from common patterns
            if not have_dt:
                # Create 'hour' from any hour-like column if standard 'hour' is missing
                if x_choice and x_choice not in df.columns:
                    # Nothing to do
                    pass
                # Create standardized 'weekday' names if we have numeric codes
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                if y_choice and y_choice in df.columns:
                    ser = df[y_choice]
                    try:
                        # Normalize to 0-6 if values appear 1-7
                        if ser.dropna().astype(int).between(1, 7).all():
                            codes = ser.astype(int) - 1
                        else:
                            codes = ser.astype(int)
                        if 'weekday' not in df.columns:
                            df['weekday'] = codes
                        # Map to names for pretty axis
                        df['weekday_name'] = codes.map(lambda i: day_names[i] if 0 <= i <= 6 else str(i))
                        # Use names for y-axis
                        y_choice = 'weekday_name'
                    except Exception:
                        # If mapping fails, keep original column
                        pass

            # Final fallback defaults
            if not x_choice and not y_choice:
                x_choice, y_choice = ('hour' if 'hour' in df.columns else _find_col_by_keywords(df, ['hour'])), (
                    'weekday' if 'weekday' in df.columns else _find_col_by_keywords(df, ['weekday', 'day_of_week', 'dayofweek', 'dow', 'wday'])
                )
            if not x_choice or not y_choice or x_choice not in df.columns or y_choice not in df.columns:
                cleanup_plot()
                return f"Error: temporal_heatmap requires valid axes (got x='{x_choice}', y='{y_choice}')."

            # Build pivot and order axes nicely (weekday Monday→Sunday if present)
            pvt = _pivot_counts(df, index_col=y_choice, col_col=x_choice)
            # Reorder rows if they look like weekday names
            if set(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']).issuperset(set(pvt.index.astype(str))):
                desired = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
                pvt = pvt.reindex([d for d in desired if d in pvt.index])
            plt.figure(figsize=(14, max(6, 0.35 * len(pvt.index))))
            annot_flag = pvt.shape[0] <= 30 and pvt.shape[1] <= 24
            sns.heatmap(pvt, annot=annot_flag, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Count'})
            ttl = f"Temporal Heatmap: {y_choice.title()} vs {x_choice.title()}"
            plt.title(ttl)
            plt.xlabel(x_choice.title())
            plt.ylabel(y_choice.title())
            return _save_and_return(ttl, plot_type, f"Heatmap of counts by {y_choice} and {x_choice}.")

        elif plot_type == "temporal_matrix":
            # More general 2D matrix: if both x and y categorical/time-derived, build pivot; else default hour x weekday
            dt_col = _detect_datetime_column(df)
            if not _ensure_datetime(df, dt_col):
                cleanup_plot()
                return "Error: temporal_matrix requires a datetime column."
            _add_temporal_features(df, dt_col)

            x_choice = x_column
            y_choice = y_column
            x_choice, y_choice = _materialize_temporal_if_requested(df, dt_col, [x_choice, y_choice])
            if not x_choice or not y_choice:
                x_choice = x_choice or 'hour'
                y_choice = y_choice or 'weekday'
            if x_choice not in df.columns or y_choice not in df.columns:
                cleanup_plot()
                return f"Error: temporal_matrix requires valid axes (got x='{x_choice}', y='{y_choice}')."

            pvt = _pivot_counts(df, index_col=y_choice, col_col=x_choice)
            plt.figure(figsize=(14, max(6, 0.35 * len(pvt.index))))
            annot_flag = pvt.shape[0] <= 30 and pvt.shape[1] <= 24
            sns.heatmap(pvt, annot=annot_flag, fmt='d', cmap='viridis', cbar_kws={'label': 'Count'})
            ttl = f"Temporal Matrix: {y_choice.title()} vs {x_choice.title()}"
            plt.title(ttl)
            plt.xlabel(x_choice.title())
            plt.ylabel(y_choice.title())
            return _save_and_return(ttl, plot_type, f"2D count matrix across {y_choice} and {x_choice}.")

        else:
            cleanup_plot()
            return f"Error: Unknown plot type '{plot_type}'."

        # For non-temporal branches above that did not early-return, keep original save+return path
        plt.tight_layout()

        plot_file_path = None
        if output_dir:
            if not Path(output_dir).is_absolute():
                project_root = Path(__file__).parent.parent.parent.parent
                output_path = project_root / output_dir
            else:
                output_path = Path(output_dir)

            output_path.mkdir(parents=True, exist_ok=True)
            sanitized_title = "".join(c for c in title if c.isalnum() or c in (" ", ".", "_")).rstrip()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file_name = f"{sanitized_title.replace(' ', '_').replace('.', '')}_{plot_type}_{timestamp}.png"
            plot_file_path = output_path / plot_file_name
            plt.savefig(plot_file_path, dpi=150, bbox_inches="tight")

        cleanup_plot()

        if plot_file_path:
            return (
                f"![{title}]({plot_file_path.name})\n"
                f"Saved to: {plot_file_path}\n"
                f"Artifact: {plot_file_path.name}\n"
                f"Summary: Plot generated successfully."
            )
        else:
            return "Plot generated but not saved to disk.\nSummary: Plot generated successfully."

    except Exception as e:
        cleanup_plot()  # Ensure plot is closed even on error
        return f"Error generating plot: {str(e)}"


@mcp.tool()
def generate_correlation_heatmap(
    file_path: Annotated[str, Field(description="Path to the CSV file to analyze")],
    columns: Annotated[
        List[str],
        Field(
            description="Optional subset of columns to include. Non-numeric columns will be ignored."
        ),
    ] = None,
    method: Annotated[
        Literal["pearson", "spearman", "kendall"],
        Field(description="Correlation method to use"),
    ] = "pearson",
    top_k: Annotated[
        int,
        Field(
            description="If too many numeric columns, limit to top-K by variance to keep the heatmap readable.",
            ge=2,
        ),
    ] = 12,
    annotate: Annotated[
        bool,
        Field(description="Annotate heatmap cells with correlation values when matrix is reasonably small"),
    ] = True,
    mask_upper: Annotated[
        bool,
        Field(description="Mask the upper triangle to focus on unique pairs"),
    ] = True,
    threshold: Annotated[
        float,
        Field(description="Absolute correlation threshold for listing top pairs in the summary", ge=0.0, le=1.0),
    ] = 0.7,
    title: Annotated[
        str,
        Field(description="Title for the heatmap figure"),
    ] = "Correlation Heatmap",
    output_dir: Annotated[
        str,
        Field(
            description="Directory to save the heatmap image. Relative paths are resolved from project root."
        ),
    ] = "src/agents/analysis/reportDemo",
) -> str:
    """
    Generate a correlation heatmap for numeric columns with sensible defaults and guardrails.

    WHEN TO USE:
    - You need a correlation overview across numeric features
    - The general plotting tool no longer supports correlation heatmaps

    WHAT IT DOES:
    - Loads the CSV with cached I/O
    - Selects numeric columns (optionally restricted by 'columns')
    - Caps width by selecting top_k columns using variance when too wide
    - Computes correlation using the chosen method (pearson|spearman|kendall)
    - Renders a seaborn heatmap (optionally lower-triangle masked)
    - Saves PNG in output_dir and returns a Markdown link with a short relative filename

    RETURNS:
    - Markdown string including:
      • Image embed of the heatmap
      • Saved path
      • Summary line with method, column count, and up to a few strong pairs (|r| >= threshold)

    PREREQUISITES:
    - At least 2 numeric columns after filtering
    - If too many numeric columns, either pass a subset via 'columns' or rely on top_k selection
    """
    try:
        if not Path(file_path).exists():
            return f"Error: File '{file_path}' does not exist."

        df = load_csv_data_cached(file_path)

        # Filter to requested columns if provided
        if columns:
            existing = [c for c in columns if c in df.columns]
            missing = [c for c in columns if c not in df.columns]
            if not existing:
                return f"Error: None of the requested columns were found: {columns}"
            work_df = df[existing]
        else:
            work_df = df

        # Keep only numeric columns
        numeric_cols = work_df.select_dtypes(include=["number"]).columns.tolist()
        if len(numeric_cols) < 2:
            return "Error: Need at least two numeric columns to compute correlations."

        numeric_df = work_df[numeric_cols]

        # If too wide, select by variance to top_k
        if len(numeric_cols) > top_k:
            try:
                variances = numeric_df.var(numeric_only=True).sort_values(ascending=False)
                selected = variances.head(top_k).index.tolist()
                numeric_df = numeric_df[selected]
            except Exception:
                numeric_df = numeric_df.iloc[:, :top_k]

        # Compute correlation matrix
        corr = numeric_df.corr(method=method)

        # Plot
        n = corr.shape[0]
        fig_size = max(8, min(18, int(0.8 * n) + 6))
        plt.figure(figsize=(fig_size, fig_size))
        mask = None
        if mask_upper:
            mask = np.triu(np.ones_like(corr, dtype=bool))
        ann = bool(annotate and n <= 15)
        sns.heatmap(
            corr,
            mask=mask,
            cmap="coolwarm",
            annot=ann,
            fmt=".2f",
            square=True,
            cbar_kws={"label": "Correlation"},
        )
        full_title = f"{title} ({method.title()})"
        plt.title(full_title)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        # Save
        if not Path(output_dir).is_absolute():
            project_root = Path(__file__).parent.parent.parent.parent
            output_path = project_root / output_dir
        else:
            output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        sanitized_title = "".join(c for c in full_title if c.isalnum() or c in (" ", ".", "_")).rstrip()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{sanitized_title.replace(' ', '_').replace('.', '')}_heatmap_{method}_{timestamp}.png"
        file_path_out = output_path / file_name
        plt.tight_layout()
        plt.savefig(file_path_out, dpi=150, bbox_inches="tight")
        cleanup_plot()

        # Build concise summary with top strong pairs
        try:
            upper = corr.where(~np.tril(np.ones(corr.shape, dtype=bool)))
            pairs = []
            for i, c1 in enumerate(upper.index):
                for j, c2 in enumerate(upper.columns):
                    val = upper.iloc[i, j]
                    if pd.notna(val) and abs(val) >= float(threshold):
                        pairs.append((c1, c2, float(val)))
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            top_pairs_text = "; ".join([f"{a}↔{b}: {v:.2f}" for a, b, v in pairs[:8]]) if pairs else "none ≥ threshold"
        except Exception:
            top_pairs_text = "n/a"

        summary = (
            f"Method: {method}; Columns analyzed: {corr.shape[0]}. "
            f"Top pairs (|r| ≥ {threshold}): {top_pairs_text}"
        )

        return (
            f"![{full_title}]({file_path_out.name})\n"
            f"Saved to: {file_path_out}\n"
            f"Artifact: {file_path_out.name}\n"
            f"Summary: {summary}"
        )

    except Exception as e:
        cleanup_plot()
        return f"Error generating correlation heatmap: {str(e)}"


@mcp.tool()
def train_classification_model(
    file_path: Annotated[
        str, Field(description="Path to the CSV file containing data")
    ],
    target_column: Annotated[
        str,
        Field(description="Name of the target (dependent) column for classification"),
    ],
    feature_columns: Annotated[
        List[str], Field(description="List of independent feature columns")
    ],
    model_type: Annotated[
        Literal["random_forest"],
        Field(
            description="Type of classification model to train (e.g., 'random_forest')"
        ),
    ] = "random_forest",
    test_size: Annotated[
        float,
        Field(
            description="Proportion of the dataset to include in the test split (0.0 to 1.0)"
        ),
    ] = 0.2,
    random_state: Annotated[
        int, Field(description="Random state for reproducibility")
    ] = 42,
) -> str:
    """
    Train and evaluate a classification model on tabular data.

    Currently supports Random Forest classifier.
    Returns a Markdown report with accuracy, classification report, and confusion matrix.
    """
    try:
        if not Path(file_path).exists():
            return f"Error: File '{file_path}' does not exist."

        df = load_csv_data_cached(file_path)

        # Drop rows with any missing values in target or features for simplicity
        model_df = df[[target_column] + feature_columns].dropna()

        if model_df.empty:
            return "Error: No complete rows after dropping missing values for target and feature columns."

        if target_column not in model_df.columns or not all(
            col in model_df.columns for col in feature_columns
        ):
            return "Error: Target column or one or more feature columns not found after cleaning."

        X = model_df[feature_columns]
        y = model_df[target_column]

        # Handle categorical features within X
        categorical_features = X.select_dtypes(include=["object"]).columns
        if len(categorical_features) > 0:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            encoded_features = encoder.fit_transform(X[categorical_features])
            encoded_feature_names = encoder.get_feature_names_out(categorical_features)
            encoded_df = pd.DataFrame(
                encoded_features, columns=encoded_feature_names, index=X.index
            )

            X = pd.concat([X.drop(columns=categorical_features), encoded_df], axis=1)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if len(y.unique()) > 1 else None,
        )

        report = [
            f"## Classification Model Training: '{model_type}' for '{target_column}'"
        ]

        if model_type == "random_forest":
            model = RandomForestClassifier(random_state=random_state)
            report.append("- Model Type: Random Forest Classifier")
        else:
            return f"Error: Model type '{model_type}' not supported. Current options: 'random_forest'."

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)

        report.append(
            f"\n### Model Performance on Test Set (Test Size: {test_size * 100:.0f}%):"
        )
        report.append(f"- Accuracy: {accuracy:.4f}")

        report.append("\n### Classification Report:")
        report.append(pd.DataFrame(class_report).transpose().to_string())

        report.append("\n### Confusion Matrix:")
        report.append(pd.DataFrame(conf_matrix).to_string())

        return "\n".join(report)

    except Exception as e:
        return f"Error training classification model: {str(e)}"


@mcp.tool()
def assess_data_quality(
    file_path: Annotated[str, Field(description="Path to the CSV file to assess")],
) -> str:
    """
    Perform comprehensive data quality assessment on a CSV dataset.

    Returns detailed analysis of:
    - Missing values and patterns
    - Data type consistency
    - Duplicate records
    - Outliers detection
    - Column completeness
    - Data distribution issues
    """
    try:
        validate_file_and_columns(file_path)
        df = load_csv_data_cached(file_path)

        report = []
        report.append("## Data Quality Assessment Report")
        report.append(f"**Dataset**: {file_path}")
        report.append(f"**Shape**: {df.shape[0]:,} rows × {df.shape[1]} columns")
        report.append(
            f"**Memory Usage**: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        )

        # Missing values analysis
        missing_summary = df.isnull().sum()
        missing_pct = (missing_summary / len(df)) * 100

        report.append("\n### Missing Values Analysis")
        if missing_summary.sum() == 0:
            report.append(" No missing values detected")
        else:
            report.append(
                f" Found missing values in {(missing_summary > 0).sum()} columns:"
            )
            for col in missing_summary[missing_summary > 0].index:
                report.append(
                    f"- **{col}**: {missing_summary[col]:,} missing ({missing_pct[col]:.1f}%)"
                )

        # Data type analysis
        column_types = detect_column_types(df)
        type_metadata = column_types.get("_metadata", {})
        types_by_column = {
            key: value for key, value in column_types.items() if key != "_metadata"
        }
        report.append("\n### Data Type Analysis")
        for col in df.columns:
            detected_type = types_by_column.get(col, "unknown")
            current_type = str(df[col].dtype)
            if detected_type != "mixed":
                report.append(f"- **{col}**: {current_type} → {detected_type}")
            else:
                report.append(f"- **{col}**: {current_type}  Mixed types detected")

        high_cardinality_cols = type_metadata.get("high_cardinality", [])
        if high_cardinality_cols:
            report.append(
                "High-cardinality categorical columns: "
                + ", ".join(high_cardinality_cols)
            )

        # Duplicates analysis
        duplicate_rows = df.duplicated().sum()
        report.append("\n### Duplicate Records")
        if duplicate_rows == 0:
            report.append(" No duplicate rows found")
        else:
            duplicate_pct = (duplicate_rows / len(df)) * 100
            report.append(
                f" Found {duplicate_rows:,} duplicate rows ({duplicate_pct:.1f}%)"
            )

        # Outliers detection for numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        outlier_summary: Dict[str, int] = {}
        if len(numeric_cols) > 0:
            report.append("\n### Outliers Analysis (IQR Method)")
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                outlier_summary[col] = len(outliers)

                if len(outliers) > 0:
                    outlier_pct = (len(outliers) / len(df)) * 100
                    report.append(
                        f"- **{col}**: {len(outliers):,} outliers ({outlier_pct:.1f}%)"
                    )

            if sum(outlier_summary.values()) == 0:
                report.append(" No outliers detected in numeric columns")

        # Column completeness and uniqueness
        report.append("\n### Column Completeness & Uniqueness")
        for col in df.columns:
            completeness = ((len(df) - df[col].isnull().sum()) / len(df)) * 100
            unique_count = df[col].nunique()
            unique_pct = (unique_count / len(df)) * 100

            status = "GOOD" if completeness > 95 else "OK" if completeness > 80 else "POOR"
            report.append(
                f"- **{col}**: {status} {completeness:.1f}% complete, "
                f"{unique_count:,} unique values ({unique_pct:.1f}%)"
            )

        # Overall quality score
        quality_score = 100
        if missing_summary.sum() > 0:
            quality_score -= min(
                30, (missing_summary.sum() / (len(df) * len(df.columns))) * 100
            )
        if duplicate_rows > 0:
            quality_score -= min(20, (duplicate_rows / len(df)) * 100)
        if len(numeric_cols) > 0 and sum(outlier_summary.values()) > 0:
            quality_score -= min(15, (sum(outlier_summary.values()) / len(df)) * 100)

        report.append(f"\n### Overall Data Quality Score: {quality_score:.1f}/100")
        if quality_score >= 90:
            report.append("🟢 Excellent data quality")
        elif quality_score >= 70:
            report.append("🟡 Good data quality with minor issues")
        else:
            report.append(" Poor data quality - needs attention")

        return "\n".join(report)

    except Exception as e:
        return f"Error assessing data quality: {str(e)}"


@mcp.tool()
def perform_automated_statistical_tests(
    file_path: Annotated[str, Field(description="Path to the CSV file to analyze")],
    significance_level: Annotated[
        float, Field(description="Significance level for tests (default 0.05)")
    ] = 0.05,
) -> str:
    """
    Automatically perform appropriate statistical tests based on data types and distributions.

    Includes:
    - Normality tests for numeric columns
    - Correlation significance tests
    - Independence tests for categorical variables
    - Homogeneity of variance tests
    - Appropriate parametric/non-parametric test selection
    """
    try:
        validate_file_and_columns(file_path)
        df = load_csv_data_cached(file_path)

        report = []
        report.append("## Automated Statistical Testing Report")
        report.append(f"**Significance Level**: {significance_level}")
        report.append(f"**Dataset**: {file_path} ({len(df)} rows)")

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # Normality tests for numeric columns
        if len(numeric_cols) > 0:
            report.append("\n### Normality Tests (Shapiro-Wilk)")
            for col in numeric_cols:
                clean_data = df[col].dropna()
                if len(clean_data) >= 3:
                    # Use a sample if data is too large for Shapiro-Wilk
                    test_data = clean_data.sample(
                        min(5000, len(clean_data)), random_state=42
                    )

                    from scipy.stats import shapiro

                    stat, p_value = shapiro(test_data)

                    is_normal = p_value > significance_level
                    result = " Normal" if is_normal else " Non-normal"
                    report.append(f"- **{col}**: {result} (p={p_value:.4f})")

                    if not is_normal:
                        # Suggest transformations
                        if (clean_data > 0).all():
                            report.append("  - Consider log transformation")
                        report.append("  - Use non-parametric tests")
                else:
                    report.append(f"- **{col}**: Insufficient data for testing")

        # Correlation significance tests
        if len(numeric_cols) >= 2:
            report.append("\n### Correlation Significance Tests")
            from scipy.stats import pearsonr, spearmanr

            significant_correlations = []
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    clean_data = df[[col1, col2]].dropna()

                    if len(clean_data) >= 3:
                        # Pearson correlation
                        pearson_r, pearson_p = pearsonr(
                            clean_data[col1], clean_data[col2]
                        )
                        # Spearman correlation
                        spearman_r, spearman_p = spearmanr(
                            clean_data[col1], clean_data[col2]
                        )

                        if pearson_p < significance_level:
                            significant_correlations.append(
                                f"- **{col1} ↔ {col2}**: "
                                f"Pearson r={pearson_r:.3f} (p={pearson_p:.4f}) "
                            )

                        if (
                            spearman_p < significance_level
                            and abs(spearman_r - pearson_r) > 0.1
                        ):
                            significant_correlations.append(
                                f"- **{col1} ↔ {col2}**: "
                                f"Spearman ρ={spearman_r:.3f} (p={spearman_p:.4f}) "
                            )

            if significant_correlations:
                report.append("**Significant Correlations Found:**")
                report.extend(significant_correlations)
            else:
                report.append(
                    "No significant correlations found at the specified level."
                )

        # Chi-square tests for categorical variables
        if len(categorical_cols) >= 2:
            report.append("\n### Independence Tests (Chi-square)")
            from scipy.stats import chi2_contingency

            for i in range(len(categorical_cols)):
                for j in range(i + 1, len(categorical_cols)):
                    col1, col2 = categorical_cols[i], categorical_cols[j]

                    # Create contingency table
                    contingency_table = pd.crosstab(df[col1], df[col2])

                    # Check if we have enough data
                    if (
                        contingency_table.sum().sum() >= 20
                        and (contingency_table >= 5).all().all()
                    ):
                        chi2, p_value, dof, expected = chi2_contingency(
                            contingency_table
                        )

                        is_independent = p_value > significance_level
                        result = "Independent" if is_independent else "Dependent"
                        status = "" if is_independent else ""

                        report.append(
                            f"- **{col1} × {col2}**: {status} {result} "
                            f"(χ²={chi2:.3f}, p={p_value:.4f})"
                        )
                    else:
                        report.append(
                            f"- **{col1} × {col2}**: Insufficient data for reliable test"
                        )

        # Variance homogeneity tests
        if len(numeric_cols) > 0 and len(categorical_cols) > 0:
            report.append("\n### Homogeneity of Variance Tests (Levene)")
            from scipy.stats import levene

            for num_col in numeric_cols[:3]:  # Limit to avoid too many tests
                for cat_col in categorical_cols[:2]:
                    groups = [
                        group[num_col].dropna()
                        for name, group in df.groupby(cat_col)
                        if len(group[num_col].dropna()) >= 3
                    ]

                    if len(groups) >= 2:
                        stat, p_value = levene(*groups)

                        equal_var = p_value > significance_level
                        result = "Equal variances" if equal_var else "Unequal variances"
                        status = "" if equal_var else ""

                        report.append(
                            f"- **{num_col} by {cat_col}**: {status} {result} "
                            f"(p={p_value:.4f})"
                        )

                        if not equal_var:
                            report.append(
                                "  - Use Welch's t-test or non-parametric alternatives"
                            )

        # Test recommendations summary
        report.append("\n### Statistical Test Recommendations")

        if len(numeric_cols) >= 2:
            report.append("**For comparing means:**")
            report.append("- If normal + equal variance: Independent t-test")
            report.append("- If normal + unequal variance: Welch's t-test")
            report.append("- If non-normal: Mann-Whitney U test")

        if len(categorical_cols) >= 2:
            report.append("**For categorical associations:**")
            report.append("- Chi-square test of independence (if assumptions met)")
            report.append("- Fisher's exact test (for small samples)")

        if len(numeric_cols) > 2:
            report.append("**For multiple comparisons:**")
            report.append("- ANOVA (if normal + equal variance)")
            report.append("- Kruskal-Wallis test (non-parametric alternative)")
            report.append("- Apply multiple comparison corrections (Bonferroni, FDR)")

        return "\n".join(report)

    except Exception as e:
        return f"Error performing automated statistical tests: {str(e)}"


@mcp.tool()
def analyze_temporal_patterns(
    file_path: Annotated[str, Field(description="Path to the CSV file with temporal data")],
    datetime_column: Annotated[str, Field(description="Name of the datetime column to analyze")],
    incident_type_column: Annotated[str, Field(description="Optional column for incident types/categories")] = None,
    output_dir: Annotated[
        str, Field(description="Directory to save temporal analysis plots")
    ] = "src/agents/analysis/reportDemo",
) -> str:
    """
    Perform comprehensive temporal pattern analysis with visualizations.

    WHEN TO USE THIS TOOL:
    - Use when dataset contains datetime/timestamp columns
    - Use to discover time-based patterns (rush hours, peak days)
    - Use to identify temporal trends and seasonality
    - Use AFTER load_and_analyze_csv confirms datetime column exists

    WHAT THIS TOOL DOES:
    - Analyzes distribution by hour of day (identifies peak hours)
    - Analyzes distribution by day of week (identifies busy days)
    - Creates hour vs day-of-week heatmap (identifies peak time windows)
    - Detects monthly/seasonal patterns if data spans multiple months
    - Generates high-quality visualizations (saved as PNG files)

    RETURNS: Markdown report with:
    - Peak hour analysis (busiest and quietest hours)
    - Day-of-week patterns (busiest and quietest days)
    - Heatmap showing hour-day intersection patterns
    - Monthly trends (if applicable)
    - Embedded visualization references

    OUTPUT FILES CREATED:
    - hourly_distribution_[timestamp].png
    - daily_distribution_[timestamp].png
    - hour_dayofweek_heatmap_[timestamp].png

    PREREQUISITES: Requires datetime column (use load_and_analyze_csv to identify)
    """
    try:
        validate_file_and_columns(file_path, [datetime_column])
        df = load_csv_data_cached(file_path)
        
        if datetime_column not in df.columns:
            return f"Error: Column '{datetime_column}' not found in dataset."
        
        # Convert to datetime if needed
        try:
            df[datetime_column] = pd.to_datetime(df[datetime_column])
        except Exception as e:
            return f"Error converting '{datetime_column}' to datetime: {str(e)}"
        
        # Extract temporal components
        df['hour'] = df[datetime_column].dt.hour
        df['day_of_week'] = df[datetime_column].dt.day_name()
        df['day_of_week_num'] = df[datetime_column].dt.dayofweek
        df['month'] = df[datetime_column].dt.month
        df['date'] = df[datetime_column].dt.date
        
        results = []
        results.append("# Temporal Pattern Analysis Report")
        results.append(f"**Dataset**: {file_path}")
        results.append(f"**Datetime Column**: {datetime_column}")
        results.append(f"**Date Range**: {df[datetime_column].min()} to {df[datetime_column].max()}")
        results.append(f"**Total Records**: {len(df):,}")
        
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Hourly Distribution Analysis
        hourly_counts = df['hour'].value_counts().sort_index()
        
        # VISUALIZATION
        # create_optimized_plot()
        # plt.figure(figsize=(12, 6))
        # hourly_counts.plot(kind='bar', color='skyblue', alpha=0.7)
        # plt.title('Distribution of Incidents by Hour of Day')
        # plt.xlabel('Hour of Day')
        # plt.ylabel('Number of Incidents')
        # plt.xticks(rotation=0)
        # plt.grid(axis='y', alpha=0.3)
        
        # Add statistics
        peak_hour = hourly_counts.idxmax()
        peak_count = hourly_counts.max()
        min_hour = hourly_counts.idxmin()
        min_count = hourly_counts.min()
        
        # plt.text(0.7, 0.9, f'Peak: {peak_hour}:00 ({peak_count:,} incidents)\nLowest: {min_hour}:00 ({min_count:,} incidents)', 
        #         transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # hourly_plot_path = output_path / f"hourly_distribution_{timestamp}.png"
        # plt.tight_layout()
        # plt.savefig(hourly_plot_path, dpi=300, bbox_inches='tight')
        # plt.close()
        
        results.append(f"\n## 1. Hourly Distribution Analysis")
        results.append(f"- **Peak Hour**: {peak_hour}:00 with {peak_count:,} incidents ({peak_count/len(df)*100:.1f}%)")
        results.append(f"- **Lowest Hour**: {min_hour}:00 with {min_count:,} incidents ({min_count/len(df)*100:.1f}%)")
        results.append(f"- **Peak vs Lowest Ratio**: {peak_count/min_count:.1f}x")
        # results.append(f"- **Visualization**: ![Hourly Distribution]({hourly_plot_path.name})")
        
        # 2. Day of Week Distribution
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_counts = df['day_of_week'].value_counts().reindex(day_order)
        
        # VISUALIZATION DISABLED - keeping only statistical analysis
        # plt.figure(figsize=(10, 6))  
        # daily_counts.plot(kind='bar', color='lightcoral', alpha=0.7)
        # plt.title('Distribution of Incidents by Day of Week')
        # plt.xlabel('Day of Week')
        # plt.ylabel('Number of Incidents')
        # plt.xticks(rotation=45)
        # plt.grid(axis='y', alpha=0.3)
        
        peak_day = daily_counts.idxmax()
        peak_day_count = daily_counts.max()
        min_day = daily_counts.idxmin()
        min_day_count = daily_counts.min()
        
        # plt.text(0.6, 0.9, f'Peak: {peak_day} ({peak_day_count:,})\nLowest: {min_day} ({min_day_count:,})', 
        #         transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # daily_plot_path = output_path / f"daily_distribution_{timestamp}.png"  
        # plt.tight_layout()
        # plt.savefig(daily_plot_path, dpi=300, bbox_inches='tight')
        # plt.close()
        
        results.append(f"\n## 2. Day of Week Distribution Analysis")
        results.append(f"- **Peak Day**: {peak_day} with {peak_day_count:,} incidents ({peak_day_count/len(df)*100:.1f}%)")
        results.append(f"- **Lowest Day**: {min_day} with {min_day_count:,} incidents ({min_day_count/len(df)*100:.1f}%)")
        results.append(f"- **Weekday vs Weekend Pattern**: {'Weekday-heavy' if peak_day not in ['Saturday', 'Sunday'] else 'Weekend-heavy'}")
        # results.append(f"- **Visualization**: ![Daily Distribution]({daily_plot_path.name})")
        
        # 3. Hour vs Day of Week Heatmap
        heatmap_data = df.groupby(['day_of_week_num', 'hour']).size().unstack(fill_value=0)
        heatmap_data.index = day_order
        
        # VISUALIZATION DISABLED - keeping only statistical analysis
        # plt.figure(figsize=(14, 8))
        # sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Number of Incidents'})
        # plt.title('Incidents Heatmap: Hour of Day vs Day of Week')
        # plt.xlabel('Hour of Day')
        # plt.ylabel('Day of Week')
        
        # heatmap_path = output_path / f"temporal_heatmap_{timestamp}.png"
        # plt.tight_layout()
        # plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        # plt.close()
        
        # Find peak time combination
        peak_combo = heatmap_data.stack().idxmax()
        peak_combo_count = heatmap_data.stack().max()
        
        results.append(f"\n## 3. Temporal Pattern Heatmap")
        results.append(f"- **Peak Time Combination**: {peak_combo[0]} at {peak_combo[1]}:00 with {peak_combo_count:,} incidents")
        results.append(f"- **Pattern**: Time-based clustering analysis reveals concentrated incident periods")
        # results.append(f"- **Visualization**: ![Temporal Heatmap]({heatmap_path.name})")
        
        # 4. Incident Type Temporal Analysis (if column provided)
        if incident_type_column and incident_type_column in df.columns:
            results.append(f"\n## 4. Incident Type Temporal Analysis")
            
            # Top incident types by hour
            type_hour_pivot = df.groupby([incident_type_column, 'hour']).size().unstack(fill_value=0)
            top_types = df[incident_type_column].value_counts().head(5).index
            
            # VISUALIZATION DISABLED - keeping only statistical analysis
            # plt.figure(figsize=(14, 10))
            # for i, incident_type in enumerate(top_types):
            #     plt.subplot(3, 2, i+1)
            #     if incident_type in type_hour_pivot.index:
            #         type_hour_pivot.loc[incident_type].plot(kind='bar', alpha=0.7)
            #         plt.title(f'{incident_type} - Hourly Distribution')
            #         plt.xlabel('Hour')
            #         plt.ylabel('Count')
            #         plt.xticks(rotation=0)
            #         
            # plt.tight_layout()
            # type_temporal_path = output_path / f"incident_types_temporal_{timestamp}.png"
            # plt.savefig(type_temporal_path, dpi=300, bbox_inches='tight')
            # plt.close()
            
            results.append(f"- **Top 5 Incident Types Temporal Patterns**: Analyzed hourly distributions")
            # results.append(f"- **Visualization**: ![Incident Types Temporal]({type_temporal_path.name})")
            
            # Incident type summary
            for incident_type in top_types[:3]:
                if incident_type in type_hour_pivot.index:
                    type_peak_hour = type_hour_pivot.loc[incident_type].idxmax()
                    type_peak_count = type_hour_pivot.loc[incident_type].max()
                    results.append(f"  - **{incident_type}**: Peak at {type_peak_hour}:00 ({type_peak_count:,} incidents)")
        
        # 5. Statistical Insights
        results.append(f"\n## 5. Key Statistical Insights")
        
        # Hour statistics
        hour_std = hourly_counts.std()
        hour_mean = hourly_counts.mean()
        hour_cv = hour_std / hour_mean
        
        results.append(f"- **Hourly Variation**: Coefficient of variation = {hour_cv:.2f} ({'High' if hour_cv > 0.5 else 'Moderate' if hour_cv > 0.3 else 'Low'} variability)")
        
        # Day statistics  
        daily_std = daily_counts.std()
        daily_mean = daily_counts.mean()
        daily_cv = daily_std / daily_mean
        
        results.append(f"- **Daily Variation**: Coefficient of variation = {daily_cv:.2f} ({'High' if daily_cv > 0.3 else 'Moderate' if daily_cv > 0.15 else 'Low'} variability)")
        
        # Weekend vs Weekday
        weekend_count = df[df['day_of_week'].isin(['Saturday', 'Sunday'])].shape[0]
        weekday_count = len(df) - weekend_count
        weekend_pct = weekend_count / len(df) * 100
        
        results.append(f"- **Weekend vs Weekday**: {weekend_pct:.1f}% weekend incidents, {100-weekend_pct:.1f}% weekday incidents")
        
        cleanup_plot()
        return "\n".join(results)
        
    except Exception as e:
        cleanup_plot()
        return f"Error in temporal analysis: {str(e)}"


@mcp.tool()
def analyze_text_content(
    file_path: Annotated[str, Field(description="Path to the CSV file")],
    text_column: Annotated[str, Field(description="Column containing text to analyze")],
    analysis_purpose: Annotated[str, Field(description="What you want to understand from the text (e.g., 'find extractable information', 'understand content structure')")] = "general analysis"
) -> str:
    """
    Analyze text column content to identify patterns and extraction opportunities.

    WHEN TO USE THIS TOOL:
    - Use when dataset contains text/string columns beyond simple categories
    - Use to discover what information is hidden in text fields
    - Use BEFORE extract_structured_data_from_text to plan extraction
    - Use when you see high-cardinality text columns in load_and_analyze_csv output

    WHAT THIS TOOL DOES:
    - Analyzes text structure and consistency
    - Identifies common terms and patterns
    - Determines if text is structured or free-form
    - Suggests extraction strategies
    - Returns IMMEDIATE analytical insights ready for synthesis

    RETURNS: Markdown report with:
    - Text statistics (entry count, average length, consistency)
    - Content structure analysis (structured vs free-form)
    - Sample content (first 5 entries)
    - Common terms and frequency
    - Extraction recommendations
    - IMMEDIATE ANALYTICAL INSIGHTS section with actionable intelligence

    USE CASE EXAMPLES:
    - Traffic incident descriptions → Extract road names, directions, locations
    - Customer feedback → Extract sentiment, key issues
    - Error logs → Extract error types, components

    PREREQUISITES: Text column must exist (check with load_and_analyze_csv)
    NEXT STEPS: If extraction potential found, use extract_structured_data_from_text
    """
    try:
        validate_file_and_columns(file_path, [text_column])
        df = load_csv_data_cached(file_path)
        
        if text_column not in df.columns:
            return f"Error: Column '{text_column}' not found in dataset."
        
        text_data = df[text_column].dropna()
        if len(text_data) == 0:
            return f"Error: No valid text data found in column '{text_column}'."
        
        # Basic text analysis
        total_entries = len(text_data)
        avg_length = text_data.str.len().mean()
        length_std = text_data.str.len().std()
        
        # Sample analysis
        sample_size = min(20, len(text_data))
        text_samples = text_data.head(sample_size).tolist()
        
        # Pattern detection
        has_consistent_structure = length_std < avg_length * 0.5 if avg_length > 0 else False
        
        # Common word analysis
        all_words = ' '.join(text_samples).lower().split()
        from collections import Counter
        common_words = Counter(all_words).most_common(10)
        
        results = []
        results.append(f"# Text Content Analysis: {text_column}")
        results.append(f"**Analysis Purpose**: {analysis_purpose}")
        results.append(f"**Dataset**: {file_path}")
        results.append("")
        
        results.append("## Text Statistics")
        results.append(f"- **Total Text Entries**: {total_entries:,}")
        results.append(f"- **Average Text Length**: {avg_length:.1f} characters")
        results.append(f"- **Length Consistency**: {'High' if has_consistent_structure else 'Variable'}")
        results.append(f"- **Sample Size Analyzed**: {sample_size}")
        results.append("")
        
        results.append("## Content Structure Analysis")
        if has_consistent_structure:
            results.append("- **Pattern Detection**: Text appears to have consistent structure - good candidate for extraction")
        else:
            results.append("- **Pattern Detection**: Text has variable structure - may contain free-form content")
        results.append("")
        
        results.append("## Sample Content")
        results.append("```")
        for i, sample in enumerate(text_samples[:5], 1):
            results.append(f"{i}. {sample[:100]}{'...' if len(sample) > 100 else ''}")
        results.append("```")
        results.append("")
        
        results.append("## Common Terms")
        for word, count in common_words:
            if len(word) > 2:  # Skip very short words
                results.append(f"- **{word}**: appears {count} times")
        results.append("")
        
        results.append("## Extraction Recommendations")
        if has_consistent_structure:
            results.append("- Text structure suggests extractable patterns are present")
            results.append("- Consider using `extract_structured_data_from_text` tool")
            results.append("- Potential for creating additional structured columns")
        else:
            results.append("- Text appears more free-form - focus on entity extraction")
            results.append("- May benefit from keyword or sentiment analysis")
        results.append("")
        
        # CRITICAL: Provide immediate analytical insights for LLM synthesis
        results.append("## IMMEDIATE ANALYTICAL INSIGHTS")
        results.append("**Use these insights to enhance your current data analysis:**")
        results.append("")
        
        # Pattern analysis for analytical synthesis
        results.append("### Text Patterns for Cross-Analysis")
        if has_consistent_structure:
            results.append("- **Structured Text Detected**: High potential for extracting categorical variables")
            results.append("- **Analysis Opportunity**: Extract categories and correlate with numerical data")
            results.append("- **Recommendation**: Use extracted categories to segment statistical analysis")
        else:
            results.append("- **Variable Text Detected**: Focus on entity and keyword extraction")
            results.append("- **Analysis Opportunity**: Extract entities and study their distribution patterns")
            results.append("- **Recommendation**: Use text content to add context to numerical findings")
        results.append("")
        
        # Content intelligence for analytical depth
        results.append("### Content Intelligence Summary")
        results.append(f"- **Dominant Terms**: {', '.join([word for word, count in common_words[:5] if len(word) > 2])}")
        results.append(f"- **Text Complexity**: {'Low' if avg_length < 50 else 'Medium' if avg_length < 150 else 'High'}")
        results.append(f"- **Information Density**: {'High' if length_std < avg_length * 0.3 else 'Medium' if length_std < avg_length * 0.7 else 'Variable'}")
        results.append("")
        
        results.append("### Analytical Synthesis Opportunities")
        results.append("- **Cross-Reference**: Compare text patterns with numerical distributions in other columns")
        results.append("- **Segmentation**: Use text content to create analytical segments for deeper insights")
        results.append("- **Context Enhancement**: Use text insights to explain statistical anomalies or trends")
        results.append("- **Pattern Discovery**: Look for correlations between text characteristics and numerical patterns")
        
        return "\n".join(results)
        
    except Exception as e:
        return f"Error analyzing text content: {str(e)}"


@mcp.tool()
def extract_structured_data_from_text(
    file_path: Annotated[str, Field(description="Path to the CSV file")],
    text_column: Annotated[str, Field(description="Column containing text to extract from")],
    extract_instruction: Annotated[str, Field(description="What to extract from the text (e.g., 'extract times and locations', 'extract key entities and values')")] = "extract road locations and operational details",
    output_dir: Annotated[str, Field(description="Directory to save enriched dataset")] = "src/agents/analysis/reportDemo"
) -> str:
    """
    Extract structured data from unstructured text columns using intelligent pattern recognition.

    WHEN TO USE THIS TOOL:
    - Use AFTER analyze_text_content confirms extraction potential
    - Use when text contains hidden categorical or numerical data
    - Use to enrich dataset with new structured columns for deeper analysis
    - Use when you want to convert free-text into analyzable categories

    WHAT THIS TOOL DOES:
    - Applies intelligent pattern recognition to extract specific information
    - Creates NEW structured columns from text content
    - Handles domain-specific extractions (traffic roads, locations, times, etc.)
    - Saves enriched dataset as new CSV file for further analysis

    DOMAIN-SPECIFIC INTELLIGENCE:
    For traffic/incident data automatically extracts:
    - Road names (AYE, PIE, CTE, BKE, ECP, KPE, KJE, SLE, TPE, MCE)
    - Directions (towards MCE, towards Changi, etc.)
    - Specific locations (exits, entrances, junctions, intersections)
    - Times and operational details (lane closures, diversions)

    EXTRACTION INSTRUCTION EXAMPLES:
    - "extract road names and locations" → Creates Road_Name, Direction columns
    - "extract times and directions" → Creates Time, Direction columns
    - "extract operational details and lane information" → Creates Operation_Type, Lane_Info columns
    - "extract entities and numerical values" → Creates Entity, Value columns

    RETURNS: Markdown report with:
    - Extraction statistics (success rate, new columns created)
    - Sample of extracted data
    - Analysis of extraction quality
    - Path to enriched CSV file

    OUTPUT FILES CREATED:
    - [original_filename]_text_extracted_[timestamp].csv (enriched dataset)

    PREREQUISITES: Use analyze_text_content first to confirm extraction feasibility
    NEXT STEPS: Use enriched CSV file for correlation analysis with new columns
    """
    try:
        validate_file_and_columns(file_path, [text_column])
        df = load_csv_data_cached(file_path)
        
        if text_column not in df.columns:
            return f"Error: Column '{text_column}' not found in dataset."
        
        text_data = df[text_column].dropna()
        if len(text_data) == 0:
            return f"Error: No valid text data found in column '{text_column}'."
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Sample text for pattern analysis
        sample_texts = text_data.head(10).tolist()
        
        # Enhanced pattern-based extraction with Singapore road intelligence
        extracted_data = {}
        
        # Singapore road intelligence
        singapore_roads = {
            'AYE': 'Ayer Rajah Expressway',
            'PIE': 'Pan Island Expressway', 
            'CTE': 'Central Expressway',
            'BKE': 'Bukit Timah Expressway',
            'ECP': 'East Coast Parkway',
            'KPE': 'Kallang-Paya Lebar Expressway',
            'KJE': 'Kranji Expressway',
            'SLE': 'Seletar Expressway',
            'TPE': 'Tampines Expressway',
            'MCE': 'Marina Coastal Expressway'
        }
        
        # Enhanced extraction logic with traffic data intelligence - PRIORITIZE ROAD EXTRACTION
        if (('operational' in extract_instruction.lower() or 
            'road' in extract_instruction.lower() or
            'location' in extract_instruction.lower() or
            'meaningful' in extract_instruction.lower()) and 
            ('traffic' in str(df.columns).lower() or
             'message' in str(df.columns).lower() or
             'incident' in str(df.columns).lower())):
            # Extract road names and locations (not incident types)
            import re
            
            road_locations = []
            directions = []
            specific_locations = []
            
            for text in df[text_column].fillna(''):
                # Extract road + direction + location
                road_match = re.search(r'on ([A-Z]{2,4})\s*\(([^)]+)\)\s*([^.]+)', text)
                if road_match:
                    road_code = road_match.group(1)
                    direction = road_match.group(2)
                    location_detail = road_match.group(3).strip()
                    
                    # Build meaningful location string
                    road_name = singapore_roads.get(road_code, road_code)
                    full_location = f"{road_name} ({direction}) {location_detail}"
                    road_locations.append(full_location)
                    directions.append(direction)
                    specific_locations.append(location_detail)
                else:
                    road_locations.append(None)
                    directions.append(None) 
                    specific_locations.append(None)
            
            extracted_data['road_location'] = road_locations
            extracted_data['direction'] = directions
            extracted_data['specific_location'] = specific_locations
        
        # Time extraction (only if NOT traffic/road related)
        elif 'time' in extract_instruction.lower() and not any(word in extract_instruction.lower() for word in ['road', 'location', 'operational']):
            import re
            time_pattern = r'\b(\d{1,2}):(\d{2})\b'
            extracted_times = []
            for text in df[text_column].fillna(''):
                time_match = re.search(time_pattern, text)
                extracted_times.append(time_match.group() if time_match else None)
            extracted_data['extracted_time'] = extracted_times
        
        # Location/entity extraction for non-traffic data
        elif any(word in extract_instruction.lower() for word in ['location', 'place', 'entity']):
            # Extract capitalized words that might be locations/entities
            import re
            entity_pattern = r'\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\b'
            extracted_entities = []
            for text in df[text_column].fillna(''):
                entities = re.findall(entity_pattern, text)
                # Take first significant entity (longer than 2 chars)
                significant_entities = [e for e in entities if len(e) > 2]
                extracted_entities.append(significant_entities[0] if significant_entities else None)
            extracted_data['extracted_entity'] = extracted_entities
        
        # Number extraction
        if any(word in extract_instruction.lower() for word in ['number', 'value', 'amount']):
            import re
            number_pattern = r'\b\d+(?:\.\d+)?\b'
            extracted_numbers = []
            for text in df[text_column].fillna(''):
                numbers = re.findall(number_pattern, text)
                extracted_numbers.append(float(numbers[0]) if numbers else None)
            extracted_data['extracted_number'] = extracted_numbers
        
        # Add extracted columns to dataframe
        enriched_df = df.copy()
        for col_name, values in extracted_data.items():
            enriched_df[col_name] = values
        
        # Save enriched dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_path / f"enriched_data_{timestamp}.csv"
        enriched_df.to_csv(output_file, index=False)
        
        # Generate analysis with extracted data directly available for LLM context
        results = []
        results.append(f"# Text Extraction Results: {text_column}")
        results.append(f"**Extraction Instruction**: {extract_instruction}")
        results.append(f"**Original Dataset**: {file_path}")
        results.append(f"**Enriched Dataset**: {output_file}")
        results.append("")
        
        results.append("## Extraction Summary")
        results.append(f"- **Original Columns**: {len(df.columns)}")
        results.append(f"- **New Columns Added**: {len(extracted_data)}")
        results.append(f"- **Total Rows Processed**: {len(df):,}")
        results.append("")
        
        if extracted_data:
            results.append("## Extracted Columns")
            for col_name, values in extracted_data.items():
                non_null_count = sum(1 for v in values if v is not None)
                success_rate = (non_null_count / len(values)) * 100
                results.append(f"- **{col_name}**: {non_null_count:,} values extracted ({success_rate:.1f}% success rate)")
            results.append("")
            
            results.append("## Sample Extracted Data")
            for i in range(min(5, len(df))):
                results.append(f"**Row {i+1}:**")
                results.append(f"- Original: {df[text_column].iloc[i][:100]}...")
                for col_name in extracted_data.keys():
                    value = enriched_df[col_name].iloc[i]
                    results.append(f"- {col_name}: {value}")
                results.append("")
            
            # CRITICAL: Provide extracted insights directly for LLM analytical synthesis
            results.append("## EXTRACTED INSIGHTS FOR IMMEDIATE ANALYSIS")
            results.append("**Use these extracted insights to enhance your analytical understanding:**")
            results.append("")
            
            for col_name, values in extracted_data.items():
                # Provide statistical summary of extracted data
                non_null_values = [v for v in values if v is not None]
                if non_null_values:
                    results.append(f"### {col_name.title().replace('_', ' ')}")
                    
                    if isinstance(non_null_values[0], str):
                        # For text data, show unique values and frequency
                        from collections import Counter
                        value_counts = Counter(non_null_values)
                        results.append(f"- **Total Unique Values**: {len(value_counts)}")
                        results.append("- **Most Common Values**:")
                        for value, count in value_counts.most_common(10):
                            percentage = (count / len(non_null_values)) * 100
                            results.append(f"  - {value}: {count} occurrences ({percentage:.1f}%)")
                    else:
                        # For numeric data, show basic stats
                        import statistics
                        results.append(f"- **Range**: {min(non_null_values)} to {max(non_null_values)}")
                        results.append(f"- **Mean**: {statistics.mean(non_null_values):.2f}")
                        if len(non_null_values) > 1:
                            results.append(f"- **Std Dev**: {statistics.stdev(non_null_values):.2f}")
                    results.append("")
            
            results.append("## ANALYTICAL SYNTHESIS OPPORTUNITIES")
            results.append("**Cross-reference these extracted insights with your structured data analysis:**")
            results.append("- Look for patterns between extracted locations/entities and numerical distributions")
            results.append("- Identify temporal correlations if time data was extracted")
            results.append("- Use extracted categories to segment your statistical analysis")
            results.append("- Create visualizations that combine structured and text-derived dimensions")
            results.append("")
            
        else:
            results.append("## Extraction Results")
            results.append("- No specific extraction patterns were applied")
            results.append("- Consider refining the extraction instruction")
            results.append("- Try more specific requests like 'extract times' or 'extract locations'")
        
        results.append("## Next Steps")
        results.append("- **IMMEDIATE**: Use the extracted insights above to enhance your current analysis")
        results.append("- **SYNTHESIS**: Cross-reference extracted patterns with statistical findings")
        results.append("- **VISUALIZATION**: Create plots that incorporate both structured and extracted data")
        results.append("- **OPTIONAL**: Use enriched dataset file for further tool-based analysis")

        # Attach a machine-readable footer for downstream automation
        try:
            footer = {
                "type": "dataset",
                "source": "extract_structured_data_from_text",
                "input_path": file_path,
                "output_path": str(output_file),
                "extracted_columns": list(extracted_data.keys()),
            }
            results.append("")
            results.append(f"<!--output_json:{json.dumps(footer)}-->")
        except Exception:
            # Footer is optional; ignore failures
            pass

        return "\n".join(results)
        
    except Exception as e:
        return f"Error extracting structured data from text: {str(e)}"


@mcp.tool()
def merge_text_extracted_data(
    file_path: Annotated[str, Field(description="Path to the CSV file with text-extracted data")],
    text_column: Annotated[str, Field(description="Original text column that was analyzed")],
    merge_strategy: Annotated[str, Field(description="How to handle conflicts: 'prefer_structured', 'prefer_extracted', 'validate_both'")] = "prefer_structured",
    output_dir: Annotated[str, Field(description="Directory to save final dataset")] = "src/agents/analysis/reportDemo"
) -> str:
    """
    Merge text-extracted data with existing structured columns intelligently.
    
    This tool handles conflicts between structured data and text-extracted data:
    - 'prefer_structured': Use existing structured data when conflicts occur
    - 'prefer_extracted': Use text-extracted data when conflicts occur  
    - 'validate_both': Flag conflicts for review and use most reliable source
    
    Creates a final enriched dataset ready for analysis.
    """
    try:
        validate_file_and_columns(file_path, [text_column])
        df = load_csv_data_cached(file_path)
        
        if text_column not in df.columns:
            return f"Error: Column '{text_column}' not found in dataset."
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Identify extracted columns
        # Primary convention: columns starting with 'extracted_'
        extracted_columns = [col for col in df.columns if col.startswith('extracted_')]
        original_columns = [col for col in df.columns if not col.startswith('extracted_')]

        # Compatibility: some extractors add well-known columns without the prefix
        # e.g., road_location, direction, specific_location (traffic text extraction)
        non_prefixed_candidates = [
            'road_location', 'direction', 'specific_location'
        ]
        for cand in non_prefixed_candidates:
            if cand in df.columns and cand not in extracted_columns:
                extracted_columns.append(cand)

        # Fallback: if no extracted columns, try the latest enriched dataset in output_dir once
        if not extracted_columns:
            try:
                out_dir = Path(output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                candidates = list(out_dir.glob("final_enriched_data_*.csv")) + list(out_dir.glob("enriched_data_*.csv"))
                if candidates:
                    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    latest = candidates[0]
                    if latest and latest.exists() and str(latest) != str(file_path):
                        df = load_csv_data_cached(str(latest))
                        file_path = str(latest)
                        extracted_columns = [col for col in df.columns if col.startswith('extracted_')]
                        original_columns = [col for col in df.columns if not col.startswith('extracted_')]
            except Exception:
                # Ignore fallback errors; proceed to standard message if still none
                pass

        if not extracted_columns:
            return "No extracted columns found. Use `extract_structured_data_from_text` first."
        
        # Conflict detection and resolution
        conflicts_detected = []
        final_df = df.copy()
        
        # Check for time conflicts
        if 'extracted_time' in extracted_columns:
            time_columns = [col for col in original_columns if any(word in col.lower() for word in ['time', 'date', 'timestamp'])]
            if time_columns:
                time_col = time_columns[0]
                conflicts = 0
                for i, row in df.iterrows():
                    if pd.notna(row['extracted_time']) and pd.notna(row[time_col]):
                        # Simple conflict detection - can be enhanced
                        extracted_time = str(row['extracted_time'])
                        if extracted_time not in str(row[time_col]):
                            conflicts += 1
                
                if conflicts > 0:
                    conflicts_detected.append(f"Time conflicts: {conflicts} discrepancies between {time_col} and extracted_time")
                    
                    if merge_strategy == "prefer_structured":
                        final_df['resolved_time'] = df[time_col]
                    elif merge_strategy == "prefer_extracted":
                        final_df['resolved_time'] = df['extracted_time'].fillna(df[time_col])
                    else:  # validate_both
                        final_df['time_conflict_flag'] = conflicts > 0
                        final_df['resolved_time'] = df[time_col]  # Default to structured
        
        # Enhance existing data with extracted information
        for col in extracted_columns:
            if col not in ['extracted_time']:  # Time handled separately
                # Add extracted data as new enriching columns
                if col.startswith('extracted_'):
                    target_col = col.replace('extracted_', 'enhanced_')
                else:
                    # Non-prefixed known extracted columns -> enhanced_<name>
                    target_col = f"enhanced_{col}"
                final_df[target_col] = df[col]
        
        # Save final merged dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_path / f"final_enriched_data_{timestamp}.csv"
        final_df.to_csv(output_file, index=False)
        
        # Generate analysis
        results = []
        results.append(f"# Data Merge Results: {text_column}")
        results.append(f"**Merge Strategy**: {merge_strategy}")
        results.append(f"**Original Dataset**: {file_path}")
        results.append(f"**Final Dataset**: {output_file}")
        results.append("")
        
        results.append("## Merge Summary")
        results.append(f"- **Original Columns**: {len(original_columns)}")
        results.append(f"- **Extracted Columns**: {len(extracted_columns)}")
        results.append(f"- **Final Columns**: {len(final_df.columns)}")
        results.append(f"- **Conflicts Detected**: {len(conflicts_detected)}")
        results.append("")
        
        if conflicts_detected:
            results.append("## Conflicts Resolved")
            for conflict in conflicts_detected:
                results.append(f"- {conflict}")
            results.append(f"- **Resolution Strategy**: {merge_strategy}")
            results.append("")
        
        results.append("## Enhanced Dataset Features")
        enhanced_columns = [col for col in final_df.columns if col.startswith('enhanced_') or col.startswith('resolved_')]
        if enhanced_columns:
            for col in enhanced_columns:
                non_null_count = final_df[col].notna().sum()
                completion_rate = (non_null_count / len(final_df)) * 100
                results.append(f"- **{col}**: {non_null_count:,} values ({completion_rate:.1f}% complete)")
        else:
            results.append("- All extracted data preserved as separate columns")
        results.append("")
        
        results.append("## Dataset Ready for Analysis")
        results.append("- Text extraction and merging complete")
        results.append("- Enhanced dataset combines structured and text-derived data")
        results.append("- Ready for comprehensive analysis and visualization")
        
        # Attach a machine-readable footer for downstream automation
        try:
            footer = {
                "type": "dataset",
                "source": "merge_text_extracted_data",
                "input_path": file_path,
                "output_path": str(output_file),
                "enhanced_columns": [c for c in final_df.columns if c.startswith('enhanced_') or c.startswith('resolved_')],
            }
            results.append("")
            results.append(f"<!--output_json:{json.dumps(footer)}-->")
        except Exception:
            # Footer is optional; ignore failures
            pass

        return "\n".join(results)
        
    except Exception as e:
        return f"Error merging text-extracted data: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
