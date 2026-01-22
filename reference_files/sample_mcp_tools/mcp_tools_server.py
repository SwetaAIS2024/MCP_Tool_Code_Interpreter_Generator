from fastmcp import FastMCP
import logging
# Essential imports (optimized)
import gc
import hashlib
import json
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Annotated, List, Literal, Union, Dict, Any, Tuple
import logging
import sys
import concurrent.futures

import torch

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)



# Optimized matplotlib configuration
import matplotlib
import numpy as np
import pandas as pd
from pydantic import Field
import pickle

matplotlib.use("Agg")  # Use non-interactive backend for better performance
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM, SVR
from sklearn.cluster import DBSCAN
from sklearn.linear_model import SGDClassifier 
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_samples,
    silhouette_score
)
from sklearn.model_selection import train_test_split
from difflib import get_close_matches
# Data science imports
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import torch.nn as nn
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


_dataset_cache = {}

mcp = FastMCP("analysis_tools")

logger = logging.getLogger(__name__)

log_messages = []

def log(msg):
    """Log to server and capture for client"""
    logger.info(msg)
    log_messages.append(msg)

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

def find_knee_point(k_distances):
    # x = sorted indices
    x = np.arange(len(k_distances))
    y = k_distances

    # Line from first point to last point
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])

    # Compute distances from each point to the line (geometric method)
    line_vec = p2 - p1

    distances = np.abs(np.cross(line_vec, np.vstack([x - x[0], y - y[0]]).T)) / np.linalg.norm(line_vec)

    knee_index = np.argmax(distances)
    return y[knee_index]

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

import sys
@mcp.tool()
def drop_empty_rows(
    file_path: Annotated[str, Field(description="Path to the CSV dataset file")],
    output_path: Annotated[str, Field(description="Path to save the cleaned CSV file")],
    target_columns: Annotated[List[str], Field(description="Optional list of specific columns to check for emptiness (if None, checks all columns)")],
) -> str:
    """
    Tool to drop empty rows from a CSV dataset.
    Loads the dataset from the specified file paths, removes rows that are completely empty
    (or empty in the specified target column), and saves the cleaned dataset to a new file.
    """
    
    try:
        log("Target columns: " + str(target_columns))
        log(f"Loading dataset from {file_path}...")
        df = load_csv_data_with_types(file_path)

        initial_row_count = len(df)
        log(f"Initial row count: {initial_row_count}")

        # Identify columns with dict or list types
        dict_columns = {}
        for col in df.columns:
            dict_rows = df[col].apply(lambda x: isinstance(x, (dict, list)))
            if dict_rows.any():
                dict_columns[col] = df[col][dict_rows]

        if dict_columns:
            log(f"Columns with dictionaries or lists:\n{dict_columns}")

        # Convert unhashable types (dict or list) to strings
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, (dict, list))).any():  # Detect dict or list
                log(f"Converting unhashable types in column: {col}")
                df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (dict, list)) else x)

        # Drop empty rows
        if not target_columns:
            log("Dropping rows that are completely empty across all columns.")
            df_cleaned = df.dropna(how='all')
        else:
            df_cleaned = df.dropna(subset=target_columns)

        final_row_count = len(df_cleaned)
        dropped_rows = initial_row_count - final_row_count

        log(f"Dropped {dropped_rows} empty rows.")
        log(f"Final row count: {final_row_count}")

        # Save cleaned dataset
        df_cleaned.to_csv(output_path, index=False)
        log(f"Cleaned dataset saved to {output_path}")
        log(f"Successfully dropped {dropped_rows} empty rows. Cleaned dataset saved to '{output_path}'.")

        # Get column type information
        column_types = detect_column_types(df_cleaned)
        type_metadata = column_types.get("_metadata", {})
        types_by_column = {
            key: value for key, value in column_types.items() if key != "_metadata"
        }

        report = []
        report.append("## CSV Data Analysis")
        report.append(f"- File: {file_path}")
        report.append(f"- Total rows: {len(df_cleaned):,}")
        report.append(f"- Total columns: {len(df_cleaned.columns)}")
        report.append(
            f"- Memory usage: {df_cleaned.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        )
        report.append(f"- Columns: {', '.join(df_cleaned.columns)}")

        # Enhanced data types with auto-detected types
        report.append("\n### Data Types:")
        for col in df_cleaned.columns:
            dtype = df_cleaned[col].dtype
            detected_type = types_by_column.get(col, "unknown")
            report.append(f"- {col}: {dtype} (detected as: {detected_type})")

        high_cardinality_cols = type_metadata.get("high_cardinality", [])
        if high_cardinality_cols:
            report.append(
                "High-cardinality categorical columns: "
                + ", ".join(high_cardinality_cols)
            )

        # Basic statistics for numeric columns
        numeric_cols = df_cleaned.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            report.append("\n### Numeric Columns Summary:")
            for col in numeric_cols:
                stats = df_cleaned[col].describe()
                report.append(
                    f"- {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
                    f"min={stats['min']}, max={stats['max']}, "
                    f"25%={stats['25%']:.2f}, 75%={stats['75%']:.2f}"
                )

        # Missing values with percentages
        missing_data = df_cleaned.isnull().sum()
        if missing_data.sum() > 0:
            report.append("\n### Missing Values:")
            for col, missing_count in missing_data.items():
                if missing_count > 0:
                    missing_pct = (missing_count / len(df_cleaned)) * 100
                    report.append(
                        f"- {col}: {missing_count} missing values ({missing_pct:.1f}%)"
                    )

        # Data quality summary
        duplicates = df_cleaned.duplicated().sum()
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

        return "\n".join(report) + f"\n\nLog Messages:\n" + "\n".join(log_messages)

    except Exception as e:
        return f"Error while dropping empty rows: {str(e)}, log: {log_messages}"

@mcp.tool()
def Clustering_Algorithms(
    file_path: Annotated[str, Field(description="Path to the CSV dataset file")],
    components_to_cluster: Annotated[List[str], Field(description="List of numeric column names to use for clustering")],
    do_PCA: Annotated[bool, Field(description="Whether to perform PCA before clustering")],
    n_clusters: Annotated[int, Field(description="Number of clusters to form (if 0, auto-detect)")],
    y_axis: Annotated[str, Field(description="Column name for Y-axis in plots (optional, pass empty strings if not needed)")],
    x_axis: Annotated[str, Field(description="Column name for X-axis in plots (optional, pass empty strings if not needed)")],
    PCA_components: Annotated[int, Field(description="Number of PCA components to reduce to before clustering")] = 5,
    auto_expand_json: Annotated[bool, Field(description="Expand JSON columns automatically")] = True,
    n_init: Annotated[Union[str, int], Field(description="n_init parameter for KMeans")] = "auto",
    random_state: Annotated[int, Field(description="Random state for clustering")] = 0,
) -> str:
    """
    clustering tool using Kmeans with PCA, multi-k optimization, sampling, and timeout protection.
    Logs are captured and returned to the MCP client terminal.
    """

    def run_timeout(func, timeout, *args, **kwargs):
        """Run a function with timeout"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(func, *args, **kwargs)
            try:
                return fut.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                log(f"Timeout after {timeout}s: {func.__name__}")
                return None

    try:

        log("Starting KMeans Clustering Tool")
        report  = ["## Comprehensive Clustering Data Analysis"]
        # Validate n_init
        if isinstance(n_init, str):
            if n_init.lower() == "auto":
                n_init = "auto"
            elif n_init.isdigit():
                n_init = int(n_init)
            else:
                raise ValueError("n_init must be 'auto' or an integer string")
        elif isinstance(n_init, int) and n_init < 1:
            raise ValueError("n_init must be >= 1")

        # File check
        if not Path(file_path).exists():
            return f"Error: File '{file_path}' does not exist."

        # Load dataset
        log("Loading dataset...")
        df = load_csv_data_with_types(file_path, auto_expand_json=auto_expand_json)
        df = df[components_to_cluster]
        numeric_cols = df.select_dtypes(include=["number"]).columns

        log(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
        log(f"Numeric columns: {len(numeric_cols)}")

        if len(numeric_cols) < 2:
            return "Not enough numeric columns for clustering."

        clean_df = df[numeric_cols].dropna()
        if len(clean_df) < 3:
            return "Not enough rows after removing NA."

        log(f"Rows after dropna: {len(clean_df)}")

        # Sampling for PCA + KMeans
        MAX_PCA_ROWS = 15000
        if len(clean_df) > MAX_PCA_ROWS:
            log(f"Sampling {len(clean_df)} → {MAX_PCA_ROWS} rows for PCA + KMeans")
            clean_df_sample = clean_df.sample(MAX_PCA_ROWS, random_state=42)
        else:
            clean_df_sample = clean_df

        # Scaling
        log("Scaling data...")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clean_df_sample)
        
        if do_PCA:
            # PCA
            pca_components = min(PCA_components, scaled_data.shape[1])
            log(f"Running PCA with {pca_components} components...")
            pca = PCA(n_components=pca_components)
            pca_data = run_timeout(pca.fit_transform, 10, scaled_data)
            if pca_data is None:
                return "PCA timed out."

            report.append(f"PCA done: {scaled_data.shape[1]} → {pca_data.shape[1]} components")
        else:
            pca_data = scaled_data
            log("Skipping PCA as per user request.")


        if n_clusters > 0:            
            k_values = [n_clusters]
            log(f"Using user-specified number of clusters: {n_clusters}")
        else:
            log("Auto-detecting optimal number of clusters using silhouette scores.")
            k_values = [2, 3, 4, 5, 6, 7, 8]
            log(f"Testing K values: {k_values}")
        # Kmeans algorithm
        
        

        best_k = None
        best_silhouette = -999
        best_labels = None

        MAX_SILH_ROWS = 2000
        if len(pca_data) > MAX_SILH_ROWS:
            sil_idx = np.random.choice(len(pca_data), MAX_SILH_ROWS, replace=False)
            pca_sil = pca_data[sil_idx]
        else:
            pca_sil = pca_data
            sil_idx = None

        for k in k_values:
            log(f"Running KMeans for k={k}...")

            def km_run(data):
                km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init, init='k-means++')
                return km.fit_predict(data)

            labels = run_timeout(km_run, 10, pca_data)
            if labels is None:
                log(f"KMeans for k={k} timed out, skipping...")
                continue

            if sil_idx is not None:
                labels_s = labels[sil_idx]
            else:
                labels_s = labels

            def sil_func():
                return silhouette_score(pca_sil, labels_s)

            log(f"Computing silhouette for k={k}...")
            sil = run_timeout(sil_func, 10)
            if sil is None:
                log(f"Silhouette timeout for k={k}")
                continue

            log(f"k={k} → silhouette={sil:.4f}")

            if sil > best_silhouette:
                best_silhouette = sil
                best_k = k
                best_labels = labels

        if best_k is None or best_labels is None:
            return "Clustering failed or all operations timed out."

        log(f"Best k = {best_k} with silhouette={best_silhouette:.4f}")

        # Kmeans report
        cluster_df = clean_df_sample.copy()
        cluster_df["cluster"] = best_labels
        plt.figure(figsize=(8, 5))
        plt.title(f"KMeans Clustering (k={best_k})")
        # Axis names
        if do_PCA:
            x_axis_name = "PCA Component 1"
            y_axis_name = "PCA Component 2"
        else:
            x_axis_name = x_axis if x_axis != "" else numeric_cols[0]
            y_axis_name = y_axis if y_axis != "" else numeric_cols[1]

        plt.xlabel(x_axis_name)
        plt.ylabel(y_axis_name)
        plt.scatter(
            clean_df_sample[x_axis] if x_axis != "" else pca_data[:, 0],
            clean_df_sample[y_axis] if y_axis != "" else pca_data[:, 1],
            c=best_labels,
            cmap="tab10",
            s=10,
            alpha=0.7,
        )
        plt.savefig("kmeans_clusters.png")
        plt.close()

        summary = (
            cluster_df.groupby("cluster")[numeric_cols]
            .agg(["mean", "count"])
            .round(3)
        )
        summary.columns = [
            f"{col[0]}_{col[1]}" for col in summary.columns
        ]

        report.append(f"## Best KMeans Clustering")
        report.append("### Plot Axes")
        report.append(f"- X-axis: **{x_axis_name}**")
        report.append(f"- Y-axis: **{y_axis_name}**")
        report.append(f"Optimal k: **{best_k}**")
        report.append(f"Silhouette: **{best_silhouette:.4f}**\n")
        report.append("### Cluster Summary")
        report.append(summary.to_markdown(index=True))

        log("KMeans tool finished normally.")

        #DBSCAN clustering
        if do_PCA:
            min_samples = pca_components * 2

        else:
            min_samples = len(numeric_cols) * 2

        # Estimate eps using k-distance method
        log("Estimating eps for DBSCAN using k-distance method...")
        neigh = NearestNeighbors(n_neighbors=min_samples)
        nbrs = neigh.fit(pca_data)
        distances, indices = nbrs.kneighbors(pca_data)
        # use k-distance = distance to the k-th nearest neighbor
        k_distances = np.sort(distances[:, -1])
        eps_auto = find_knee_point(k_distances)

        log("Detected eps: " + str(eps_auto))
        db = run_timeout(lambda: DBSCAN(eps=eps_auto, min_samples=min_samples).fit(pca_data), 10)
        if db is None:
            return "DBSCAN fit timed out."
        #arbitrary eps and min_samples, could be parameterized
        labels = db.labels_
        n_noise_ = list(labels).count(-1)

        plt.figure(figsize=(8, 5))
        plt.title(f"DBSCAN Clustering (eps={eps_auto:.2f})")
        plt.xlabel(x_axis_name)
        plt.ylabel(y_axis_name)
        plt.scatter(
            clean_df_sample[x_axis] if x_axis != "" else pca_data[:, 0],
            clean_df_sample[y_axis] if y_axis != "" else pca_data[:, 1],
            c=labels,
            cmap="tab10",
            s=10,
            alpha=0.7,
        )
        plt.savefig("dbscan_clusters.png")
        plt.close()

        #DBSCAN report
        report.append(f"## DBSCAN Clustering")
        report.append(f"Optimal eps: **{eps_auto}**")
        report.append(f"Min_Samples: **{min_samples}**\n")
        report.append("### Cluster Summary")
        report.append("Estimated number of noise points: %d" % n_noise_)
        #I built this tool then i realised the llm cant process the qualitative insights from clustering

        # Flatten MultiIndex columns to something readable
        
        markdown_content = ""
        for item in report:
            markdown_content += f"* {item}\n" # Use '*' or '-' for bullet points
        with open("Clustering_report.md", "w",encoding="utf-8") as md_file:
            md_file.write(markdown_content)

        report.append("\n### Logs")
        report.extend(log_messages)

        return "\n".join(report)

    except Exception as e:
        log(f"ERROR: {e}")
        report.append("\n### Logs")
        report.extend(log_messages)
        return f"clustering failed: {e}\n" + "\n".join(report)

@mcp.tool()
def abnormality_detection(
    file_path: Annotated[str, Field(description="Path to the CSV dataset file")],
    auto_expand_json: Annotated[bool, Field(description="Expand JSON columns automatically")] = True,
):
    """Detect anomalies in the dataset."""
    try:
        report = ["## Anomaly Detection Report"]
        # File check
        if not Path(file_path).exists():
            return f"Error: File '{file_path}' does not exist."

        # Load dataset
        log("Loading dataset...")
        df = load_csv_data_with_types(file_path, auto_expand_json=auto_expand_json)
        numeric_cols = df.select_dtypes(include=["number"]).columns

        log(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
        log(f"Numeric columns: {len(numeric_cols)}")

        if len(numeric_cols) < 2:
            return "Not enough numeric columns for clustering."

        clean_df = df[numeric_cols].dropna()
        if len(clean_df) < 3:
            return "Not enough rows after removing NA."

        log(f"Rows after dropna: {len(clean_df)}")

        # Sampling for PCA + KMeans
        MAX_PCA_ROWS = 15000
        if len(clean_df) > MAX_PCA_ROWS:
            log(f"Sampling {len(clean_df)} → {MAX_PCA_ROWS} rows for PCA + KMeans")
            clean_df_sample = clean_df.sample(MAX_PCA_ROWS, random_state=42)
        else:
            clean_df_sample = clean_df

        # Scaling
        log("Scaling data...")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clean_df_sample)

        # Initialize and fit the One-Class SVM model
        clf = OneClassSVM(nu=0.05, kernel="rbf")  # Adjust the nu parameter as needed
        clf.fit(scaled_data)
        # Predict whether each data point is an anomaly (-1 for anomalies, 1 for normal data)
        predictions = clf.predict(scaled_data)
        anomalies_indices = np.where(predictions == -1)[0]
        report.append("Detected anomalies: " + str(anomalies_indices))
        markdown_content = ""
        for item in report:
            markdown_content += f"* {item}\n" # Use '*' or '-' for bullet points
        with open("anomaly_detection_report.md", "w",encoding="utf-8") as md_file:
            md_file.write(markdown_content)
        report.append("\n### Logs")
        report.extend(log_messages) 
        return "\n".join(report)
    except Exception as e:
        log(f"ERROR: {e}")
    pass

@mcp.tool()
def remove_abnormalities(
    file_path: Annotated[str, Field(description="Path to the CSV dataset file")],
    output_path: Annotated[str, Field(description="Path to save the cleaned CSV file")],
    auto_expand_json: Annotated[bool, Field(description="Expand JSON columns automatically")] = True,
):
    """Remove anomalies from the dataset and save cleaned data."""
    try:
        log("Loading dataset...")
        df = load_csv_data_with_types(file_path, auto_expand_json=auto_expand_json)
        numeric_cols = df.select_dtypes(include=["number"]).columns

        log(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
        log(f"Numeric columns: {len(numeric_cols)}")

        if len(numeric_cols) < 2:
            return "Not enough numeric columns for clustering."

        clean_df = df[numeric_cols].dropna()
        if len(clean_df) < 3:
            return "Not enough rows after removing NA."

        log(f"Rows after dropna: {len(clean_df)}")

        # Scaling
        log("Scaling data...")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clean_df)

        # Initialize and fit the One-Class SVM model
        clf = OneClassSVM(nu=0.05, kernel="rbf")  # Adjust the nu parameter as needed
        clf.fit(scaled_data)
        # Predict whether each data point is an anomaly (-1 for anomalies, 1 for normal data)
        predictions = clf.predict(scaled_data)
        normal_indices = np.where(predictions == 1)[0]
        cleaned_df = clean_df.iloc[normal_indices]

        log(f"Removed {len(clean_df) - len(cleaned_df)} anomalies. Cleaned data has {len(cleaned_df)} rows.")

        # Save cleaned dataset
        cleaned_df.to_csv(output_path, index=False)
        log(f"Cleaned dataset saved to {output_path}")

        return f"Successfully removed anomalies. Cleaned dataset saved to '{output_path}'."
    except Exception as e:
        return f"ERROR: {e}"
    pass

@mcp.tool()
def build_regression_model(
    file_path: Annotated[str, Field(description="Path to the CSV dataset file")],
    independent_vars: Annotated[List[str], Field(description="List of independent variable column names")],
    dependent_var: Annotated[str, Field(description="Dependent variable column name")],
    auto_expand_json: Annotated[bool, Field(description="Expand JSON columns automatically")] = True,
):
    """Perform regression analysis using both svm and linear on the dataset."""
    try:
        report = ["## Regression Analysis Report"]
        # File check
        if not Path(file_path).exists():
            return f"Error: File '{file_path}' does not exist."

        # Load dataset
        log("Loading dataset...")
        df = load_csv_data_with_types(file_path, auto_expand_json=auto_expand_json)
        numeric_cols = df.select_dtypes(include=["number"]).columns

        log(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
        log(f"Numeric columns: {len(numeric_cols)}")

        if len(numeric_cols) < 2:
            return "Not enough numeric columns for clustering."

        clean_df = df[numeric_cols].dropna()
        if len(clean_df) < 3:
            return "Not enough rows after removing NA."

        log(f"Rows after dropna: {len(clean_df)}")

        # Sampling for PCA + KMeans
        MAX_PCA_ROWS = 15000
        if len(clean_df) > MAX_PCA_ROWS:
            log(f"Sampling {len(clean_df)} → {MAX_PCA_ROWS} rows for PCA + KMeans")
            clean_df_sample = clean_df.sample(MAX_PCA_ROWS, random_state=42)
        else:
            clean_df_sample = clean_df

        X = clean_df_sample[independent_vars]
        y = clean_df_sample[dependent_var]

        # Scaling
        log("Scaling data...")
        numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical_features = [
            col for col in X.columns if col not in numeric_features
        ]
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ]
        )

        regressor_pipeline = Pipeline(
            steps=[
                ("pre", preprocessor),
                ("clf", LinearRegression()),
            ]
        )
        SVR_pipeline = Pipeline(
            steps=[
                ("pre", preprocessor),
                ("clf", SVR(kernel='rbf'))
            ]
        )
        
        #linear regession model
        lin_model = regressor_pipeline.fit(X, y)
        r_squared = lin_model.score(X, y)
        feature_names = lin_model.named_steps["pre"].get_feature_names_out()
        coefficients = lin_model.named_steps["clf"].coef_
        coef_table = dict(zip(feature_names, coefficients))
        report.append("Linear Regression Coefficients: " + str(coef_table))

        #SVR regression model
        svr_model = SVR_pipeline.fit(X, y)
        r_squared_svr = svr_model.score(X, y)
        report.append(f"SVM Regression R-squared: {r_squared_svr}")
        
        markdown_content = ""
        for item in report:
            markdown_content += f"* {item}\n" # Use '*' or '-' for bullet points
        with open("regression_analysis_report.md", "w",encoding="utf-8") as md_file:
            md_file.write(markdown_content)
        report.append("\n### Logs")
        report.extend(log_messages) 
        return "\n".join(report)
    except Exception as e:
        return(f"ERROR: {e}")
    pass

def safe_parse_columns(requested_cols):
    """Normalizes various broken LLM formats into a clean list of names."""
    if requested_cols is None:
        return []

    # Handle strings including "[...]" or weird formatting
    if isinstance(requested_cols, str):
        cleaned = re.sub(r'[\[\]\(\)\'"]', '', requested_cols)
        col_list = [c.strip() for c in cleaned.split(",") if c.strip()]
        return col_list

    if isinstance(requested_cols, list):
        return [str(c).strip() for c in requested_cols]

    return [str(requested_cols).strip()]


def validate_columns_safe(requested_cols, df_cols, require_nonempty=True, tool_name=""):
    """Full safety wrapper: sanitizes input, validates, fuzzy matches."""
    parsed = safe_parse_columns(requested_cols)

    if require_nonempty and len(parsed) == 0:
        return {
            "ok": False,
            "error": f"No columns provided for {tool_name}.",
            "valid": [],
            "invalid": []
        }

    valid = [c for c in parsed if c in df_cols]
    invalid = [c for c in parsed if c not in df_cols]

    suggestions = {
        col: get_close_matches(col, df_cols, n=3, cutoff=0.6)
        for col in invalid
    }

    if invalid:
        return {
            "ok": False,
            "valid": valid,
            "invalid": invalid,
            "error": (
                f"Invalid columns provided to {tool_name}:\n"
                f"{invalid}\n"
                f"Suggestions:\n" + "\n".join(
                    f"{col}: {matches}" for col, matches in suggestions.items()
                ) +
                f"\nAvailable columns:\n{list(df_cols)}"
            )
        }

    return {"ok": True, "valid": valid, "invalid": []}
# --------------------------------------------
# FULLY REWRITTEN SGD CLASSIFIER TOOL
# --------------------------------------------
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV

@mcp.tool()
def build_SGD_Classifier(
    file_path: Annotated[str, Field(description="Path to CSV dataset")],
    independent_vars: Annotated[str, Field(description="Feature columns to use")],
    dependent_var: Annotated[str, Field(description="Target column name (categorical)")],
    auto_expand_json: Annotated[bool, Field(description="Auto-expand JSON columns")] = True,
) -> str:

    try:
        # --- File check ---
        if not Path(file_path).exists():
            return f"Error: File '{file_path}' does not exist."

        # --- Load the dataset ---
        log("Loading dataset…")
        df = load_csv_data_with_types(file_path, auto_expand_json=auto_expand_json)

        log(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
        log(f"Columns: {list(df.columns)}")

        # Drop rows where target is missing
        if dependent_var not in df.columns:
            return f"Error: Target column '{dependent_var}' not found in dataset."

        df = df.dropna(subset=[dependent_var])

        # --- Validate column names (LLM-safe) ---
        vr = validate_columns_safe(
            independent_vars, df.columns, True,
            tool_name="SGDClassifier(independent_vars)"
        )
        if not vr["ok"]:
            return vr["error"]
        valid_independent = vr["valid"]

        dr = validate_columns_safe(
            dependent_var, df.columns, True,
            tool_name="SGDClassifier(dependent_var)"
        )
        if not dr["ok"]:
            return dr["error"]
        target_col = dr["valid"][0]

        log("Independent vars validated: " + str(valid_independent))
        log("Dependent var validated: " + str(target_col))

        # Extract X and y
        X = df[valid_independent]
        y = df[target_col]

        # --- Handle datetime columns automatically ---
        datetime_cols = X.select_dtypes(include=["datetime64", "datetime64[ns]"]).columns
        if len(datetime_cols) > 0:
            log(f"Converting datetime columns to timestamps: {list(datetime_cols)}")
            for col in datetime_cols:
                X[col] = X[col].astype("int64") / 1e9  # seconds since epoch

        # --- Separate numeric + categorical features ---
        numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical_features = [
            col for col in X.columns if col not in numeric_features
        ]

        log(f"Numeric features: {numeric_features}")
        log(f"Categorical features: {categorical_features}")

        # --- Build preprocessing pipeline with imputer ---
        # SimpleImputer will fill missing values with the median for numeric columns
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),  # Impute missing values with median
                    ("scaler", StandardScaler())  # Then scale the numeric columns
                ]), numeric_features),
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # Impute missing values with the most frequent value
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))  # Then encode categorical variables
                ]), categorical_features),
            ]
        )

        clf_pipeline = Pipeline(
            steps=[
                ("pre", preprocessor),
                ("clf", SGDClassifier(max_iter=1000, tol=1e-3)),
            ]
        )

        # --- Train/test split ---
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # --- GridSearchCV for hyperparameter tuning ---
        param_grid = {
            'clf__alpha': [0.0001, 0.001, 0.01],
            'clf__loss': ['hinge', 'log', 'modified_huber'],  
        }

        clf = GridSearchCV(clf_pipeline, param_grid, cv=5)
        clf.fit(X_train, y_train)

        # --- Accuracy ---
        accuracy = clf.score(X_test, y_test)
        log(f"Accuracy: {accuracy}")

        # --- Save the model ---
        model_path = "sgd_classifier_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(clf, f)

    except Exception as e:
        log(f"ERROR: {e}")

    return "\n".join(log_messages)

@mcp.tool()
def create_input_data(
    file_path: Annotated[str, Field(description="Path to the CSV dataset file")],
    independent_vars: Annotated[str, Field(description="Feature columns to use")],
    auto_expand_json: Annotated[bool, Field(description="Auto-expand JSON columns")] = True,
) -> str:
    """Create input data CSV for prediction using specified independent variables."""
    try:
        # --- File check ---
        if not Path(file_path).exists():
            return f"Error: File '{file_path}' does not exist."

        # --- Load the dataset ---
        log("Loading dataset…")
        df = load_csv_data_with_types(file_path, auto_expand_json=auto_expand_json)

        log(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
        log(f"Columns: {list(df.columns)}")

        # --- Validate column names (LLM-safe) ---
        vr = validate_columns_safe(
            independent_vars, df.columns, True,
            tool_name="CreateInputData(independent_vars)"
        )
        if not vr["ok"]:
            return vr["error"]
        valid_independent = vr["valid"]

        log("Independent vars validated: " + str(valid_independent))

        input_df = df[valid_independent]

        # --- Save input data ---
        output_path = "sgd_classifier_input_data.csv"
        input_df.to_csv(output_path, index=False)
        log(f"Input data saved to {output_path}")

    except Exception as e:
        log(f"ERROR: {e}")

    return "\n".join(log_messages
)

@mcp.tool()
def run_classifier_to_predict(
    type: Annotated[str, Field(description="Type of classifier to use (HGBoosting or SGD)")],
    model_path: Annotated[str, Field(description="Path to the saved SGD Classifier model")],
    input_data_path: Annotated[str, Field(description="Path to the CSV file with input data for prediction")],
    auto_expand_json: Annotated[bool, Field(description="Auto-expand JSON columns")] = True,
) -> str:
    """Load a saved SGD Classifier model and make predictions on new data."""
    try:
        # --- File check ---
        if not Path(model_path).exists():
            return f"Error: Model file '{model_path}' does not exist."
        if not Path(input_data_path).exists():
            return f"Error: Input data file '{input_data_path}' does not exist."

        # --- Load the model ---
        log("Loading model…")
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)

        # --- Load the input data ---
        log("Loading input data…")
        df = load_csv_data_with_types(input_data_path, auto_expand_json=auto_expand_json)
        log(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

        # --- Make predictions ---
        predictions = clf.predict(df)
        df['predictions'] = predictions

        # --- Save predictions ---
        output_path = type + "_predictions.csv"
        df.to_csv(output_path, index=False)
        log(f"Predictions saved to {output_path}")

    except Exception as e:
        log(f"ERROR: {e}")

    return "\n".join(log_messages)


@mcp.tool()
def build_HGBoosting_Classifier(
    file_path: Annotated[str, Field(description="Path to the CSV dataset file")],
    independent_vars: Annotated[str, Field(description="Feature columns to use")],
    dependent_var: Annotated[str, Field(description="Target column name (categorical)")],
    auto_expand_json: Annotated[bool, Field(description="Expand JSON columns automatically")] = True,
):
    """Placeholder for neural network analysis tool."""
    try:
        # --- File check ---
        if not Path(file_path).exists():
            return f"Error: File '{file_path}' does not exist."

        # --- Load the dataset ---
        log("Loading dataset…")
        df = load_csv_data_with_types(file_path, auto_expand_json=auto_expand_json)

        log(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
        log(f"Columns: {list(df.columns)}")
        
        # Drop rows where target is missing
        if dependent_var not in df.columns:
            return f"Error: Target column '{dependent_var}' not found in dataset."

        df = df.dropna(subset=[dependent_var])

        # --- Validate column names (LLM-safe) ---
        vr = validate_columns_safe(
            independent_vars, df.columns, True,
            tool_name="SGDClassifier(independent_vars)"
        )
        if not vr["ok"]:
            return vr["error"]
        valid_independent = vr["valid"]

        dr = validate_columns_safe(
            dependent_var, df.columns, True,
            tool_name="SGDClassifier(dependent_var)"
        )
        if not dr["ok"]:
            return dr["error"]
        target_col = dr["valid"][0]

        log("Independent vars validated: " + str(valid_independent))
        log("Dependent var validated: " + str(target_col))

        X = df[valid_independent]
        y = df[target_col]    
        # --- Handle datetime columns automatically ---
        datetime_cols = X.select_dtypes(include=["datetime64", "datetime64[ns]"]).columns
        if len(datetime_cols) > 0:
            log(f"Converting datetime columns to timestamps: {list(datetime_cols)}")
            for col in datetime_cols:
                X[col] = X[col].astype("int64") / 1e9  # seconds since epoch

        # --- Separate numeric + categorical features ---
        numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical_features = [c for c in X.columns if c not in numeric_features]

        log(f"Numeric features: {numeric_features}")
        log(f"Categorical features: {categorical_features}")

        # --- Convert categorical columns to pandas 'category' dtype ---
        # --- Convert all object/string columns into pandas category dtype ---
        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = X[col].astype("category")
            log(f"Converted {col} -> category")

        # --- No scaling or encoding required ---
        preprocessor = ColumnTransformer(
            transformers=[
                ("keep_all", "passthrough", X.columns)
            ],
            remainder="drop"
        )

        clf_pipeline = Pipeline(
            steps=[
                ("pre", preprocessor),
                ("clf", HistGradientBoostingClassifier(
                    categorical_features="from_dtype",  # auto-detect categories
                    max_iter=100
                )),
            ]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        clf = clf_pipeline.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)

        log(f"HGBoost Model Accuracy: {accuracy}")
        # --- Save the model ---
        model_path = "HGB_classifier_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(clf, f)

        return "\n".join(log_messages)


    except Exception as e:
        log(f"ERROR: {e}")
        return "\n".join(log_messages)

@mcp.tool()
def run_time_series_analysis(
    file_path: Annotated[str, Field(description="Path to the CSV dataset file")],
    datetime_column: Annotated[str, Field(description="Name of the datetime column")],
    value_column: Annotated[str, Field(description="Name of the value column for analysis, MUST BE NUMERIC, DONT USE ID VALUES")] ,
    auto_expand_json: Annotated[bool, Field(description="Expand JSON columns automatically")] = True,
):
    """Perform time series analysis on the dataset."""
    try:
        report = ["## Time Series Analysis Report"]
        # --- File check ---
        if not Path(file_path).exists():
            return f"Error: File '{file_path}' does not exist."

        # --- Load the dataset ---
        log("Loading dataset…")
        df = load_csv_data_with_types(file_path, auto_expand_json=auto_expand_json)

        log(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
        log(f"Columns: {list(df.columns)}")

        if datetime_column not in df.columns:
            return f"Error: Datetime column '{datetime_column}' not found in dataset."
        if value_column not in df.columns:
            return f"Error: Value column '{value_column}' not found in dataset."
        
        log(f"Using datetime column: {datetime_column}"
            f", value column: {value_column}")
        # Convert datetime column to datetime type
        df[datetime_column] = pd.to_datetime(df[datetime_column], errors='coerce')
        df = df.dropna(subset=[datetime_column, value_column])

        # Set datetime as index
        df.set_index(datetime_column, inplace=True)
        ts = df[value_column].sort_index()

        # Resample to daily frequency
        ts_daily = ts.resample('D').mean().interpolate()
        #seasonal naive
        log("Performing Seasonal Naive Forecasting...")
        seasonal_period = 30  # assuming monthly seasonality for daily data
        ts_forecast = ts_daily.shift(seasonal_period)
        plt.figure(figsize=(12, 6))
        plt.plot(ts_daily[-365:], label='Original', color='blue')
        plt.plot(ts_forecast[-365:], label='Seasonal Naive Forecast', color='orange')
        plt.legend()    
        plt.title('Seasonal Naive Forecasting')
        plot_path = "seasonal_naive_forecast.png"
        plt.savefig(plot_path)
        log(f"Seasonal Naive forecast plot saved to {plot_path}")

        # Prepare data for MAPE calculation
        s1 = pd.Series(ts_daily[-30:],name='Actual')
        s2 = pd.Series(ts_forecast[-30:],name='Seasonal_Naive')
        test = pd.concat([s1, s2], axis=1)
        test.dropna(inplace=True)
        log(f"Prepared {len(test)} rows for MAPE calculation.") 

        #MAPE calculation
        def mape(forecast):
            return np.mean(np.abs((test["Actual"] - forecast) / test["Actual"])) * 100
        log("Calculating MAPE for Seasonal Naive Forecast...")
        naive_mape = mape( test["Seasonal_Naive"])

        report.append(f"MAPE (Seasonal Naive Forecast): {naive_mape:.2f}%")

        # Test stationarity with ADF test
        log("Performing Augmented Dickey-Fuller Test...")
        from statsmodels.tsa.stattools import adfuller
        adf_result = adfuller(ts_daily)
        report.append(f"ADF Statistic: {adf_result[0]}")
        report.append(f"p-value: {adf_result[1]}")

        if adf_result[1] <= 0.05:
            report.append("The time series is stationary (reject H0).")
        else:
            report.append("The time series is non-stationary (fail to reject H0).")

        ts_daily_diff = ts_daily.diff().dropna()
        # ADF test on differenced series
        log("Performing ADF Test on Differenced Series...") 
        adf_result_diff = adfuller(ts_daily_diff)
        report.append(f"Differenced ADF Statistic: {adf_result_diff[0]}")
        report.append(f"Differenced p-value: {adf_result_diff[1]}") 
        if adf_result_diff[1] <= 0.05:
            report.append("The differenced time series is stationary (reject H0).")
        else:
            report.append("The differenced time series is non-stationary (fail to reject H0).")
        

        # Decompose time series
        log("Performing Seasonal Decomposition...")
        decomposition = seasonal_decompose(ts_daily, model='additive', period=30)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        plt.figure(figsize=(12, 8))
        decomposition.plot().suptitle('Additive Decomposition', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.legend()
        plt.title('Time Series Decomposition')
        plot_path = "time_series_decomposition.png"
        plt.savefig(plot_path)
        log(f"Decomposition plot saved to {plot_path}")
        log("Time Series Decomposition Completed.")

        from statsmodels.tsa.arima.model import ARIMA
        import pmdarima as pm

        model = pm.auto_arima(
            ts_daily,
            start_p=1, start_q=1,
            max_p=5, max_q=5,
            m=1,
            d=None,
            seasonal=False,
            trace=True,
        )
        log("Auto ARIMA Model Summary:")
        log(model.summary().as_text())
        model_order = model.order
        report.append(f"Auto ARIMA selected order: {model_order}")
        # --- Fit ARIMA on all data except last 30 days ---
        forecast_steps = 30
        train_data = ts_daily[:-forecast_steps]  # all except last 30 days
        test_data = ts_daily[-forecast_steps:]   # last 30 days

        log("Fitting ARIMA model...")
        model = ARIMA(train_data, order=model_order).fit()

        # --- Forecast the last 30 days ---
        arima_forecast = model.forecast(steps=forecast_steps)
        arima_forecast = pd.Series(arima_forecast, index=test_data.index)

        arima_mape = mape(arima_forecast)

        report.append(f"ARIMA Last-30-Days MAPE: {arima_mape:.2f}%")
        log(f"ARIMA Last-30-Days MAPE: {arima_mape:.2f}%")

        # --- Optional plot ---
        plt.figure(figsize=(12, 6))
        plt.plot(test_data.index, test_data.values, label="Actual Last 30 Days")
        plt.plot(test_data.index, arima_forecast, label="ARIMA Forecast", linestyle="--")
        plt.xlabel("Date")
        plt.ylabel(value_column)
        plt.title("ARIMA Forecast for Last 30 Days")
        plt.legend()
        forecast_plot_path = "ARIMA_last30_forecast.png"
        plt.savefig(forecast_plot_path)
        plt.close()
        log(f"ARIMA last-30-days forecast plot saved to {forecast_plot_path}")

        # SARIMA model
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        log("Fitting SARIMA model...")
        sarima_order = model_order
        seasonal_order = (1, 1, 1, 12)  # assuming monthly seasonality for daily data
        sarima_model = SARIMAX(train_data, order=sarima_order, seasonal_order=seasonal_order).fit(disp=False)

        sarima_forecast = sarima_model.forecast(steps=forecast_steps)
        sarima_forecast = pd.Series(sarima_forecast, index=test_data.index)
        sarima_mape = mape(sarima_forecast)
        report.append(f"SARIMA Last-30-Days MAPE: {sarima_mape:.2f}%")
        log(f"SARIMA Last-30-Days MAPE: {sarima_mape:.2f}%")

        
        # --- Optional plot ---
        plt.figure(figsize=(12, 6))
        plt.plot(test_data.index, test_data.values, label="Actual Last 30 Days")
        plt.plot(test_data.index, sarima_forecast, label="SARIMA Forecast", linestyle="--")
        plt.xlabel("Date")
        plt.ylabel(value_column)
        plt.title("SARIMA Forecast for Last 30 Days")
        plt.legend()
        forecast_plot_path = "SARIMA_last30_forecast.png"
        plt.savefig(forecast_plot_path)
        plt.close()
        log(f"SARIMA last-30-days forecast plot saved to {forecast_plot_path}")

        # LSTM model
        log("Scaling for LSTM model...")
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(ts_daily.values.reshape(-1, 1))
        scaled_train = scaled_data[:-forecast_steps]
        scaled_test = scaled_data[-forecast_steps:]
        log("Creating LSTM datasets...")
        def create_lstm_dataset(data, time_step=1):
            X, Y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:(i + time_step), 0])
                Y.append(data[i + time_step, 0])
            return np.array(X), np.array(Y)
        X_train, y_train = create_lstm_dataset(scaled_train)
        X_test, y_test = create_lstm_dataset(scaled_test)
        log("tensorflow import for LSTM model...")

        # Define the LSTM model
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(LSTMModel, self).__init__()
                # LSTM layer
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                # Fully connected layer (output size of 1)
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                # Pass through LSTM layer
                lstm_out, (hn, cn) = self.lstm(x)  # lstm_out: batch_size x seq_len x hidden_size
                # Only take the output from the last time step
                out = self.fc(lstm_out[:, -1, :])  # (batch_size, output_size)
                return out

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        log("Building and training LSTM model...")

        def training_loop(n_epochs, optimiser, model, loss_fn, X_train, X_val, y_train, y_val):
            for epoch in range(1, n_epochs + 1):
                model.train()  # Set the model to training mode
                output_train = model(X_train)  # Forward pass for training data
                loss_train = loss_fn(output_train, y_train)  # Training loss

                model.eval()  # Set the model to evaluation mode
                with torch.no_grad():  # No gradient calculation during validation
                    output_val = model(X_val)
                    loss_val = loss_fn(output_val, y_val)  # Validation loss
                
                optimiser.zero_grad()  # Zero gradients for the next iteration
                loss_train.backward()  # Backward pass
                optimiser.step()  # Optimizer step
                
                if epoch == 1 or epoch % 1000 == 0:
                    log(f"Epoch {epoch}, Training loss {loss_train.item():.4f}, Validation loss {loss_val.item():.4f}")
        # Define model parameters
        input_size = 1  # Since each time step only has one feature
        hidden_size = 64  # You can adjust this based on your dataset and complexity
        output_size = 1  # Single output prediction per sequence

        # Instantiate the model
        lstm_model = LSTMModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_test, dtype=torch.float32)

        # Optimizer and loss function
        optimiser = torch.optim.Adam(lstm_model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        # Train the model
        log("Building and training LSTM model...")
        training_loop(10000, optimiser, lstm_model, loss_fn, X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor)

        
        # Initialize the input for prediction
        input_seq = X_test[0].reshape(1, 1, 1)  # Start with the first test sequence

        predictions = []

        # Generate predictions for the next 30 days
        for _ in range(forecast_steps):
            with torch.no_grad():
                pred = lstm_model(torch.tensor(input_seq, dtype=torch.float32))
            predictions.append(pred.item())  # Store the predicted value

            # Update the input sequence with the predicted value
            input_seq = np.roll(input_seq, -1, axis=1)  # Shift the window by one time step
            input_seq[0, -1, 0] = pred.item()  # Add the prediction to the end of the window

        # Create a Pandas Series for the predicted values
        forecast_index = test_data.index  # This will align the forecast with the test data index
        forecast_series = pd.Series(predictions, index=forecast_index)

        # --- Evaluate the Model: MAPE ---
        # Actual values (test_data) vs forecasted values
        mape = np.mean(np.abs((test_data.values - forecast_series.values) / test_data.values)) * 100
        log(f"LSTM MAPE for last 30 days: {mape:.2f}%")                 
        # Predict on the test set
        lstm_model.eval()  # Set the model to evaluation mode
        lstm_predictions = forecast_series

        # --- Optional plot ---
        plt.figure(figsize=(12, 6))
        plt.plot(test_data.index, test_data.values, label="Actual Last 30 Days")
        plt.plot(test_data.index, lstm_predictions, label="LSTM Forecast", linestyle="--")
        plt.xlabel("Date")
        plt.ylabel(value_column)
        plt.title("LSTM Forecast for Last 30 Days")
        plt.legend()
        forecast_plot_path = "LSTM_last30_forecast.png"
        plt.savefig(forecast_plot_path)
        plt.close()
        log(f"LSTM last-30-days forecast plot saved to {forecast_plot_path}")


        markdown_content = ""
        for item in report:
            markdown_content += f"* {item}\n" # Use '*' or '-' for bullet points
        with open("time_series_analysis_report.md", "w",encoding="utf-8") as md_file:
            md_file.write(markdown_content)
        report.append("\n### Logs")
        report.extend(log_messages) 
        return "\n".join(report)

    except Exception as e:
        log(f"ERROR: {e}")
        return "\n".join(log_messages)

#FIX LATER - ADD HYPERPARAMETER TUNING
@mcp.tool()
def ARIMA_forecasting(
    file_path: Annotated[str, Field(description="Path to the CSV dataset file")],
    datetime_column: Annotated[str, Field(description="Name of the datetime column")],
    value_column: Annotated[str, Field(description="Name of the value column")],
    forecast_periods: Annotated[int, Field(description="Number of periods to forecast")] = 30,
    auto_expand_json: Annotated[bool, Field(description="Expand JSON columns automatically")] = True,
):
    """Perform ARIMA forecasting on the dataset."""
    try:
        report = ["## ARIMA Forecasting Report"]
        # --- File check ---
        if not Path(file_path).exists():
            return f"Error: File '{file_path}' does not exist."

        # --- Load the dataset ---
        log("Loading dataset…")
        df = load_csv_data_with_types(file_path, auto_expand_json=auto_expand_json)

        log(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
        log(f"Columns: {list(df.columns)}")

        if datetime_column not in df.columns:
            return f"Error: Datetime column '{datetime_column}' not found in dataset."
        if value_column not in df.columns:
            return f"Error: Value column '{value_column}' not found in dataset."

        # Convert datetime column to datetime type
        df[datetime_column] = pd.to_datetime(df[datetime_column], errors='coerce')
        df = df.dropna(subset=[datetime_column, value_column])

        # Set datetime as index
        df.set_index(datetime_column, inplace=True)
        ts = df[value_column].sort_index()

        # Resample to daily frequency
        ts_daily = ts.resample('D').mean().interpolate()

        from statsmodels.tsa.arima.model import ARIMA

        log("Fitting ARIMA model...")
        model = ARIMA(ts_daily, order=(5, 1, 0)).fit()
        log(model.summary().as_text())

        # --- Forecast future periods ---
        arima_forecast = model.forecast(steps=forecast_periods)
        forecast_index = pd.date_range(
            start=ts_daily.index[-1] + pd.Timedelta(days=1),
            periods=forecast_periods,
            freq='D'
        )
        arima_forecast = pd.Series(arima_forecast, index=forecast_index)

        report.append(f"ARIMA Forecast for next {forecast_periods} periods:")
        report.append(arima_forecast.to_string())

        markdown_content = ""
        for item in report:
            markdown_content += f"* {item}\n" # Use '*' or '-' for bullet points
        with open("arima_forecasting_report.md", "w",encoding="utf-8") as md_file:
            md_file.write(markdown_content)
        report.append("\n### Logs")
        report.extend(log_messages) 
        return "\n".join(report)
    except Exception as e:
        log(f"ERROR: {e}")
        return "\n".join(log_messages)

if __name__ == "__main__":
    mcp.run()