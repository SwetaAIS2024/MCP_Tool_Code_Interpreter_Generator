# Module PR 11: Utils Package

**Module**: `src/utils/`  
**Priority**: P0 (Shared utilities)  
**Estimated Effort**: 2-3 days  
**Dependencies**: None (foundation module)

---

## 1. Module Purpose

The Utils package provides shared helper functions for:
- **CSV Operations** - Load data with type detection
- **Type Detection** - Auto-detect numeric, categorical, datetime columns
- **Validation** - File path security, column existence
- **Security** - Import allowlisting, file path validation

**Key Principle**: Reusable, well-tested utilities used across all modules.

---

## 2. Module Structure

```
src/utils/
├── __init__.py
├── csv_helpers.py
├── type_detection.py
├── validation_helpers.py
└── security_helpers.py
```

---

## 3. Core Components

### 3.1 CSV Helpers (`csv_helpers.py`)

```python
"""CSV data loading and manipulation utilities."""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any


def load_csv_data_with_types(
    file_path: Path,
    auto_detect_types: bool = True
) -> pd.DataFrame:
    """
    Load CSV file with automatic type detection.
    
    Args:
        file_path: Path to CSV file
        auto_detect_types: Whether to auto-detect column types
    
    Returns:
        DataFrame with properly typed columns
    """
    # Load CSV
    df = pd.read_csv(file_path)
    
    if auto_detect_types:
        df = auto_detect_column_types(df)
    
    return df


def auto_detect_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Auto-detect and convert column types.
    
    Detects:
    - Numeric (int, float)
    - Datetime
    - Categorical
    - Boolean
    """
    from .type_detection import detect_column_types
    
    type_mapping = detect_column_types(df)
    
    for col, dtype in type_mapping.items():
        if dtype == "numeric":
            df[col] = pd.to_numeric(df[col], errors='coerce')
        elif dtype == "datetime":
            df[col] = pd.to_datetime(df[col], errors='coerce')
        elif dtype == "boolean":
            df[col] = df[col].astype(bool)
        elif dtype == "categorical":
            df[col] = df[col].astype('category')
    
    return df


def preview_csv_data(
    file_path: Path,
    n_rows: int = 5
) -> Dict[str, Any]:
    """
    Load CSV preview with metadata.
    
    Returns:
        Dict with preview, shape, columns, dtypes
    """
    df = pd.read_csv(file_path, nrows=n_rows)
    
    return {
        "preview": df.to_dict(orient="records"),
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }


def save_dataframe_to_csv(
    df: pd.DataFrame,
    output_path: Path,
    index: bool = False
) -> None:
    """Save DataFrame to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=index)
```

### 3.2 Type Detection (`type_detection.py`)

```python
"""Column type detection utilities."""

import pandas as pd
from typing import Dict


def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Detect semantic type of each column.
    
    Returns:
        Dict mapping column name to type: 
        "numeric", "datetime", "categorical", "boolean", "text"
    """
    type_map = {}
    
    for col in df.columns:
        type_map[col] = detect_column_type(df[col])
    
    return type_map


def detect_column_type(series: pd.Series) -> str:
    """
    Detect type of a single column.
    
    Detection order:
    1. Boolean (if all values in [True, False, 0, 1])
    2. Numeric (if convertible to float)
    3. Datetime (if parseable as date)
    4. Categorical (if <50% unique values)
    5. Text (default)
    """
    # Skip if all nulls
    if series.isnull().all():
        return "text"
    
    # Check boolean
    unique_vals = series.dropna().unique()
    if set(unique_vals).issubset({True, False, 0, 1, "True", "False"}):
        return "boolean"
    
    # Check numeric
    try:
        pd.to_numeric(series, errors='raise')
        return "numeric"
    except (ValueError, TypeError):
        pass
    
    # Check datetime
    try:
        pd.to_datetime(series, errors='raise')
        # Additional check: ensure it's not just numbers
        if not series.dtype in ['int64', 'float64']:
            return "datetime"
    except (ValueError, TypeError):
        pass
    
    # Check categorical (< 50% unique)
    if len(unique_vals) / len(series) < 0.5:
        return "categorical"
    
    return "text"


def infer_aggregation_functions(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Suggest appropriate aggregation functions for each column.
    
    Returns:
        Dict mapping column to list of suggested functions
    """
    type_map = detect_column_types(df)
    suggestions = {}
    
    for col, dtype in type_map.items():
        if dtype == "numeric":
            suggestions[col] = ["sum", "mean", "median", "min", "max", "std", "count"]
        elif dtype in ["categorical", "text"]:
            suggestions[col] = ["count", "nunique", "mode"]
        elif dtype == "datetime":
            suggestions[col] = ["min", "max", "count"]
        elif dtype == "boolean":
            suggestions[col] = ["sum", "count"]
    
    return suggestions
```

### 3.3 Validation Helpers (`validation_helpers.py`)

```python
"""Data validation utilities."""

import pandas as pd
from pathlib import Path
from typing import List, Optional


def validate_file_exists(file_path: Path) -> None:
    """Check if file exists."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not file_path.is_file():
        raise ValueError(f"Not a file: {file_path}")


def validate_columns_exist(
    df: pd.DataFrame,
    required_columns: List[str]
) -> None:
    """
    Check if required columns exist in DataFrame.
    
    Raises:
        ValueError if columns are missing
    """
    missing = set(required_columns) - set(df.columns)
    
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )


def validate_column_types(
    df: pd.DataFrame,
    type_requirements: dict[str, str]
) -> None:
    """
    Validate column types match requirements.
    
    Args:
        df: DataFrame to validate
        type_requirements: Dict mapping column -> expected type
    """
    for col, expected_type in type_requirements.items():
        if col not in df.columns:
            raise ValueError(f"Column not found: {col}")
        
        actual_type = str(df[col].dtype)
        
        if expected_type == "numeric" and actual_type not in ["int64", "float64"]:
            raise TypeError(
                f"Column '{col}' must be numeric, got {actual_type}"
            )


def validate_dataframe_not_empty(df: pd.DataFrame) -> None:
    """Check DataFrame is not empty."""
    if len(df) == 0:
        raise ValueError("DataFrame is empty")
```

### 3.4 Security Helpers (`security_helpers.py`)

```python
"""Security validation utilities."""

import re
from pathlib import Path
from typing import List, Set


ALLOWED_IMPORTS = {
    "pandas", "numpy", "datetime", "json", "math", "statistics",
    "typing", "collections", "itertools", "functools"
}

DANGEROUS_PATTERNS = [
    r'\beval\(',
    r'\bexec\(',
    r'\b__import__\(',
    r'\bcompile\(',
    r'\bglobals\(',
    r'\blocals\(',
]


def validate_safe_file_path(file_path: Path, allowed_dir: Path) -> None:
    """
    Validate file path is within allowed directory.
    
    Prevents directory traversal attacks.
    """
    # Resolve to absolute path
    resolved_path = file_path.resolve()
    allowed_resolved = allowed_dir.resolve()
    
    # Check if path is within allowed directory
    try:
        resolved_path.relative_to(allowed_resolved)
    except ValueError:
        raise SecurityError(
            f"File path outside allowed directory: {file_path}"
        )


def validate_safe_imports(code: str) -> None:
    """
    Check that code only imports from allowlist.
    
    Raises:
        SecurityError if unsafe imports detected
    """
    import ast
    
    tree = ast.parse(code)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split('.')[0]
                if module not in ALLOWED_IMPORTS:
                    raise SecurityError(f"Import not allowed: {module}")
        
        elif isinstance(node, ast.ImportFrom):
            module = node.module.split('.')[0] if node.module else ""
            if module and module not in ALLOWED_IMPORTS:
                raise SecurityError(f"Import not allowed: {module}")


def validate_no_dangerous_code(code: str) -> None:
    """
    Check for dangerous code patterns.
    
    Raises:
        SecurityError if dangerous patterns found
    """
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, code):
            raise SecurityError(f"Dangerous pattern detected: {pattern}")


class SecurityError(Exception):
    """Raised when security validation fails."""
    pass
```

---

## 4. Testing

### 4.1 CSV Helpers Tests

```python
def test_load_csv_with_types():
    """Test CSV loading with type detection."""
    # Create test CSV
    df = pd.DataFrame({
        "id": ["1", "2", "3"],
        "value": ["10.5", "20.3", "15.7"],
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"]
    })
    
    csv_path = Path("test.csv")
    df.to_csv(csv_path, index=False)
    
    loaded = load_csv_data_with_types(csv_path, auto_detect_types=True)
    
    assert loaded["id"].dtype == "int64"
    assert loaded["value"].dtype == "float64"
    assert pd.api.types.is_datetime64_any_dtype(loaded["date"])
    
    csv_path.unlink()
```

### 4.2 Type Detection Tests

```python
def test_detect_numeric_column():
    """Test numeric type detection."""
    series = pd.Series(["1", "2", "3.5"])
    
    assert detect_column_type(series) == "numeric"


def test_detect_categorical_column():
    """Test categorical detection."""
    series = pd.Series(["A", "B", "A", "C", "B", "A"])
    
    assert detect_column_type(series) == "categorical"
```

### 4.3 Security Tests

```python
def test_dangerous_code_detection():
    """Test detection of dangerous code."""
    dangerous_code = "eval('malicious code')"
    
    with pytest.raises(SecurityError):
        validate_no_dangerous_code(dangerous_code)


def test_import_validation():
    """Test import allowlist."""
    safe_code = "import pandas as pd"
    unsafe_code = "import subprocess"
    
    validate_safe_imports(safe_code)  # Should pass
    
    with pytest.raises(SecurityError):
        validate_safe_imports(unsafe_code)
```

---

## 5. Configuration

```yaml
utils:
  csv:
    auto_detect_types: true
    preview_rows: 10
  
  type_detection:
    categorical_threshold: 0.5  # <50% unique → categorical
  
  security:
    allowed_imports:
      - pandas
      - numpy
      - datetime
      - json
      - math
```

---

## 6. Dependencies

```txt
pandas>=2.0.0
numpy>=1.24.0
```

---

## 7. Implementation Checklist

- [ ] Create `csv_helpers.py` with load/save functions
- [ ] Create `type_detection.py` with auto-detection logic
- [ ] Create `validation_helpers.py` with validation functions
- [ ] Create `security_helpers.py` with security checks
- [ ] Add comprehensive unit tests (>95% coverage)
- [ ] Test with real CSV files
- [ ] Document all public functions
- [ ] Add examples in docstrings

---

**Estimated Lines of Code**: 500-700  
**Test Coverage Target**: >95%  
**Ready for Implementation**: ✅
