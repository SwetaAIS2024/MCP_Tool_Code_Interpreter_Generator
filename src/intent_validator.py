"""Intent Validator - Validate extracted intent before spec/code generation."""

from typing import Dict, List, Tuple, Optional
from src.logger_config import get_logger, log_section, log_error, log_success

logger = get_logger(__name__)


class IntentValidator:
    """Validates extracted intent for logical consistency and completeness."""
    
    def validate(self, intent: Dict, available_columns: List[str]) -> Tuple[bool, List[str], List[str]]:
        """Validate extracted intent before proceeding to spec generation.
        
        Args:
            intent: Extracted intent dictionary
            available_columns: List of available columns from dataset
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []
        
        # Extract key fields
        operation = intent.get("operation", "")
        required_columns = intent.get("required_columns", [])
        missing_columns = intent.get("missing_columns", [])
        has_gap = intent.get("has_gap", False)
        implementation_plan = intent.get("implementation_plan", [])
        
        # ============================================================================
        # CRITICAL CHECKS - Must pass
        # ============================================================================
        
        # Check 1: Required columns must not be empty (for most operations)
        if operation not in ["describe_summary"] and not required_columns:
            errors.append(f"Operation '{operation}' requires columns, but required_columns is empty")
        
        # Check 2: All required columns must exist in available_columns
        for col in required_columns:
            if col not in available_columns:
                errors.append(f"Required column '{col}' not found in dataset. Available: {available_columns[:10]}")
        
        # Check 3: If missing_columns has items, this is a blocker
        if missing_columns:
            errors.append(f"Missing columns detected: {missing_columns}. Cannot proceed without these columns.")
        
        # Check 4: Operation-specific column count requirements
        operation_errors = self._validate_operation_columns(operation, required_columns)
        errors.extend(operation_errors)
        
        # Check 5: Statistical operations must have implementation plan
        if has_gap and operation == "custom_transform":
            if not implementation_plan:
                errors.append("Statistical operation detected but implementation_plan is empty")
            else:
                # Validate plan mentions statistical method or library
                plan_str = str(implementation_plan).lower()
                
                # Comprehensive list of statistical keywords
                statistical_keywords = [
                    # Basic tests
                    'anova', 't-test', 'ttest', 'z-test', 'f-test', 'chi-square', 'chi2', 'chi_square',
                    # Correlation
                    'correlation', 'pearson', 'spearman', 'kendall', 'corr',
                    # Regression
                    'regression', 'linear model', 'logistic', 'glm', 'poisson', 'ols',
                    # Post-hoc tests
                    'tukey', 'bonferroni', 'scheffe', 'dunnett', 'sidak', 'post-hoc', 'posthoc',
                    # Non-parametric tests
                    'mann-whitney', 'wilcoxon', 'kruskal', 'friedman', 'sign test',
                    # Normality & homogeneity tests
                    'shapiro', 'kolmogorov', 'anderson', 'levene', 'bartlett',
                    # Effect size
                    'effect size', 'cohen', 'eta-squared', 'omega-squared', 'eta_squared',
                    # Multivariate
                    'pca', 'factor analysis', 'manova', 'discriminant',
                    # Clustering
                    'cluster', 'k-means', 'kmeans', 'hierarchical', 'dbscan',
                    # Time series
                    'arima', 'sarima', 'time series', 'autocorrelation', 'acf', 'pacf', 'forecast',
                    # Resampling
                    'bootstrap', 'permutation', 'monte carlo', 'cross-validation',
                    # Statistical libraries (as fallback)
                    'scipy.stats', 'statsmodels', 'scikit-learn', 'sklearn',
                    'f_oneway', 'ttest_ind', 'chi2_contingency', 'pearsonr', 'spearmanr'
                ]
                
                if any(keyword in plan_str for keyword in statistical_keywords):
                    # Good - plan mentions statistical method or library
                    pass
                else:
                    # Not necessarily an error - user might be doing custom analysis
                    # Just warn that we couldn't identify a standard statistical method
                    warnings.append("Statistical operation but couldn't identify standard statistical method in plan (may be custom analysis)")
        
        # ============================================================================
        # WARNING CHECKS - Non-critical but should be reviewed
        # ============================================================================
        
        # Warning 1: Too many columns for simple operations
        if operation in ["groupby_aggregate", "filter"] and len(required_columns) > 5:
            warnings.append(f"Operation '{operation}' has {len(required_columns)} columns - this may be too complex")
        
        # Warning 2: Single column for comparison operations
        if operation == "groupby_aggregate" and len(required_columns) < 2:
            warnings.append(f"groupby_aggregate typically needs 2+ columns (grouping + metric), found {len(required_columns)}")
        
        # Note: Warning 3 is now integrated into Check 5 above
        
        return len(errors) == 0, errors, warnings
    
    def _validate_operation_columns(self, operation: str, required_columns: List[str]) -> List[str]:
        """Validate operation has appropriate number/type of columns.
        
        Args:
            operation: Operation type
            required_columns: List of required column names
            
        Returns:
            List of error messages
        """
        errors = []
        col_count = len(required_columns)
        
        if operation == "groupby_aggregate":
            # Need at least 2 columns: grouping column + metric column
            if col_count < 2:
                errors.append(f"groupby_aggregate requires at least 2 columns (grouping + metric), found {col_count}")
        
        elif operation == "pivot":
            # Need at least 2 columns: index + values (columns optional)
            if col_count < 2:
                errors.append(f"pivot requires at least 2 columns (index + values), found {col_count}")
        
        elif operation == "filter":
            # Need at least 1 column to filter on
            if col_count < 1:
                errors.append(f"filter requires at least 1 column, found {col_count}")
        
        elif operation == "custom_transform":
            # Statistical operations need specific column counts
            # ANOVA: need grouping column + metric (2+)
            # Correlation: need 2 numeric columns
            # Regression: need 2+ columns (predictors + target)
            if col_count < 2:
                errors.append(f"custom_transform (statistical) typically requires 2+ columns, found {col_count}")
        
        elif operation == "sort_limit":
            # Need at least 1 column to sort by
            if col_count < 1:
                errors.append(f"sort_limit requires at least 1 column, found {col_count}")
        
        return errors


def validate_intent(intent: Dict, available_columns: List[str]) -> Tuple[bool, List[str], List[str]]:
    """Helper function to validate intent.
    
    Args:
        intent: Extracted intent
        available_columns: Available dataset columns
        
    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    validator = IntentValidator()
    return validator.validate(intent, available_columns)


def log_validation_results(is_valid: bool, errors: List[str], warnings: List[str]) -> None:
    """Log validation results with appropriate formatting.
    
    Args:
        is_valid: Whether validation passed
        errors: List of error messages
        warnings: List of warning messages
    """
    log_section(logger, "INTENT VALIDATION")
    
    if is_valid:
        log_success(logger, "✅ Intent validation passed")
    else:
        log_error(logger, f"❌ Intent validation failed with {len(errors)} errors")
    
    if errors:
        logger.error(f"\nERRORS ({len(errors)}):")
        for i, err in enumerate(errors, 1):
            logger.error(f"  {i}. {err}")
    
    if warnings:
        logger.warning(f"\nWARNINGS ({len(warnings)}):")
        for i, warn in enumerate(warnings, 1):
            logger.warning(f"  {i}. {warn}")
