"""Generated MCP tool: injury_intersection_analysis"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np
from scipy.stats import shapiro, ttest_ind, mannwhitneyu

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def injury_intersection_analysis(file_path: str):
    # Load data
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return {"result": {}, "metadata": {"error": f"Error loading file: {str(e)}"}}

    # Check required columns
    required_columns = ["intersection_related_i", "injuries_total"]
    if not all(col in df.columns for col in required_columns):
        missing_columns = [col for col in required_columns if col not in df.columns]
        return {
            "result": {},
            "metadata": {"error": f"Missing columns: {', '.join(missing_columns)}"},
        }

    # Drop rows with NaN values in required columns
    df = df[required_columns].dropna()

    # Split data into two groups based on 'intersection_related_i'
    group_y = df[df["intersection_related_i"] == "Y"]["injuries_total"]
    group_n = df[df["intersection_related_i"] != "Y"]["injuries_total"]

    # Filter groups with at least 2 samples
    if len(group_y) < 2 or len(group_n) < 2:
        return {
            "result": {},
            "metadata": {"error": "Each group must have at least 2 samples."},
        }

    # Check normality using Shapiro-Wilk test
    normality_y = shapiro(group_y)
    normality_n = shapiro(group_n)

    # Perform appropriate statistical test
    if normality_y.pvalue > 0.05 and normality_n.pvalue > 0.05:
        # Both groups are normal, perform t-test
        test_result = ttest_ind(group_y, group_n)
        test_used = "t-test"
        # Calculate Cohen's d
        mean1, mean2 = group_y.mean(), group_n.mean()
        std1, std2 = group_y.std(), group_n.std()
        n1, n2 = len(group_y), len(group_n)
        cohens_d = (mean1 - mean2) / np.sqrt(
            ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
        )
        effect_size = cohens_d
    else:
        # At least one group is non-normal, perform Mann-Whitney U test
        test_result = mannwhitneyu(group_y, group_n)
        test_used = "mann-whitney-u"
        effect_size = None

    # Determine significance
    significance = test_result.pvalue < 0.05

    # Prepare result dictionary
    result = {
        "statistic": test_result.statistic,
        "p_value": test_result.pvalue,
        "significant": significance,
        "effect_size": effect_size,
        "group_means": {"Y": group_y.mean(), "not-Y": group_n.mean()},
        "test_used": test_used,
    }

    # Prepare metadata dictionary
    metadata = {
        "normality": {
            "Y": {"statistic": normality_y.statistic, "p_value": normality_y.pvalue},
            "not-Y": {
                "statistic": normality_n.statistic,
                "p_value": normality_n.pvalue,
            },
        }
    }

    return {"result": result, "metadata": metadata}
