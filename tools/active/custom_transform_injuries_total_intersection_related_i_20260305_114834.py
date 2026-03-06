"""Generated MCP tool: custom_transform_injuries_total_intersection_related_i"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np
from scipy.stats import shapiro, ttest_ind, mannwhitneyu

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def custom_transform_injuries_total_intersection_related_i(file_path: str):
    try:
        # Load data
        df = pd.read_csv(file_path)

        # Check for required columns
        required_columns = ["injuries_total", "intersection_related_i"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(
                f"CSV file must contain the columns: {', '.join(required_columns)}"
            )

        # Drop rows with NaN values in required columns
        df = df[required_columns].dropna()

        # Split data into two groups based on intersection_related_i
        group_y = df[df["intersection_related_i"] == "Y"]["injuries_total"]
        group_n = df[df["intersection_related_i"] != "Y"]["injuries_total"]

        # Filter groups with at least 2 samples
        if len(group_y) < 2 or len(group_n) < 2:
            raise ValueError("Each group must have at least 2 samples.")

        # Check normality of injuries_total distribution in each group using Shapiro-Wilk test
        normality_y = shapiro(group_y)
        normality_n = shapiro(group_n)

        # Perform statistical test based on normality
        if normality_y.pvalue > 0.05 and normality_n.pvalue > 0.05:
            # Both groups are normally distributed, perform independent samples t-test
            test_result = ttest_ind(group_y, group_n)
            test_used = "t-test"
        else:
            # Any group is not normally distributed, perform Mann-Whitney U test
            test_result = mannwhitneyu(group_y, group_n)
            test_used = "mann-whitney-u"

        # Compute effect size (Cohen's d)
        mean_y = group_y.mean()
        mean_n = group_n.mean()
        std_y = group_y.std(ddof=1)
        std_n = group_n.std(ddof=1)
        n_y = len(group_y)
        n_n = len(group_n)

        cohens_d = (mean_y - mean_n) / np.sqrt(
            ((n_y - 1) * std_y**2 + (n_n - 1) * std_n**2) / (n_y + n_n - 2)
        )

        # Prepare result dictionary
        result = {
            "test_statistic": test_result.statistic,
            "p_value": test_result.pvalue,
            "significant": test_result.pvalue < 0.05,
            "effect_size": cohens_d,
            "group_means": {"Y": mean_y, "not-Y": mean_n},
            "test_used": test_used,
        }

        # Prepare metadata dictionary
        metadata = {
            "normality_y": {
                "statistic": normality_y.statistic,
                "p_value": normality_y.pvalue,
            },
            "normality_n": {
                "statistic": normality_n.statistic,
                "p_value": normality_n.pvalue,
            },
        }

        return {"result": result, "metadata": metadata}

    except Exception as e:
        return {"result": {}, "metadata": {"error": str(e)}}
