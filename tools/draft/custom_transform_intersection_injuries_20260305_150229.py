"""Generated MCP tool: custom_transform_intersection_injuries"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np
from scipy.stats import shapiro, ttest_ind, mannwhitneyu

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def custom_transform_intersection_injuries(file_path: str):
    try:
        # Load data
        df = pd.read_csv(file_path)

        # Check for required columns
        required_columns = ["intersection_related_i", "injuries_total"]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"CSV file must contain the following columns: {', '.join(required_columns)}"
            )

        # Drop rows with missing values in required columns
        df = df[required_columns].dropna()

        # Split data into two groups based on 'intersection_related_i'
        group_y = df[df["intersection_related_i"] == "Y"]["injuries_total"]
        group_n = df[df["intersection_related_i"] != "Y"]["injuries_total"]

        # Filter groups with at least 2 samples
        if len(group_y) < 2 or len(group_n) < 2:
            raise ValueError(
                "Each group must have at least 2 samples for statistical testing."
            )

        # Check normality of 'injuries_total' in each group using Shapiro-Wilk test
        normality_y, p_value_y = shapiro(group_y)
        normality_n, p_value_n = shapiro(group_n)

        # Determine if data is normal
        is_normal_y = p_value_y > 0.05
        is_normal_n = p_value_n > 0.05

        # Perform appropriate statistical test
        if is_normal_y and is_normal_n:
            # Perform independent samples t-test
            test_result = ttest_ind(group_y, group_n)
            statistic = test_result.statistic
            p_value = test_result.pvalue
            test_used = "t-test"

            # Compute Cohen's d
            mean_y = group_y.mean()
            mean_n = group_n.mean()
            std_y = group_y.std(ddof=1)
            std_n = group_n.std(ddof=1)
            n_y = len(group_y)
            n_n = len(group_n)
            cohens_d = (mean_y - mean_n) / np.sqrt(
                ((n_y - 1) * std_y**2 + (n_n - 1) * std_n**2) / (n_y + n_n - 2)
            )
            effect_size = cohens_d
        else:
            # Perform Mann-Whitney U test
            test_result = mannwhitneyu(group_y, group_n)
            statistic = test_result.statistic
            p_value = test_result.pvalue
            test_used = "mann-whitney-u"
            effect_size = None

        # Determine significance
        significance = p_value < 0.05

        # Compute group means
        group_means = {"Y": group_y.mean(), "not-Y": group_n.mean()}

        # Prepare result dictionary
        result = {
            "statistic": statistic,
            "p_value": p_value,
            "significant": significance,
            "effect_size": effect_size,
            "group_means": group_means,
            "test_used": test_used,
        }

        # Prepare metadata dictionary
        metadata = {
            "normality": {
                "Y": {"statistic": normality_y, "p_value": p_value_y},
                "not-Y": {"statistic": normality_n, "p_value": p_value_n},
            }
        }

        return {"result": result, "metadata": metadata}

    except Exception as e:
        return {"result": {}, "metadata": {"error": str(e)}}
