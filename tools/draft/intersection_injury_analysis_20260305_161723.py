"""Generated MCP tool: intersection_injury_analysis"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np
from scipy.stats import shapiro, ttest_ind, mannwhitneyu

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def intersection_injury_analysis(file_path: str):
    try:
        # Load data
        df = pd.read_csv(file_path)

        # Check for required columns
        required_columns = ["intersection_related_i", "injuries_total"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(
                f"CSV file must contain the following columns: {', '.join(required_columns)}"
            )

        # Select required columns and drop NaN values
        df = df[required_columns].dropna()

        # Split data into two groups based on 'intersection_related_i'
        group_y = df[df["intersection_related_i"] == "Y"]["injuries_total"]
        group_n = df[df["intersection_related_i"] != "Y"]["injuries_total"]

        # Filter groups with at least 2 samples
        if len(group_y) < 2 or len(group_n) < 2:
            raise ValueError("Each group must have at least 2 samples.")

        # Check normality of each group using Shapiro-Wilk test
        normality_y = shapiro(group_y).pvalue > 0.05
        normality_n = shapiro(group_n).pvalue > 0.05

        # Perform statistical test based on normality
        if normality_y and normality_n:
            # Perform independent samples t-test
            test_result = ttest_ind(group_y, group_n)
            statistic = test_result.statistic
            p_value = test_result.pvalue
            test_used = "t-test"

            # Compute Cohen's d
            mean1, mean2 = group_y.mean(), group_n.mean()
            std1, std2 = group_y.std(), group_n.std()
            n1, n2 = len(group_y), len(group_n)
            cohens_d = (mean1 - mean2) / np.sqrt(
                ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
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

        # Calculate group means
        group_means = {"Y": group_y.mean(), "not-Y": group_n.mean()}

        # Prepare result dictionary
        result = {
            "statistic": statistic,
            "p_value": p_value,
            "significance": significance,
            "effect_size": effect_size,
            "group_means": group_means,
            "test_used": test_used,
        }

        # Prepare metadata dictionary
        metadata = {"normality_y": normality_y, "normality_n": normality_n}

        return {"result": result, "metadata": metadata}

    except Exception as e:
        return {"result": {}, "metadata": {"error": str(e)}}
