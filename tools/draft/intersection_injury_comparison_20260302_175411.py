"""Generated MCP tool: intersection_injury_comparison"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np
from scipy.stats import shapiro, ttest_ind, mannwhitneyu

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def intersection_injury_comparison(file_path: str):
    try:
        # Load data
        df = pd.read_csv(file_path)

        # Select required columns
        df = df[["intersection_related_i", "injuries_total"]]

        # Split the data into two groups
        group_y = df[df["intersection_related_i"] == "Y"]["injuries_total"]
        group_not_y = df[df["intersection_related_i"] != "Y"]["injuries_total"]

        # Check if both groups have enough data for normality test
        if len(group_y) < 3 or len(group_not_y) < 3:
            raise ValueError(
                "Not enough data in one of the groups to perform normality test."
            )

        # Check normality using Shapiro-Wilk test
        _, p_value_y = shapiro(group_y)
        _, p_value_not_y = shapiro(group_not_y)

        normal_y = p_value_y > 0.05
        normal_not_y = p_value_not_y > 0.05

        # Perform appropriate statistical test
        if normal_y and normal_not_y:
            stat, p_value = ttest_ind(group_y, group_not_y)
            effect_size = (group_y.mean() - group_not_y.mean()) / np.sqrt(
                (group_y.var() + group_not_y.var()) / 2
            )
            test_used = "t-test"
        else:
            stat, p_value = mannwhitneyu(group_y, group_not_y)
            effect_size = (np.mean(group_y) - np.mean(group_not_y)) / np.sqrt(
                (np.std(group_y) ** 2 + np.std(group_not_y) ** 2) / 2
            )
            test_used = "Mann-Whitney U"

        # Determine significance
        significant = p_value < 0.05

        # Prepare result dictionary
        result = {
            "test_used": test_used,
            "statistic": stat,
            "p_value": p_value,
            "significant": significant,
            "effect_size": effect_size,
            "group_means": {"Y": group_y.mean(), "not_Y": group_not_y.mean()},
            "group_counts": {"Y": len(group_y), "not_Y": len(group_not_y)},
        }

        # Prepare metadata dictionary
        metadata = {"tool_name": "intersection_injury_comparison", "version": "1.0"}

        return {"result": result, "metadata": metadata}

    except Exception as e:
        return {
            "result": {"error": str(e)},
            "metadata": {
                "tool_name": "intersection_injury_comparison",
                "version": "1.0",
            },
        }
