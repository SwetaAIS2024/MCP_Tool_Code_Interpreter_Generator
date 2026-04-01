"""Generated MCP tool: custom_transform_lighting_condition_injuries_total"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np
from scipy.stats import shapiro, ttest_ind, mannwhitneyu

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def custom_transform_lighting_condition_injuries_total(file_path: str):
    try:
        # Load data
        df = pd.read_csv(file_path)

        # Check for required columns
        required_columns = ["lighting_condition", "injuries_total"]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"CSV file must contain the following columns: {', '.join(required_columns)}"
            )

        # Drop rows with missing values in required columns
        df = df[required_columns].dropna()

        # Split data into two groups based on lighting_condition
        daylight_group = df[df["lighting_condition"] == "DAYLIGHT"]["injuries_total"]
        non_daylight_group = df[df["lighting_condition"] != "DAYLIGHT"][
            "injuries_total"
        ]

        # Filter groups with at least 2 samples
        if len(daylight_group) < 2 or len(non_daylight_group) < 2:
            raise ValueError(
                "Both 'DAYLIGHT' and non-'DAYLIGHT' groups must have at least 2 samples."
            )

        # Check normality of injuries_total in each group using Shapiro-Wilk test
        _, daylight_p_value = shapiro(daylight_group)
        _, non_daylight_p_value = shapiro(non_daylight_group)

        # Determine if data is normally distributed
        daylight_normal = daylight_p_value > 0.05
        non_daylight_normal = non_daylight_p_value > 0.05

        # Perform appropriate statistical test
        if daylight_normal and non_daylight_normal:
            test_result = ttest_ind(daylight_group, non_daylight_group)
            test_statistic = test_result.statistic
            p_value = test_result.pvalue
            test_used = "t-test"
        else:
            test_result = mannwhitneyu(daylight_group, non_daylight_group)
            test_statistic = test_result.statistic
            p_value = test_result.pvalue
            test_used = "mann-whitney-u"

        # Compute effect size (Cohen's d)
        mean1, mean2 = daylight_group.mean(), non_daylight_group.mean()
        std1, std2 = daylight_group.std(), non_daylight_group.std()
        n1, n2 = len(daylight_group), len(non_daylight_group)
        cohens_d = (mean1 - mean2) / np.sqrt(
            ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
        )

        # Determine significance
        significant = p_value < 0.05

        # Prepare result dictionary
        result = {
            "test_statistic": test_statistic,
            "p_value": p_value,
            "significant": significant,
            "effect_size": cohens_d,
            "group_means": {"DAYLIGHT": mean1, "NON-DAYLIGHT": mean2},
            "test_used": test_used,
        }

        # Prepare metadata dictionary
        metadata = {
            "daylight_normal": daylight_normal,
            "non_daylight_normal": non_daylight_normal,
        }

        return {"result": result, "metadata": metadata}

    except Exception as e:
        return {"result": {}, "metadata": {"error": str(e)}}
