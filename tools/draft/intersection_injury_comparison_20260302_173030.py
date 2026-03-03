"""Generated MCP tool: intersection_injury_comparison"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np
from scipy.stats import ttest_ind, shapiro, levene, mannwhitneyu

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def intersection_injury_comparison(file_path: str):
    try:
        # Load data from CSV file
        df = pd.read_csv(file_path)

        # Check for required columns
        required_columns = ["intersection_related_i", "injuries_total"]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"CSV must contain the following columns: {required_columns}"
            )

        # Drop rows with NaN values in required columns
        df = df[required_columns].dropna()

        # Filter dataset into two groups based on 'intersection_related_i'
        group_intersection = df[df["intersection_related_i"] == "Y"]["injuries_total"]
        group_non_intersection = df[df["intersection_related_i"] != "Y"][
            "injuries_total"
        ]

        # Check for minimum samples in each group
        if len(group_intersection) < 2 or len(group_non_intersection) < 2:
            raise ValueError("Each group must have at least 2 samples.")

        # Calculate summary statistics for both groups
        stats_intersection = {
            "mean": group_intersection.mean(),
            "median": group_intersection.median(),
            "std_dev": group_intersection.std(),
        }
        stats_non_intersection = {
            "mean": group_non_intersection.mean(),
            "median": group_non_intersection.median(),
            "std_dev": group_non_intersection.std(),
        }

        # Check assumptions for t-test: normality and homogeneity of variances
        normality_test_intersection = shapiro(group_intersection)
        normality_test_non_intersection = shapiro(group_non_intersection)
        variance_test = levene(group_intersection, group_non_intersection)

        assumptions_met = True

        if (
            normality_test_intersection.pvalue < 0.05
            or normality_test_non_intersection.pvalue < 0.05
        ):
            assumptions_met = False
        if variance_test.pvalue < 0.05:
            assumptions_met = False

        # Perform statistical test based on assumption check
        if assumptions_met:
            t_test_result = ttest_ind(group_intersection, group_non_intersection)
            p_value = t_test_result.pvalue
            test_type = "t-test"
        else:
            mann_whitneyu_result = mannwhitneyu(
                group_intersection, group_non_intersection
            )
            p_value = mann_whitneyu_result.pvalue
            test_type = "Mann-Whitney U test"

        # Interpret p-value to determine statistical significance
        is_significant = p_value < 0.05

        # Prepare result and metadata dictionaries
        result = {
            "test_type": test_type,
            "p_value": p_value,
            "is_significant": is_significant,
        }

        metadata = {
            "group_intersection_stats": stats_intersection,
            "group_non_intersection_stats": stats_non_intersection,
            "normality_test_intersection": normality_test_intersection._asdict(),
            "normality_test_non_intersection": normality_test_non_intersection._asdict(),
            "variance_test": variance_test._asdict(),
        }

        return {"result": result, "metadata": metadata}

    except Exception as e:
        return {"result": {}, "metadata": {"error": str(e)}}
