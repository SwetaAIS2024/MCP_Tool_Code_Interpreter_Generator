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
            metadata = {
                "normality": {
                    "intersection_related_i_Y": {
                        "statistic": normality_test_intersection.statistic,
                        "p_value": normality_test_intersection.pvalue,
                    },
                    "intersection_related_i_not_Y": {
                        "statistic": normality_test_non_intersection.statistic,
                        "p_value": normality_test_non_intersection.pvalue,
                    },
                }
            }

        if variance_test.pvalue < 0.05:
            assumptions_met = False
            metadata["homogeneity_of_variances"] = {
                "statistic": variance_test.statistic,
                "p_value": variance_test.pvalue,
            }

        # Perform statistical test based on assumption check
        if assumptions_met:
            t_test_result = ttest_ind(group_intersection, group_non_intersection)
            result = {
                "t_statistic": t_test_result.statistic,
                "p_value": t_test_result.pvalue,
            }
        else:
            u_test_result = mannwhitneyu(group_intersection, group_non_intersection)
            result = {
                "u_statistic": u_test_result.statistic,
                "p_value": u_test_result.pvalue,
            }

        # Interpret p-value to determine statistical significance
        significant = result["p_value"] < 0.05

        # Prepare metadata with summary statistics and assumption checks
        metadata.update(
            {
                "summary_statistics": {
                    "intersection_related_i_Y": stats_intersection,
                    "intersection_related_i_not_Y": stats_non_intersection,
                },
                "assumptions_met": assumptions_met,
                "significant": significant,
            }
        )

        return {"result": result, "metadata": metadata}

    except Exception as e:
        return {"result": {}, "metadata": {"error": str(e)}}
