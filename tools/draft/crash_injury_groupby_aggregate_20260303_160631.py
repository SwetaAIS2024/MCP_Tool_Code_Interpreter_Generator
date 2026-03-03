"""Generated MCP tool: crash_injury_groupby_aggregate"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def crash_type_injury_analysis(file_path: str):
    try:
        # Load the dataset and select required columns
        df = pd.read_csv(file_path)
        required_columns = [
            "crash_type",
            "injuries_total",
            "injuries_fatal",
            "injuries_severe",
            "injuries_minor",
        ]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"CSV file must contain the following columns: {required_columns}"
            )

        # Group data by crash_type to aggregate injury statistics
        grouped = (
            df.groupby("crash_type")[
                [
                    "injuries_total",
                    "injuries_fatal",
                    "injuries_severe",
                    "injuries_minor",
                ]
            ]
            .mean()
            .reset_index()
        )

        # Prepare data for ANOVA
        anova_data = {
            "total": [
                df[df["crash_type"] == crash_type]["injuries_total"].dropna().values
                for crash_type in grouped["crash_type"]
            ],
            "fatal": [
                df[df["crash_type"] == crash_type]["injuries_fatal"].dropna().values
                for crash_type in grouped["crash_type"]
            ],
            "severe": [
                df[df["crash_type"] == crash_type]["injuries_severe"].dropna().values
                for crash_type in grouped["crash_type"]
            ],
            "minor": [
                df[df["crash_type"] == crash_type]["injuries_minor"].dropna().values
                for crash_type in grouped["crash_type"]
            ],
        }

        # Perform one-way ANOVA
        anova_results = {}
        post_hoc_tests = {}

        for injury_type, data_lists in anova_data.items():
            f_stat, p_value = f_oneway(*data_lists)
            anova_results[injury_type] = {"f_stat": f_stat, "p_value": p_value}

            # Perform post-hoc tests if significant
            if p_value < 0.05:
                tukey_hsd = pairwise_tukeyhsd(
                    endog=df["injuries_" + injury_type],
                    groups=df["crash_type"],
                    alpha=0.05,
                )
                post_hoc_tests[injury_type] = {
                    "groups": tukey_hsd.groupsunique.tolist(),
                    "test_statistic": tukey_hsd.meandiffs.tolist(),
                    "p_value": tukey_hsd.pvalues.tolist(),
                }

        # Prepare result dictionary
        result = {
            "injury_stats": grouped.to_dict(orient="records"),
            "anova_results": anova_results,
            "post_hoc_tests": post_hoc_tests,
        }

        metadata = {
            "creation_date": pd.Timestamp.now().isoformat(),
            "tool_name": "crash_type_injury_analysis",
        }

        return {"result": result, "metadata": metadata}

    except Exception as e:
        return {"result": {}, "metadata": {"error": str(e)}}
