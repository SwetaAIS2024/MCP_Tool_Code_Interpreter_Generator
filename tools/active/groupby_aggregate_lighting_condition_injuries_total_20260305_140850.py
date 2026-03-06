from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def groupby_aggregate_lighting_condition_injuries_total(file_path: str):
    try:
        # Load the data
        df = pd.read_csv(file_path)

        # Check for required columns
        required_columns = ["lighting_condition", "injuries_total"]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"CSV file must contain the following columns: {', '.join(required_columns)}"
            )

        # Select required columns and drop rows with missing values
        df = df[required_columns].dropna()

        # Filter out groups with less than 2 samples
        group_counts = df["lighting_condition"].value_counts()
        valid_groups = group_counts[group_counts >= 2].index
        df = df[df["lighting_condition"].isin(valid_groups)]

        # Create two groups: 'DAYLIGHT' and not 'DAYLIGHT'
        daylight_group = df[df["lighting_condition"] == "DAYLIGHT"]
        non_daylight_group = df[df["lighting_condition"] != "DAYLIGHT"]

        # Check if both groups have at least 2 samples
        if len(daylight_group) < 2 or len(non_daylight_group) < 2:
            raise ValueError(
                "Both 'DAYLIGHT' and non-'DAYLIGHT' groups must have at least 2 samples."
            )

        # Compute descriptive statistics for each group
        def compute_stats(group):
            return {
                "count": group["injuries_total"].count(),
                "mean": group["injuries_total"].mean(),
                "median": group["injuries_total"].median(),
                "std": group["injuries_total"].std(),
                "min": group["injuries_total"].min(),
                "max": group["injuries_total"].max(),
                "25th_percentile": group["injuries_total"].quantile(0.25),
                "75th_percentile": group["injuries_total"].quantile(0.75),
            }

        daylight_stats = compute_stats(daylight_group)
        non_daylight_stats = compute_stats(non_daylight_group)

        # Calculate comparison metrics
        absolute_diff_means = daylight_stats["mean"] - non_daylight_stats["mean"]
        percentage_diff = (
            (absolute_diff_means / non_daylight_stats["mean"]) * 100
            if non_daylight_stats["mean"] != 0
            else np.nan
        )
        higher_mean_group = (
            "DAYLIGHT"
            if daylight_stats["mean"] > non_daylight_stats["mean"]
            else "not_DAYLIGHT"
        )

        # Prepare the result
        result = {
            "DAYLIGHT": daylight_stats,
            "not_DAYLIGHT": non_daylight_stats,
            "comparison": {
                "absolute_diff_means": absolute_diff_means,
                "percentage_diff": percentage_diff,
                "higher_mean_group": higher_mean_group,
            },
        }

        # Prepare metadata
        metadata = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_file": file_path,
        }

        return {"result": result, "metadata": metadata}

    except Exception as e:
        return {"error": str(e)}
