"""Generated MCP tool: groupby_aggregate_crash_day_of_week_injuries"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np
import scipy.stats as stats

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def groupby_aggregate_crash_day_of_week_injuries(file_path: str):
    try:
        # Load the dataset
        df = pd.read_csv(file_path)

        # Required columns
        required_columns = [
            "crash_day_of_week",
            "injuries_total",
            "injuries_fatal",
            "injuries_incapacitating",
            "injuries_non_incapacitating",
            "injuries_reported_not_evident",
            "injuries_no_indication",
        ]

        # Check for missing columns
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in the dataset: {missing_columns}")

        # Filter the data to include only records where crash_day_of_week is 1 (Sunday) or 7 (Saturday)
        df = df[df["crash_day_of_week"].isin([1, 7])]

        # Drop rows with NaN values in required columns
        df = df[required_columns].dropna()

        # Group the filtered data by crash_day_of_week
        grouped = df.groupby("crash_day_of_week")

        # Calculate descriptive statistics for each group
        descriptive_stats = {}
        for name, group in grouped:
            stats_dict = group.describe().T.to_dict()
            descriptive_stats[name] = stats_dict

        # Calculate absolute and percentage difference between group means
        sunday_mean = grouped.mean().loc[1]
        saturday_mean = grouped.mean().loc[7]

        absolute_diff = sunday_mean - saturday_mean
        percent_diff = (absolute_diff / sunday_mean) * 100

        difference = {
            "absolute_diff": absolute_diff.to_dict(),
            "percent_diff": percent_diff.to_dict(),
        }

        # Prepare the result dictionary
        result = {"comparison": descriptive_stats, "difference": difference}

        # Prepare the metadata dictionary
        metadata = {"file_path": file_path, "groups": list(descriptive_stats.keys())}

        return {"result": result, "metadata": metadata}

    except Exception as e:
        return {"result": None, "metadata": {"error": str(e)}}
