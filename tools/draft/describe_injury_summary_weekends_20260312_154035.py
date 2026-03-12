from fastmcp import FastMCP
import pandas as pd
import numpy as np
import scipy.stats as stats
import time

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def describe_injury_summary_weekends(file_path: str):
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Define the required columns
        required_columns = [
            "crash_day_of_week",
            "injuries_total",
            "injuries_fatal",
            "injuries_incapacitating",
            "injuries_non_incapacitating",
            "injuries_reported_not_evident",
            "injuries_no_indication",
        ]

        # Check if all required columns are present
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in the CSV file: {missing_columns}")

        # Filter the DataFrame to include only the required columns
        df = df[required_columns]

        # Drop rows with any NaN values in the required columns
        df = df.dropna()

        # Filter data to include only weekend days (Saturday and Sunday)
        weekend_df = df[df["crash_day_of_week"].isin(["Saturday", "Sunday"])]

        # Check if there are enough data points
        if len(weekend_df) < 2:
            raise ValueError("Not enough data points to compute statistics.")

        # Define the injury columns to compute statistics for
        injury_columns = [
            "injuries_total",
            "injuries_fatal",
            "injuries_incapacitating",
            "injuries_non_incapacitating",
            "injuries_reported_not_evident",
            "injuries_no_indication",
        ]

        # Compute descriptive statistics for each injury type
        summary_stats = {}
        for column in injury_columns:
            stats_dict = {
                "count": weekend_df[column].count(),
                "mean": weekend_df[column].mean(),
                "median": weekend_df[column].median(),
                "std": weekend_df[column].std(),
                "min": weekend_df[column].min(),
                "max": weekend_df[column].max(),
                "p25": weekend_df[column].quantile(0.25),
                "p75": weekend_df[column].quantile(0.75),
            }
            summary_stats[column] = stats_dict

        # Prepare the result and metadata
        result = {"summary_statistics": summary_stats}
        metadata = {
            "file_path": file_path,
            "number_of_weekend_crashes": len(weekend_df),
            "run_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        return {"result": result, "metadata": metadata}

    except Exception as e:
        return {"result": {}, "metadata": {"error": str(e)}}
