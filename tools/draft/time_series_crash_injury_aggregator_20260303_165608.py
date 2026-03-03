"""Generated MCP tool: time_series_crash_injury_aggregator"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def time_series_crash_injury_aggregator(file_path: str):
    try:
        # Load data from CSV file
        df = pd.read_csv(file_path)

        # Check for required columns
        required_columns = ["crash_date", "most_severe_injury"]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"CSV file must contain the following columns: {required_columns}"
            )

        # Drop rows with missing values in required columns
        df = df[required_columns].dropna()

        # Parse the crash_date column to datetime
        df["crash_date"] = pd.to_datetime(df["crash_date"])

        # Extract year-month period
        df["period"] = df["crash_date"].dt.to_period("M")

        # Group by 'period' and count injuries
        injury_counts = df.groupby("period").size().reset_index(name="injury_count")

        # Sort by period ascending
        injury_counts = injury_counts.sort_values(by="period")

        # Prepare result as a list of dictionaries
        result = injury_counts.to_dict(orient="records")

        # Metadata for the function execution
        metadata = {"status": "success", "message": "Data aggregated successfully"}

    except Exception as e:
        result = []
        metadata = {"status": "error", "message": str(e)}

    return {"result": result, "metadata": metadata}
