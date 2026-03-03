"""Generated MCP tool: time_series_crash_injury_aggregator"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np
from datetime import datetime

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def time_series_crash_injury_aggregator(file_path: str):
    try:
        # Load data from CSV file
        df = pd.read_csv(file_path)

        # Check for required columns
        required_columns = ["crash_date", "injuries_total"]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"CSV must contain the following columns: {required_columns}"
            )

        # Drop rows with missing values in required columns
        df = df[required_columns].dropna()

        # Convert crash_date to datetime format
        df["crash_date"] = pd.to_datetime(df["crash_date"], errors="coerce")

        # Extract month and year from crash_date
        df["year_month"] = df["crash_date"].dt.to_period("M")

        # Group by year_month and sum injuries_total
        aggregated_df = df.groupby("year_month")["injuries_total"].sum().reset_index()

        # Sort results chronologically
        aggregated_df.sort_values(by="year_month", inplace=True)

        # Convert year_month to string for readability in result
        aggregated_df["year_month"] = aggregated_df["year_month"].dt.strftime("%Y-%m")

        # Prepare result and metadata
        result = {
            "year_month": aggregated_df["year_month"].tolist(),
            "total_injuries": aggregated_df["injuries_total"].tolist(),
        }
        metadata = {
            "file_path": file_path,
            "number_of_records": len(df),
            "number_of_aggregated_periods": len(aggregated_df),
        }

        return {"result": result, "metadata": metadata}

    except Exception as e:
        return {"result": {}, "metadata": {"error": str(e)}}
