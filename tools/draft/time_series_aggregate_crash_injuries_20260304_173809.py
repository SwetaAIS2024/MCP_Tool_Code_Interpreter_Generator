"""Generated MCP tool: time_series_aggregate_crash_injuries"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def time_series_aggregate_crash_injuries(file_path: str):
    try:
        # Load data
        df = pd.read_csv(file_path)

        # Check for required columns
        required_columns = ["crash_date", "injuries_total"]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"CSV file must contain the following columns: {', '.join(required_columns)}"
            )

        # Handle missing values
        df = df[required_columns].dropna()

        # Parse crash_date column
        df["crash_date"] = pd.to_datetime(df["crash_date"])

        # Extract year from crash_date
        df["year"] = df["crash_date"].dt.year

        # Group by year and calculate sum of injuries_total
        aggregated_df = df.groupby("year")["injuries_total"].sum().reset_index()

        # Sort the result by year in ascending order
        aggregated_df = aggregated_df.sort_values(by="year", ascending=True)

        # Prepare result and metadata
        result = aggregated_df.to_dict(orient="records")
        metadata = {
            "description": "Yearly trend of total injuries from crash data",
            "columns": ["year", "injuries_total"],
        }

        return {"result": result, "metadata": metadata}

    except Exception as e:
        return {"result": {}, "metadata": {"error": str(e)}}
