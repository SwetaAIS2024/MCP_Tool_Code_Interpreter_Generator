"""Generated MCP tool: time_series_aggregate_crash_date_injuries_total"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def time_series_aggregate_crash_date_injuries_total(file_path: str):
    try:
        # Load data from CSV file
        df = pd.read_csv(file_path)

        # Check for required columns
        required_columns = ["crash_date", "injuries_total"]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"CSV file must contain the following columns: {required_columns}"
            )

        # Drop rows with missing values in required columns
        df = df[required_columns].dropna()

        # Parse the crash_date column to datetime
        df["crash_date"] = pd.to_datetime(df["crash_date"])

        # Extract year period from crash_date
        df["period"] = df["crash_date"].dt.to_period("Y")

        # Group by 'period' and aggregate sum of injuries_total
        aggregated_df = df.groupby("period")["injuries_total"].sum().reset_index()

        # Convert period to string for JSON serialization
        aggregated_df["period"] = aggregated_df["period"].astype(str)

        # Prepare result and metadata
        result = aggregated_df.to_dict(orient="records")
        metadata = {"status": "success", "message": "Data processed successfully"}

    except Exception as e:
        result = []
        metadata = {"status": "error", "message": str(e)}

    return {"result": result, "metadata": metadata}
