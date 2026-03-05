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

        # Drop rows with missing values in required columns
        df = df[required_columns].dropna()

        # Parse crash_date column
        df["crash_date"] = pd.to_datetime(df["crash_date"])

        # Extract year from crash_date
        df["year"] = df["crash_date"].dt.year

        # Group by year and calculate total injuries
        aggregated_data = df.groupby("year")["injuries_total"].sum().reset_index()

        # Sort the results by year in ascending order
        aggregated_data = aggregated_data.sort_values(by="year").reset_index(drop=True)

        # Rename columns for result
        aggregated_data.columns = ["year", "total_injuries"]

        # Prepare result and metadata
        result = aggregated_data.to_dict(orient="records")
        metadata = {"status": "success", "message": "Data aggregated successfully"}

        return {"result": result, "metadata": metadata}

    except Exception as e:
        metadata = {"status": "error", "message": str(e)}
        return {"result": {}, "metadata": metadata}
