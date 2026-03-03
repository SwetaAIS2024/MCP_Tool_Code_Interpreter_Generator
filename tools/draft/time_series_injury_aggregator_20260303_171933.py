"""Generated MCP tool: time_series_injury_aggregator"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def time_series_injury_aggregator(file_path: str):
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

        # Parse the full date column using pd.to_datetime
        df["crash_date"] = pd.to_datetime(df["crash_date"])

        # Extract year-month period
        df["period"] = df["crash_date"].dt.to_period("M")

        # Group by 'period' and aggregate injuries_total using sum
        aggregated_df = df.groupby("period")["injuries_total"].sum().reset_index()

        # Sort by period ascending
        aggregated_df = aggregated_df.sort_values(by="period").reset_index(drop=True)

        # Prepare the result list of dictionaries
        result_list = [
            {"period": str(row["period"]), "injury_count": int(row["injuries_total"])}
            for _, row in aggregated_df.iterrows()
        ]

        return {"result": result_list, "metadata": {}}

    except Exception as e:
        return {"result": [], "metadata": {"error": str(e)}}
