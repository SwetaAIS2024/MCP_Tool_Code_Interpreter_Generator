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
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Check for required columns
        required_columns = ["crash_month", "injuries_total"]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"CSV must contain the following columns: {required_columns}"
            )

        # Drop rows with missing values in required columns
        df = df[required_columns].dropna()

        # Convert crash_month to datetime format and extract year-month
        df["crash_month"] = pd.to_datetime(df["crash_month"]).dt.to_period("M")

        # Group by 'crash_month' and sum 'injuries_total'
        aggregated_data = df.groupby("crash_month", as_index=False)[
            "injuries_total"
        ].sum()

        # Sort the data by 'crash_month'
        aggregated_data.sort_values(by="crash_month", inplace=True)

        # Create a date range from the minimum to the maximum crash month
        min_date = aggregated_data["crash_month"].min()
        max_date = aggregated_data["crash_month"].max()
        full_date_range = pd.date_range(
            start=min_date.to_timestamp(), end=max_date.to_timestamp(), freq="M"
        ).to_period("M")

        # Create a DataFrame with the full date range
        full_date_df = pd.DataFrame({"crash_month": full_date_range})

        # Merge the aggregated data with the full date range to insert missing months
        result_df = pd.merge(
            full_date_df, aggregated_data, on="crash_month", how="left"
        ).fillna(0)

        # Convert 'injuries_total' back to integer after filling NaNs with 0
        result_df["injuries_total"] = result_df["injuries_total"].astype(int)

        # Prepare the result dictionary
        result_dict = {
            "result": result_df.to_dict("records"),
            "metadata": {
                "min_date": min_date.strftime("%Y-%m"),
                "max_date": max_date.strftime("%Y-%m"),
            },
        }

        return result_dict

    except Exception as e:
        return {"result": {}, "metadata": {"error": str(e)}}
