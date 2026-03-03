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
        grouped_data = df.groupby("crash_month")["injuries_total"].sum().reset_index()

        # Create a date range from the minimum to the maximum crash month
        all_months = pd.date_range(
            start=grouped_data["crash_month"].min().to_timestamp(),
            end=grouped_data["crash_month"].max().to_timestamp(),
            freq="M",
        ).to_period("M")

        # Create a DataFrame with all months
        all_months_df = pd.DataFrame({"crash_month": all_months})

        # Merge to include all months, filling missing values with 0
        merged_data = pd.merge(
            all_months_df, grouped_data, on="crash_month", how="left"
        ).fillna(0)

        # Convert 'injuries_total' back to integer
        merged_data["injuries_total"] = merged_data["injuries_total"].astype(int)

        # Sort by crash_month in ascending order
        merged_data.sort_values(by="crash_month", inplace=True)

        # Return the result with metadata
        return {
            "result": {
                "data": merged_data.to_dict(orient="records"),
                "metadata": {
                    "operation": "time_series_crash_injury_aggregator",
                    "timestamp": datetime.now().isoformat(),
                },
            }
        }

    except Exception as e:
        return {
            "result": None,
            "metadata": {"error": str(e), "timestamp": datetime.now().isoformat()},
        }
