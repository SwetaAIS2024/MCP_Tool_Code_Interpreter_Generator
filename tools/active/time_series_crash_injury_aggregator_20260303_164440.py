"""Generated MCP tool: time_series_crash_injury_aggregator"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np

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

        # Convert crash_month to datetime and ensure it's sorted
        df["crash_month"] = pd.to_datetime(df["crash_month"])
        df.sort_values(by="crash_month", inplace=True)

        # Group by month and sum injuries_total
        monthly_injuries = df.resample("ME", on="crash_month").sum().reset_index()

        # Create a date range to identify missing months
        all_months = pd.date_range(
            start=monthly_injuries["crash_month"].min(),
            end=monthly_injuries["crash_month"].max(),
            freq="ME",
        )

        # Reindex the DataFrame to include all months, filling missing values with 0
        monthly_injuries.set_index("crash_month", inplace=True)
        monthly_injuries = monthly_injuries.reindex(
            all_months, fill_value=0
        ).reset_index()
        monthly_injuries.rename(columns={"index": "crash_month"}, inplace=True)

        # Prepare the result dictionary
        result = {"monthly_injuries": monthly_injuries.to_dict(orient="records")}

        # Prepare metadata
        metadata = {
            "total_months": len(monthly_injuries),
            "first_month": monthly_injuries["crash_month"].min().strftime("%Y-%m"),
            "last_month": monthly_injuries["crash_month"].max().strftime("%Y-%m"),
        }

        return {"result": result, "metadata": metadata}

    except Exception as e:
        return {"result": {}, "metadata": {"error": str(e)}}
