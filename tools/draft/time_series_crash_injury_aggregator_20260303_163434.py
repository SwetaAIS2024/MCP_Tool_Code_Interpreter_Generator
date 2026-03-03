from fastmcp import FastMCP
import pandas as pd
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

        # Group data by 'crash_month' and sum 'injuries_total'
        grouped_data = df.groupby("crash_month").sum().reset_index()

        # Create a date range to ensure all months are included
        min_date = df["crash_month"].min()
        max_date = df["crash_month"].max()
        full_date_range = pd.date_range(
            start=min_date.to_timestamp(), end=max_date.to_timestamp(), freq="M"
        ).to_period("M")

        # Reindex the grouped data to include all months, filling missing values with 0
        grouped_data.set_index("crash_month", inplace=True)
        full_grouped_data = grouped_data.reindex(
            full_date_range, fill_value=0
        ).reset_index()

        # Rename columns for clarity in output
        full_grouped_data.rename(
            columns={
                "crash_month": "crash_month",
                "injuries_total": "sum_injuries_total",
            },
            inplace=True,
        )

        # Prepare the result dictionary
        result = {"data": full_grouped_data.to_dict(orient="records")}

        # Prepare the metadata dictionary
        metadata = {
            "operation": "time_series_crash_injury_aggregation",
            "timestamp": datetime.now().isoformat(),
        }

        return {"result": result, "metadata": metadata}

    except Exception as e:
        return {"error": str(e)}
