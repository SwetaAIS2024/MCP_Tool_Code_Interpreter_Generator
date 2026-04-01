"""Generated MCP tool: time_series_injury_aggregator"""

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend - must be set before pyplot
from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np
import io
import base64
import matplotlib.pyplot as plt

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def time_series_injury_aggregator(file_path: str):
    try:
        # Load data and select required columns
        df = pd.read_csv(file_path)
        required_columns = ["crash_date", "injuries_total"]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"CSV file must contain the following columns: {required_columns}"
            )

        df = df[required_columns].dropna()

        # Parse crash_date column using pd.to_datetime
        df["crash_date"] = pd.to_datetime(df["crash_date"])

        # Extract year-month period and store in 'period' column
        df["period"] = df["crash_date"].dt.to_period("M")

        # Group by 'period' and calculate sum of injuries_total
        monthly_injuries = df.groupby("period")["injuries_total"].sum().reset_index()

        # Sort results by period in ascending order
        monthly_injuries.sort_values(by="period", inplace=True)

        # Filter for January to March
        jan_mar_injuries = monthly_injuries[
            monthly_injuries["period"].dt.month.isin([1, 2, 3])
        ]

        # Generate line plot with period on x-axis and total injuries on y-axis
        plt.figure(figsize=(10, 6))
        plt.plot(
            jan_mar_injuries["period"].astype(str),
            jan_mar_injuries["injuries_total"],
            marker="o",
        )
        plt.title("Monthly Trend of Total Injuries (January to March)")
        plt.xlabel("Month-Year")
        plt.ylabel("Total Injuries")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save plot to in-memory buffer and encode as base64
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        plt.close("all")
        buf.seek(0)
        plot_b64 = base64.b64encode(buf.read()).decode("utf-8")

        # Return result along with base64-encoded plot
        return {
            "result": jan_mar_injuries.to_dict(),
            "metadata": {"plot_base64": plot_b64},
        }

    except Exception as e:
        return {"result": {}, "metadata": {"error": str(e)}}
