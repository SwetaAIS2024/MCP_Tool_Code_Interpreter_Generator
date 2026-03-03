"""Generated MCP tool: groupby_weather_condition_year_injuries"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def groupby_weather_condition_year_injuries(file_path: str):
    try:
        # Load the dataset containing injury data
        df = pd.read_csv(file_path)

        # Check for required columns
        required_columns = ["weather_condition", "crash_date", "injuries_total"]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"CSV file must contain the following columns: {required_columns}"
            )

        # Select the required columns and drop rows with missing values
        df = df[required_columns].dropna()

        # Convert crash_date to datetime and extract year
        df["crash_date"] = pd.to_datetime(df["crash_date"])
        df["year"] = df["crash_date"].dt.year

        # Group by weather_condition and year, then sum injuries_total
        grouped_data = (
            df.groupby(["weather_condition", "year"])["injuries_total"]
            .sum()
            .reset_index()
        )

        # Prepare the result dictionary
        result = {}
        for index, row in grouped_data.iterrows():
            if row["weather_condition"] not in result:
                result[row["weather_condition"]] = {}
            result[row["weather_condition"]][row["year"]] = row["injuries_total"]

        # Prepare metadata
        metadata = {
            "total_records": len(df),
            "unique_weather_conditions": df["weather_condition"].nunique(),
            "years_range": (df["year"].min(), df["year"].max()),
        }

        return {"result": result, "metadata": metadata}

    except Exception as e:
        return {"result": {}, "metadata": {"error": str(e)}}
