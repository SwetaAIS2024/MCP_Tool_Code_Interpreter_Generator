"""Generated MCP tool: groupby_weather_injuries"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def groupby_weather_injuries(file_path: str):
    try:
        # Load the data from CSV file
        df = pd.read_csv(file_path)

        # Check for required columns
        required_columns = ["injuries_fatal", "weather_condition"]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"CSV must contain the following columns: {required_columns}"
            )

        # Drop rows with NaN values in required columns
        df = df[required_columns].dropna()

        # Ensure data types are correct
        if not pd.api.types.is_numeric_dtype(df["injuries_fatal"]):
            raise TypeError("Column 'injuries_fatal' must be numeric.")

        # Group by 'weather_condition' and calculate the count of 'injuries_fatal'
        grouped_data = (
            df.groupby("weather_condition")["injuries_fatal"].count().reset_index()
        )

        # Rename columns for clarity
        grouped_data.columns = ["weather_condition", "fatal_injury_count"]

        # Sort the results in descending order based on the count of fatal injuries
        sorted_grouped_data = grouped_data.sort_values(
            by="fatal_injury_count", ascending=False
        )

        # Filter out groups with less than 2 samples
        filtered_sorted_grouped_data = sorted_grouped_data[
            sorted_grouped_data["fatal_injury_count"] >= 2
        ]

        # Ensure there are at least 2 groups remaining
        if len(filtered_sorted_grouped_data) < 2:
            raise ValueError(
                "Not enough data to perform the operation. At least 2 groups with 2 or more samples are required."
            )

        # Prepare result and metadata
        result = filtered_sorted_grouped_data.to_dict(orient="records")
        metadata = {
            "total_groups": len(filtered_sorted_grouped_data),
            "file_path": file_path,
        }

        return {"result": result, "metadata": metadata}

    except Exception as e:
        return {"result": {}, "metadata": {"error": str(e)}}
