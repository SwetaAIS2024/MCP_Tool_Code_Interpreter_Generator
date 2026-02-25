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

        # Ensure 'injuries_fatal' is of integer type
        if not pd.api.types.is_integer_dtype(df["injuries_fatal"]):
            raise TypeError("Column 'injuries_fatal' must be of integer type.")

        # Group by 'weather_condition' and count 'injuries_fatal'
        grouped_data = (
            df.groupby("weather_condition")["injuries_fatal"].count().reset_index()
        )

        # Filter groups with at least 2 samples
        grouped_data = grouped_data[grouped_data["injuries_fatal"] >= 2]

        if len(grouped_data) < 2:
            raise ValueError(
                "Not enough data to perform grouping and counting. Ensure there are at least 2 groups with >=2 samples each."
            )

        # Sort the results in descending order based on the count of fatal injuries
        sorted_grouped_data = grouped_data.sort_values(
            by="injuries_fatal", ascending=False
        )

        # Prepare the result dictionary
        result_dict = {
            "weather_condition": sorted_grouped_data["weather_condition"].tolist(),
            "fatal_injury_count": sorted_grouped_data["injuries_fatal"].tolist(),
        }

        # Prepare metadata
        metadata = {"total_groups": len(sorted_grouped_data), "file_path": file_path}

        return {"result": result_dict, "metadata": metadata}

    except FileNotFoundError:
        return {"error": f"File not found: {file_path}"}
    except pd.errors.EmptyDataError:
        return {"error": "CSV file is empty"}
    except Exception as e:
        return {"error": str(e)}
