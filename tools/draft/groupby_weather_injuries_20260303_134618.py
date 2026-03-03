"""Generated MCP tool: groupby_weather_injuries"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def groupby_weather_injuries(file_path: str):
    try:
        # Load the data and select the required columns
        df = pd.read_csv(file_path)
        required_columns = ["weather_condition", "injuries_fatal"]

        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"CSV file must contain the following columns: {required_columns}"
            )

        # Drop rows with NaN values in the required columns
        df = df[required_columns].dropna()

        # Filter the data to include only rows where injuries_fatal > 0
        filtered_df = df[df["injuries_fatal"] > 0]

        if filtered_df.empty:
            return {
                "result": {},
                "metadata": {"message": "No fatal injuries found after filtering"},
            }

        # Group the filtered data by the weather_condition column and calculate the count of fatal injuries
        grouped_results = (
            filtered_df.groupby("weather_condition")
            .agg({"injuries_fatal": "count"})
            .reset_index()
        )

        # Rename columns for clarity in the result
        grouped_results.columns = ["weather_condition", "fatal_injury_count"]

        # Convert the result to a dictionary
        result_dict = grouped_results.to_dict(orient="records")

        return {"result": result_dict, "metadata": {}}

    except FileNotFoundError:
        return {"result": {}, "metadata": {"error": "File not found"}}
    except pd.errors.EmptyDataError:
        return {"result": {}, "metadata": {"error": "CSV file is empty"}}
    except Exception as e:
        return {"result": {}, "metadata": {"error": str(e)}}
