"""Generated MCP tool: groupby_aggregate_weather_condition_injuries_fatal"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def groupby_aggregate_weather_condition_injuries_fatal(file_path: str):
    try:
        # Load the data from CSV file
        df = pd.read_csv(file_path)

        # Check for required columns
        required_columns = ["weather_condition", "injuries_fatal"]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"CSV file must contain the following columns: {required_columns}"
            )

        # Select and drop NaN values from required columns
        df = df[required_columns].dropna()

        # Filter rows where injuries_fatal > 0
        filtered_df = df[df["injuries_fatal"] > 0]

        # Group by weather_condition and count the number of fatal injuries
        grouped_results = (
            filtered_df.groupby("weather_condition")["injuries_fatal"]
            .count()
            .reset_index()
        )

        # Rename columns for clarity in result
        grouped_results.columns = ["weather_condition", "fatal_injury_count"]

        # Prepare metadata
        metadata = {
            "total_rows": len(df),
            "filtered_rows": len(filtered_df),
            "grouped_conditions": len(grouped_results),
        }

        return {
            "result": grouped_results.to_dict(orient="records"),
            "metadata": metadata,
        }

    except FileNotFoundError:
        return {"result": {}, "metadata": {"error": "File not found"}}
    except pd.errors.EmptyDataError:
        return {"result": {}, "metadata": {"error": "CSV file is empty"}}
    except ValueError as ve:
        return {"result": {}, "metadata": {"error": str(ve)}}
    except Exception as e:
        return {
            "result": {},
            "metadata": {"error": f"An unexpected error occurred: {str(e)}"},
        }
