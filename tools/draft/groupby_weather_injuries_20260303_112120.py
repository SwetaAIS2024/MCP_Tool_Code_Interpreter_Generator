"""Generated MCP tool: groupby_weather_injuries"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def groupby_weather_injuries(file_path: str):
    try:
        # Load the dataset from the specified file path
        df = pd.read_csv(file_path)

        # Check for required columns
        required_columns = ["weather_condition", "injuries_fatal"]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"CSV must contain the following columns: {required_columns}"
            )

        # Filter out rows where injuries_fatal is not a number or is less than or equal to 0
        df = df[required_columns].dropna()
        df = df[df["injuries_fatal"] > 0]

        # Group by 'weather_condition' and calculate the sum of fatal injuries for each group
        grouped_data = (
            df.groupby("weather_condition")["injuries_fatal"].sum().reset_index()
        )

        # Convert the result to a dictionary
        result_dict = grouped_data.to_dict(orient="records")

        # Prepare metadata
        metadata = {
            "total_records": len(df),
            "unique_weather_conditions": len(grouped_data),
        }

        return {"result": result_dict, "metadata": metadata}

    except FileNotFoundError:
        return {"error": "File not found", "metadata": {}}
    except pd.errors.EmptyDataError:
        return {"error": "No data in the file", "metadata": {}}
    except Exception as e:
        return {"error": str(e), "metadata": {}}
