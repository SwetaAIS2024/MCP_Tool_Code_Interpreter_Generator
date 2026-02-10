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
                f"CSV file must contain the following columns: {required_columns}"
            )

        # Drop rows with NaN values in required columns
        df = df[required_columns].dropna()

        # Ensure 'injuries_fatal' is of integer type
        if not pd.api.types.is_integer_dtype(df["injuries_fatal"]):
            raise TypeError("Column 'injuries_fatal' must be of integer type.")

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

        # Convert the result to a dictionary
        result_dict = sorted_grouped_data.to_dict(orient="records")

        # Prepare metadata
        metadata = {
            "total_records": len(df),
            "unique_weather_conditions": df["weather_condition"].nunique(),
            "groups_with_fatal_injuries": len(result_dict),
        }

        return {"result": result_dict, "metadata": metadata}

    except FileNotFoundError:
        return {"error": "File not found. Please check the file path."}
    except pd.errors.EmptyDataError:
        return {"error": "The CSV file is empty."}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
