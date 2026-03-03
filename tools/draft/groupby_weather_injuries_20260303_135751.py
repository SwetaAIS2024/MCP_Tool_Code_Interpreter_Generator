"""Generated MCP tool: groupby_weather_injuries"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def groupby_weather_injuries(file_path: str):
    try:
        # Step 1: Load the dataset
        df = pd.read_csv(file_path)

        # Check for required columns
        required_columns = ["weather_condition", "injuries_fatal"]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"CSV file must contain the following columns: {required_columns}"
            )

        # Handle NaN values by dropping rows with missing values in required columns
        df = df[required_columns].dropna()

        # Step 2: Filter the data to include only rows where injuries_fatal > 0
        filtered_df = df[df["injuries_fatal"] > 0]

        # Guard for minimum samples: Ensure there are at least 2 groups with >=2 samples each
        group_counts = filtered_df["weather_condition"].value_counts()
        valid_groups = group_counts[group_counts >= 2].index
        if len(valid_groups) < 2:
            raise ValueError(
                "There must be at least two weather conditions with at least two records each where injuries_fatal > 0."
            )

        # Filter the dataframe to include only valid groups
        filtered_df = filtered_df[filtered_df["weather_condition"].isin(valid_groups)]

        # Step 3: Group the filtered data by weather_condition
        grouped_data = filtered_df.groupby("weather_condition")

        # Step 4: Calculate the count of records for each group
        result_counts = grouped_data.size().to_dict()

        # Step 5: Return the grouped counts as the final result
        return {"result": result_counts, "metadata": {}}

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
