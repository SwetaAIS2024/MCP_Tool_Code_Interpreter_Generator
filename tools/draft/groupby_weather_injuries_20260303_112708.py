"""Generated MCP tool: groupby_weather_injuries"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def groupby_weather_injuries(file_path: str):
    try:
        # Load dataset
        df = pd.read_csv(file_path)

        # Check for required columns
        required_columns = ["weather_condition", "injuries_fatal"]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"CSV file must contain the following columns: {required_columns}"
            )

        # Select and drop NaN values from required columns
        df = df[required_columns].dropna()

        # Filter rows where 'injuries_fatal' > 0
        filtered_df = df[df["injuries_fatal"] > 0]

        # Group data by 'weather_condition'
        grouped_data = filtered_df.groupby("weather_condition")

        # Count records per group
        injury_counts = grouped_data.size().reset_index(name="count")

        # Filter groups with at least 2 samples
        valid_groups = injury_counts[injury_counts["count"] >= 2]

        if len(valid_groups) < 2:
            raise ValueError(
                "Not enough data to perform the group-by aggregation. Each weather condition must have at least 2 records with injuries_fatal > 0."
            )

        # Prepare result dictionary
        result_dict = valid_groups.set_index("weather_condition")["count"].to_dict()

        return {
            "result": result_dict,
            "metadata": {
                "total_records": len(filtered_df),
                "valid_groups_count": len(valid_groups),
            },
        }

    except FileNotFoundError:
        return {"error": "File not found. Please check the file path."}
    except pd.errors.EmptyDataError:
        return {"error": "The CSV file is empty."}
    except ValueError as ve:
        return {"error": str(ve)}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}
