"""Generated MCP tool: traffic_control_device_injury_aggregation"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def traffic_control_device_injury_aggregation(file_path: str):
    try:
        # Load the data from CSV
        df = pd.read_csv(file_path)

        # Define required columns
        required_columns = [
            "traffic_control_device",
            "injuries_total",
            "injuries_fatal",
            "injuries_incapacitating",
            "injuries_non_incapacitating",
        ]

        # Check if all required columns are present
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in the CSV file: {missing_columns}")

        # Drop rows with NaN values in any of the required columns
        df = df[required_columns].dropna()

        # Group by 'traffic_control_device' and calculate sums for each injury type
        grouped_data = (
            df.groupby("traffic_control_device")
            .agg(
                {
                    "injuries_total": "sum",
                    "injuries_fatal": "sum",
                    "injuries_incapacitating": "sum",
                    "injuries_non_incapacitating": "sum",
                }
            )
            .reset_index()
        )

        # Sort the results by 'traffic_control_device' in ascending order
        grouped_data = grouped_data.sort_values(
            by="traffic_control_device", ascending=True
        )

        # Prepare the result dictionary
        result = grouped_data.to_dict(orient="records")

        # Prepare metadata
        metadata = {"total_groups": len(grouped_data), "file_path": file_path}

        return {"result": result, "metadata": metadata}

    except FileNotFoundError:
        return {"error": "File not found", "metadata": {}}
    except pd.errors.EmptyDataError:
        return {"error": "No data in the CSV file", "metadata": {}}
    except ValueError as ve:
        return {"error": str(ve), "metadata": {}}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}", "metadata": {}}
