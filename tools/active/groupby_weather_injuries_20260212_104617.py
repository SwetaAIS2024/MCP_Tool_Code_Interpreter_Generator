from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def groupby_weather_injuries(file_path: str):
    """
    Groups data by weather_condition and calculates the count of fatal injuries for each group
    after filtering records where injuries_fatal > 0.

    Parameters:
    file_path (str): Path to CSV file

    Returns:
    dict: A dictionary with 'result' and 'metadata' keys
    """

    try:
        # Load the data from the CSV file
        df = pd.read_csv(file_path)

        # Check for required columns
        required_columns = ["weather_condition", "injuries_fatal"]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"CSV file must contain the following columns: {required_columns}"
            )

        # Filter out rows where injuries_fatal is not greater than 0
        filtered_df = df[df["injuries_fatal"] > 0]

        # Drop NaN values from the required columns
        filtered_df = filtered_df[required_columns].dropna()

        # Group by weather_condition and calculate the count of fatal injuries
        grouped = (
            filtered_df.groupby("weather_condition").size().reset_index(name="count")
        )

        # Sort the results to identify which weather conditions have the highest and lowest counts of fatal injuries
        grouped = grouped.sort_values(by="count", ascending=False)

        # Prepare the result in the required format
        result = grouped.to_dict(orient="records")

        # Prepare metadata
        metadata = {
            "operation": "groupby_aggregate",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        return {"result": result, "metadata": metadata}

    except Exception as e:
        return {"result": [], "metadata": {"error": str(e)}}
