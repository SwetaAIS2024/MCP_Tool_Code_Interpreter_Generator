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

        # Drop rows with NaN values in required columns
        df = df[required_columns].dropna()

        # Filter records where injuries_fatal is greater than 0
        filtered_df = df[df["injuries_fatal"] > 0]

        # Group by weather_condition and calculate the count of fatal injuries for each group
        grouped_data = (
            filtered_df.groupby("weather_condition")
            .size()
            .reset_index(name="fatal_injury_count")
        )

        # Sort the results to identify which weather conditions have the highest and lowest counts of fatal injuries
        sorted_grouped_data = grouped_data.sort_values(
            by="fatal_injury_count", ascending=False
        )

        # Convert the result to a dictionary with 'weather_condition' and 'count'
        result_dict = sorted_grouped_data.rename(
            columns={"fatal_injury_count": "count"}
        ).to_dict(orient="records")

        # Prepare metadata
        metadata = {
            "total_records": len(df),
            "filtered_records": len(filtered_df),
            "unique_weather_conditions": len(grouped_data),
            "operation": "groupby_aggregate",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        return {"result": result_dict, "metadata": metadata}

    except FileNotFoundError:
        return {"error": "File not found"}
    except pd.errors.EmptyDataError:
        return {"error": "No data in the file"}
    except Exception as e:
        return {"error": str(e)}
