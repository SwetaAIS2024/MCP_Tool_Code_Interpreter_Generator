from fastmcp import FastMCP
import pandas as pd
import numpy as np

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def groupby_weather_injuries(file_path: str):
    try:
        # Load data from CSV file
        df = pd.read_csv(file_path)

        # Check for required columns
        required_columns = ["injuries_fatal", "weather_condition"]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"CSV must contain the following columns: {required_columns}"
            )

        # Drop rows with NaN values in required columns
        df = df[required_columns].dropna()

        # Convert injuries_fatal to integer type if necessary
        if not pd.api.types.is_integer_dtype(df["injuries_fatal"]):
            try:
                df["injuries_fatal"] = df["injuries_fatal"].astype(int)
            except ValueError:
                raise TypeError(
                    "Column 'injuries_fatal' must be convertible to integers."
                )

        # Group by weather_condition and calculate the count of injuries_fatal
        grouped_data = (
            df.groupby("weather_condition")["injuries_fatal"].count().reset_index()
        )

        # Rename columns for clarity
        grouped_data.columns = ["weather_condition", "fatal_injury_count"]

        # Sort results in descending order based on fatal injury count
        sorted_grouped_data = grouped_data.sort_values(
            by="fatal_injury_count", ascending=False
        )

        # Filter groups with at least one entry
        filtered_grouped_data = sorted_grouped_data[
            sorted_grouped_data["fatal_injury_count"] > 0
        ]

        # Prepare the result and metadata
        result = filtered_grouped_data.to_dict(orient="records")
        metadata = {"operation": "groupby_aggregate"}

        return {"result": result, "metadata": metadata}

    except Exception as e:
        return {"result": [], "metadata": {"error": str(e)}}
