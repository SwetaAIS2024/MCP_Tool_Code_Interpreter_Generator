"""Generated MCP tool: groupby_weather_injuries"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def groupby_weather_injuries(file_path: str):
    try:
        # Load data from CSV file
        df = pd.read_csv(file_path)

        # Check for required columns
        required_columns = ["weather_condition", "injuries_total"]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"CSV file must contain the following columns: {required_columns}"
            )

        # Drop rows with missing values in required columns
        df = df[required_columns].dropna()

        # Convert injuries_total to numeric, raising error if conversion fails
        df["injuries_total"] = pd.to_numeric(df["injuries_total"], errors="raise")

        # Group by 'weather_condition' and sum 'injuries_total'
        grouped_data = (
            df.groupby("weather_condition")["injuries_total"].sum().reset_index()
        )

        # Filter out groups with less than 2 samples
        grouped_data = grouped_data[grouped_data["injuries_total"] >= 1]

        if len(grouped_data) < 2:
            raise ValueError(
                "Not enough data to perform group-by aggregation. Ensure at least two weather conditions have valid injury counts."
            )

        # Sort results by injury count ascendingly
        result_df = grouped_data.sort_values(by="injuries_total", ascending=True)

        # Prepare the result and metadata dictionaries
        result = result_df.to_dict(orient="records")
        metadata = {
            "status": "success",
            "message": "Group-by aggregation completed successfully.",
        }

    except Exception as e:
        result = {}
        metadata = {"status": "error", "message": str(e)}

    return {"result": result, "metadata": metadata}
