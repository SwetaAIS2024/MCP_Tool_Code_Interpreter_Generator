from fastmcp import FastMCP
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def crash_severity_custom_transform(file_path: str):
    try:
        # Load the data from CSV file
        df = pd.read_csv(file_path)

        # Check for required columns
        required_columns = [
            "injuries_total",
            "injuries_fatal",
            "num_units",
            "trafficway_type",
        ]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"CSV must contain the following columns: {required_columns}"
            )

        # Drop rows with missing values in required columns
        df = df[required_columns].dropna()

        # Ensure data types are correct
        if not (
            pd.api.types.is_numeric_dtype(df["injuries_total"])
            and pd.api.types.is_numeric_dtype(df["injuries_fatal"])
            and pd.api.types.is_numeric_dtype(df["num_units"])
        ):
            raise ValueError(
                "Data types of injuries_total, injuries_fatal, or num_units are incorrect."
            )

        # Create a new column 'crash_severity_index'
        df["crash_severity_index"] = (
            df["injuries_fatal"] * 2 + df["injuries_total"]
        ) / df["num_units"]

        # Normalize the crash_severity_index values using Min-Max Scaler
        scaler = MinMaxScaler()
        df["normalized_severity"] = scaler.fit_transform(df[["crash_severity_index"]])

        # Group by trafficway_type and calculate the average normalized severity
        grouped = (
            df.groupby("trafficway_type")["normalized_severity"].mean().reset_index()
        )

        # Sort the results in descending order based on the average normalized severity
        result_df = grouped.sort_values(by="normalized_severity", ascending=False)

        # Prepare the result as a list of dictionaries
        result = result_df.to_dict(orient="records")

        # Return the result with metadata
        return {
            "result": result,
            "metadata": {
                "count": len(result),
                "generated_at": datetime.now().isoformat(),
            },
        }

    except Exception as e:
        return {
            "result": [],
            "metadata": {"error": str(e), "generated_at": datetime.now().isoformat()},
        }
