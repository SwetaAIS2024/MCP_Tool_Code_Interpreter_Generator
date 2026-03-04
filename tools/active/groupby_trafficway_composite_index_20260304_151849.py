"""Generated MCP tool: groupby_trafficway_composite_index"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def groupby_trafficway_composite_index(file_path: str):
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
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Drop rows with NaN values in the required columns
        df = df[required_columns].dropna()

        # Calculate the composite index
        df["composite_index"] = (
            df["injuries_total"] + 2 * df["injuries_fatal"] + df["num_units"]
        )

        # Group by trafficway_type and calculate the mean of the composite index
        grouped_df = (
            df.groupby("trafficway_type")["composite_index"].mean().reset_index()
        )

        # Filter out groups with less than 2 samples
        grouped_df = grouped_df[
            grouped_df["trafficway_type"].map(df["trafficway_type"].value_counts()) >= 2
        ]

        # Check if there are at least 2 groups remaining
        if len(grouped_df) < 2:
            raise ValueError(
                "Not enough data to perform grouping with at least 2 samples per group."
            )

        # Prepare the result and metadata
        result = grouped_df.to_dict(orient="records")
        metadata = {
            "columns": required_columns,
            "num_samples": len(df),
            "num_groups": len(grouped_df),
        }

        return {"result": result, "metadata": metadata}

    except Exception as e:
        return {"result": {}, "metadata": {"error": str(e)}}
