"""Generated MCP tool: weather_injuries_correlation"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np
from scipy.stats import pearsonr

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def weather_injuries_correlation(file_path: str):
    try:
        # Load the data
        df = pd.read_csv(file_path)

        # Check for required columns
        required_columns = ["weather_condition", "injuries_total"]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"CSV file must contain the following columns: {', '.join(required_columns)}"
            )

        # Select required columns and drop NaN values
        df = df[required_columns].dropna()

        # Convert 'weather_condition' to a categorical variable if necessary
        if not pd.api.types.is_categorical_dtype(df["weather_condition"]):
            df["weather_condition"] = df["weather_condition"].astype("category")

        # Encode 'weather_condition' to numerical values
        df["weather_condition_encoded"] = df["weather_condition"].cat.codes

        # Check if there are at least 2 unique weather conditions
        if df["weather_condition_encoded"].nunique() < 2:
            raise ValueError(
                "There must be at least 2 unique weather conditions to calculate correlation."
            )

        # Calculate Pearson correlation coefficient and p-value
        correlation_coefficient, p_value = pearsonr(
            df["weather_condition_encoded"], df["injuries_total"]
        )

        # Interpretation of the results
        interpretation = (
            "No correlation"
            if correlation_coefficient == 0
            else (
                "Positive correlation"
                if correlation_coefficient > 0
                else "Negative correlation"
            )
        )

        # Return the result and metadata
        return {
            "result": {
                "correlation_coefficient": correlation_coefficient,
                "p_value": p_value,
                "interpretation": interpretation,
            },
            "metadata": {"file_path": file_path, "num_samples": len(df)},
        }

    except Exception as e:
        return {"result": None, "metadata": {"error": str(e)}}
