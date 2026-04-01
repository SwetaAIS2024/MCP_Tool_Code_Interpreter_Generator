from fastmcp import FastMCP
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import time

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

        # Ensure there are at least two unique weather conditions
        if df["weather_condition"].nunique() < 2:
            raise ValueError(
                "There must be at least two unique weather conditions to calculate correlation."
            )

        # Encode 'weather_condition' to numerical values for correlation calculation
        df["weather_condition_encoded"] = df["weather_condition"].cat.codes

        # Calculate Pearson correlation coefficient and p-value
        correlation_coefficient, p_value = pearsonr(
            df["weather_condition_encoded"], df["injuries_total"]
        )

        # Interpretation of the results
        interpretation = (
            "No correlation"
            if abs(correlation_coefficient) < 0.1
            else (
                "Weak correlation"
                if abs(correlation_coefficient) < 0.3
                else (
                    "Moderate correlation"
                    if abs(correlation_coefficient) < 0.5
                    else (
                        "Strong correlation"
                        if abs(correlation_coefficient) < 0.7
                        else "Very strong correlation"
                    )
                )
            )
        )

        # Return the result and metadata
        return {
            "result": {
                "correlation_coefficient": correlation_coefficient,
                "p_value": p_value,
                "interpretation": interpretation,
            },
            "metadata": {
                "status": "success",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "file_path": file_path,
                "num_samples": len(df),
            },
        }

    except Exception as e:
        return {
            "result": None,
            "metadata": {
                "status": "error",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(e),
            },
        }
