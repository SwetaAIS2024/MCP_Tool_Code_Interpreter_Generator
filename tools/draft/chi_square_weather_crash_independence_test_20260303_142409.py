from fastmcp import FastMCP
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def chi_square_weather_crash_independence_test(file_path: str):
    try:
        # Load the data from CSV file
        df = pd.read_csv(file_path)

        # Check for required columns
        required_columns = ["weather_condition", "crash_type"]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"CSV file must contain the following columns: {required_columns}"
            )

        # Drop rows with NaN values in the required columns
        df = df[required_columns].dropna()

        # Create a contingency table
        contingency_table = pd.crosstab(df["weather_condition"], df["crash_type"])

        # Check if there are at least 2 groups and each group has at least 2 samples
        if len(contingency_table) < 2 or (contingency_table < 2).any().any():
            raise ValueError(
                "Contingency table must have at least 2 groups with at least 2 samples each."
            )

        # Perform chi-square test of independence
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

        # Determine significance at alpha=0.05
        significant = p_value < 0.05

        # Return the result and metadata
        return {
            "result": {
                "chi_square_statistic": chi2_stat,
                "p_value": p_value,
                "significant": significant,
            },
            "metadata": {"method": "chi2_contingency"},
        }

    except Exception as e:
        return {"error": str(e)}
