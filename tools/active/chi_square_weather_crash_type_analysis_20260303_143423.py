"""Generated MCP tool: chi_square_weather_crash_type_analysis"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np
from scipy.stats import chi2_contingency

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def chi_square_weather_crash_type_analysis(file_path: str):
    try:
        # Load the data from CSV file
        df = pd.read_csv(file_path)

        # Check for required columns
        required_columns = ["weather_condition", "crash_type"]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"CSV must contain the following columns: {required_columns}"
            )

        # Select and drop missing values from required columns
        df = df[required_columns].dropna()

        # Ensure both variables are categorical
        df["weather_condition"] = df["weather_condition"].astype("category")
        df["crash_type"] = df["crash_type"].astype("category")

        # Create a contingency table
        contingency_table = pd.crosstab(df["weather_condition"], df["crash_type"])

        # Perform chi-square test of independence
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

        # Determine significance at 0.05 level
        significant = p_value < 0.05

        # Prepare result and metadata
        result = {
            "chi_square_statistic": chi2_stat,
            "p_value": p_value,
            "significant": significant,
        }
        metadata = {
            "degrees_of_freedom": dof,
            "expected_frequencies": expected.tolist(),
        }

        return {"result": result, "metadata": metadata}

    except Exception as e:
        return {"error": str(e)}
