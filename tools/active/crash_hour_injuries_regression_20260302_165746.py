"""Generated MCP tool: crash_hour_injuries_regression"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def crash_hour_injuries_regression(file_path: str):
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)

        # Validate required columns
        required_columns = ["crash_hour", "injuries_total"]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"CSV file must contain the following columns: {required_columns}"
            )

        # Drop rows with missing values in required columns
        df = df[required_columns].dropna()

        # Group by crash_hour and calculate sum of injuries_total for each hour
        grouped_data = df.groupby("crash_hour")["injuries_total"].sum().reset_index()

        # Ensure there are at least 2 groups (hours)
        if len(grouped_data) < 2:
            raise ValueError(
                "There must be at least 2 different crash hours in the data."
            )

        # Prepare data for regression
        X = grouped_data["crash_hour"]
        y = grouped_data["injuries_total"]

        # Add a constant to the independent variable
        X = sm.add_constant(X)

        # Fit linear regression model
        model = OLS(y, X).fit()

        # Extract coefficients, p-values, and R-squared from the regression model
        coefficients = model.params.to_dict()
        p_values = model.pvalues.to_dict()
        r_squared = model.rsquared

        # Format results to include significance of crash_hour in explaining injuries_total
        result = {
            "coefficients": coefficients,
            "p_values": p_values,
            "r_squared": r_squared,
        }

        metadata = {
            "message": "Linear regression analysis completed successfully.",
            "data_points_used": len(grouped_data),
        }

    except Exception as e:
        result = {}
        metadata = {
            "error": str(e),
            "message": "An error occurred during the analysis.",
        }

    return {"result": result, "metadata": metadata}
