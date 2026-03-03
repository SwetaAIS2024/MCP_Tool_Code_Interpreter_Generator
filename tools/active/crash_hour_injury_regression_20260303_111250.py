"""Generated MCP tool: crash_hour_injury_regression"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def crash_hour_injury_regression(file_path: str):
    try:
        # Load the data and select the required columns
        df = pd.read_csv(file_path)
        required_columns = ["crash_hour", "injuries_total"]

        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"CSV file must contain the following columns: {required_columns}"
            )

        df = df[required_columns].dropna()

        # Group the data by crash_hour to calculate the average number of injuries per hour
        grouped_data = df.groupby("crash_hour")["injuries_total"].mean().reset_index()

        # Prepare data for linear regression
        X = grouped_data["crash_hour"]
        y = grouped_data["injuries_total"]

        # Add a constant to the predictor variable set
        X = sm.add_constant(X)

        # Apply a linear regression model using statsmodels OLS
        model = ols("y ~ X", data=grouped_data).fit()

        # Extract coefficients and R-squared value from the regression model
        coefficients = model.params.to_dict()
        r_squared = model.rsquared

        result = {"coefficients": coefficients, "r_squared": r_squared}

        metadata = {
            "description": "Linear regression analysis of injuries across different hours of the day",
            "data_points": len(grouped_data),
        }

    except Exception as e:
        return {"result": {}, "metadata": {"error": str(e)}}

    return {"result": result, "metadata": metadata}
