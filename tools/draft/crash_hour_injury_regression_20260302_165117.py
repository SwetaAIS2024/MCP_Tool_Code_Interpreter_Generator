"""Generated MCP tool: crash_hour_injury_regression"""

from fastmcp import FastMCP
import pandas as pd
import time
import statsmodels.api as sm
from statsmodels.formula.api import ols

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def crash_hour_injury_regression(file_path: str):
    try:
        # Read CSV file
        df = pd.read_csv(file_path)

        # Validate required columns
        required_columns = ["crash_hour", "injuries_total"]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"CSV must contain the following columns: {required_columns}"
            )

        # Drop rows with NaN values in required columns
        df = df[required_columns].dropna()

        # Group data by crash_hour and calculate total injuries for each hour
        grouped_data = df.groupby("crash_hour")["injuries_total"].sum().reset_index()

        # Ensure there are at least 2 groups
        if len(grouped_data) < 2:
            raise ValueError(
                "Not enough unique crash hours to perform regression analysis."
            )

        # Prepare data for regression
        X = grouped_data["crash_hour"]
        y = grouped_data["injuries_total"]

        # Add a constant to the independent variable
        X = sm.add_constant(X)

        # Fit linear regression model using OLS
        model = ols("y ~ X", data=grouped_data).fit()

        # Compute R-squared value
        r_squared = model.rsquared

        # Extract regression coefficients and p-values
        coefficients = model.params.to_dict()
        p_values = model.pvalues.to_dict()

        # Prepare result dictionary
        result = {
            "coefficients": coefficients,
            "p_values": p_values,
            "r_squared": r_squared,
        }

        # Prepare metadata dictionary
        metadata = {"num_samples": len(grouped_data), "columns_used": required_columns}

        return {"result": result, "metadata": metadata}

    except Exception as e:
        return {"result": {}, "metadata": {"error": str(e)}}
