"""Generated MCP tool: injuries_num_units_linear_regression"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np
from scipy.stats import linregress
import statsmodels.api as sm

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def injuries_num_units_linear_regression(file_path: str):
    try:
        # Load the data from CSV file
        df = pd.read_csv(file_path)

        # Check for required columns
        required_columns = ["injuries_total", "num_units"]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"CSV file must contain the following columns: {required_columns}"
            )

        # Drop rows with NaN values in the required columns
        df = df[required_columns].dropna()

        # Check if there are enough data points
        if len(df) < 2:
            raise ValueError("Not enough data points to perform linear regression.")

        # Define independent and dependent variables
        X = df["num_units"]
        y = df["injuries_total"]

        # Add a constant to the independent variable for statsmodels OLS
        X_with_const = sm.add_constant(X)

        # Perform linear regression using statsmodels OLS
        model = sm.OLS(y, X_with_const).fit()

        # Extract coefficients and R-squared value
        slope = model.params["num_units"]
        intercept = model.params["const"]
        r_squared = model.rsquared

        # Prepare the result dictionary
        result = {"slope": slope, "intercept": intercept, "r_squared": r_squared}

        # Prepare the metadata dictionary
        metadata = {"method": "OLS", "data_points": len(df)}

        return {"result": result, "metadata": metadata}

    except Exception as e:
        return {"error": str(e)}
