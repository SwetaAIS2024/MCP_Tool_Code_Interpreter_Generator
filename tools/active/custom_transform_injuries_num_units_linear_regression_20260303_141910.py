"""Generated MCP tool: custom_transform_injuries_num_units_linear_regression"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np
from scipy.stats import linregress

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def custom_transform_injuries_num_units_linear_regression(file_path: str):
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
        df_cleaned = df[required_columns].dropna()

        # Check if there are enough data points
        if len(df_cleaned) < 2:
            raise ValueError(
                "Not enough data points to perform linear regression. At least two non-NaN data points are required."
            )

        # Perform linear regression analysis
        slope, intercept, r_value, p_value, _ = linregress(
            df_cleaned["num_units"], df_cleaned["injuries_total"]
        )

        # Calculate R-squared value
        r_squared = r_value**2

        # Prepare the result dictionary
        result = {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
            "p_value": p_value,
        }

        # Prepare the metadata dictionary
        metadata = {"num_samples": len(df_cleaned)}

        return {"result": result, "metadata": metadata}

    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_path} was not found.")
    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file is empty.")
    except Exception as e:
        raise RuntimeError(f"An error occurred: {str(e)}")
