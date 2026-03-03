"""Generated MCP tool: linear_regression_injuries_num_units"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np
from scipy.stats import linregress

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def linear_regression_injuries_num_units(file_path: str):
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

        # Perform linear regression analysis
        slope, intercept, r_value, p_value, std_err = linregress(
            df["num_units"], df["injuries_total"]
        )

        # Prepare the result dictionary
        result = {
            "slope": slope,
            "intercept": intercept,
            "r_value": r_value,
            "p_value": p_value,
            "std_err": std_err,
        }

        # Prepare the metadata dictionary
        metadata = {"sample_size": len(df), "columns_used": required_columns}

        return {"result": result, "metadata": metadata}

    except FileNotFoundError:
        return {"error": f"File not found: {file_path}"}
    except pd.errors.EmptyDataError:
        return {"error": "The file is empty."}
    except ValueError as ve:
        return {"error": str(ve)}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}
