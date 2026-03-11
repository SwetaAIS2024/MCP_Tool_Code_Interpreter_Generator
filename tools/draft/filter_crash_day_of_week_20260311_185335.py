"""Generated MCP tool: filter_crash_day_of_week"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def filter_crash_day_of_week(file_path: str):
    try:
        # Load the dataset
        df = pd.read_csv(file_path)

        # Check if the required column exists
        if "crash_day_of_week" not in df.columns:
            raise ValueError(
                "The required column 'crash_day_of_week' is missing in the dataset."
            )

        # Ensure the 'crash_day_of_week' column is of integer type
        if not pd.api.types.is_integer_dtype(df["crash_day_of_week"]):
            raise TypeError("The 'crash_day_of_week' column must be of integer type.")

        # Filter the dataset to include only weekdays (Monday to Friday)
        filtered_df = df[
            (df["crash_day_of_week"] >= 1) & (df["crash_day_of_week"] <= 5)
        ]

        # Compute the total count of crashes on weekdays
        total_count = len(filtered_df)

        # Return the result in the specified format
        return {"result": {"total_count": total_count}, "metadata": {}}

    except FileNotFoundError:
        return {"result": {}, "metadata": {"error": "File not found"}}

    except pd.errors.EmptyDataError:
        return {"result": {}, "metadata": {"error": "The file is empty"}}

    except pd.errors.ParserError:
        return {"result": {}, "metadata": {"error": "Error parsing the file"}}

    except ValueError as ve:
        return {"result": {}, "metadata": {"error": str(ve)}}

    except TypeError as te:
        return {"result": {}, "metadata": {"error": str(te)}}

    except Exception as e:
        return {
            "result": {},
            "metadata": {"error": "An unexpected error occurred: " + str(e)},
        }
