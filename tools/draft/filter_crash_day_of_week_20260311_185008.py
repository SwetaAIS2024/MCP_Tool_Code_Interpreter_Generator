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

        # Check if the required column is present
        if "crash_day_of_week" not in df.columns:
            raise ValueError(
                "The required column 'crash_day_of_week' is missing in the dataset."
            )

        # Ensure the 'crash_day_of_week' column is of integer type
        if not pd.api.types.is_integer_dtype(df["crash_day_of_week"]):
            raise TypeError("The 'crash_day_of_week' column must be of integer type.")

        # Filter the data to include only rows where 'crash_day_of_week' is 1 (Sunday) or 7 (Saturday)
        filtered_df = df[df["crash_day_of_week"].isin([1, 7])]

        # Compute the total count of such crashes
        total_count = len(filtered_df)

        # Prepare the result and metadata
        result = {"total_count": total_count}
        metadata = {"filter": "crash_day_of_week is 1 (Sunday) or 7 (Saturday)"}

        return {"result": result, "metadata": metadata}

    except FileNotFoundError:
        return {"result": {}, "metadata": {"error": "File not found"}}
    except pd.errors.EmptyDataError:
        return {"result": {}, "metadata": {"error": "The file is empty"}}
    except pd.errors.ParserError:
        return {"result": {}, "metadata": {"error": "Error parsing the file"}}
    except Exception as e:
        return {"result": {}, "metadata": {"error": str(e)}}
