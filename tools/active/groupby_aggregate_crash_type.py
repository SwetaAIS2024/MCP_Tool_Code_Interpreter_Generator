"""Generated MCP tool: groupby_aggregate_crash_type"""

from fastmcp import FastMCP
import pandas as pd
from typing import Dict, Any
import time

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def groupby_aggregate_crash_type(file_path: str) -> Dict[str, Any]:
    """
    Reads a CSV file containing traffic accident data, groups the data by 'crash_type',
    counts occurrences of each crash type, sorts the results in descending order based on these
    counts, and returns the top 5 most frequent crash types.

    Parameters:
    - file_path (str): The path to the CSV file containing traffic accident data.

    Returns:
    - Dict[str, Any]: A dictionary with 'result' and 'metadata' keys. The 'result' key contains
      a DataFrame with the top 5 most frequent crash types and their counts. The 'metadata'
      key provides additional information about the operation.
    """
    try:
        # Load data from the specified CSV file
        df = pd.read_csv(file_path)

        # Validate that the required 'crash_type' column exists in the dataset
        if "crash_type" not in df.columns:
            raise ValueError(
                "The required 'crash_type' column is missing in the dataset."
            )

        # Group the data by unique values of 'crash_type'
        grouped_data = df.groupby("crash_type")

        # Aggregate and count occurrences of each crash type within these groups
        crash_counts = grouped_data.size().reset_index(name="count")

        # Sort the aggregated counts in descending order based on count values
        sorted_crash_counts = crash_counts.sort_values(by="count", ascending=False)

        # Limit the results to the top 5 rows from the sorted data
        top_5_crashes = sorted_crash_counts.head(5)

        # Format the final output as a table with 'crash_type' and its corresponding counts
        result = {
            "result": top_5_crashes.to_dict(orient="records"),
            "metadata": {
                "status": "success",
                "message": "Top 5 most frequent crash types retrieved successfully.",
            },
        }

    except FileNotFoundError:
        result = {
            "result": [],
            "metadata": {
                "status": "error",
                "message": f"The file at {file_path} was not found.",
            },
        }

    except pd.errors.EmptyDataError:
        result = {
            "result": [],
            "metadata": {"status": "error", "message": "The CSV file is empty."},
        }

    except ValueError as ve:
        result = {"result": [], "metadata": {"status": "error", "message": str(ve)}}

    except Exception as e:
        result = {
            "result": [],
            "metadata": {
                "status": "error",
                "message": f"An unexpected error occurred: {str(e)}",
            },
        }

    return result
