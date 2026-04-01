"""Generated MCP tool: injury_summary_statistics"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np
import scipy.stats as stats

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def injury_summary_statistics(file_path: str):
    try:
        # Load the dataset
        df = pd.read_csv(file_path)

        # Check for required columns
        required_columns = [
            "injuries_total",
            "injuries_fatal",
            "injuries_incapacitating",
            "injuries_non_incapacitating",
            "injuries_reported_not_evident",
            "injuries_no_indication",
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Drop rows with NaN values in required columns
        df = df[required_columns].dropna()

        # Compute descriptive statistics
        statistics = df.describe(percentiles=[0.25, 0.5, 0.75]).T

        # Convert statistics to dictionary
        statistics_dict = statistics.to_dict()

        # Prepare result and metadata
        result = {"summary_statistics": statistics_dict}
        metadata = {"file_path": file_path, "columns_used": required_columns}

        return {"result": result, "metadata": metadata}

    except FileNotFoundError:
        return {"result": {}, "metadata": {"error": "File not found"}}
    except pd.errors.EmptyDataError:
        return {"result": {}, "metadata": {"error": "File is empty"}}
    except pd.errors.ParserError:
        return {"result": {}, "metadata": {"error": "File parsing error"}}
    except ValueError as ve:
        return {"result": {}, "metadata": {"error": str(ve)}}
    except Exception as e:
        return {
            "result": {},
            "metadata": {"error": f"An unexpected error occurred: {str(e)}"},
        }
