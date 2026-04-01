from fastmcp import FastMCP
import pandas as pd
import numpy as np

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def pivot_crash_month_weather_condition_count(file_path: str):
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Check for required columns
        required_columns = ["crash_month", "weather_condition"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        # Drop rows with missing values in required columns
        df = df[required_columns].dropna()

        # Group by 'crash_month' and 'weather_condition' and count the number of crashes
        grouped = (
            df.groupby(["crash_month", "weather_condition"])
            .size()
            .unstack(fill_value=0)
        )

        # Convert pivot table to integer type
        pivot_table = grouped.astype(int)

        # Prepare the result and metadata
        result = pivot_table.to_dict()
        metadata = {
            "description": "Pivot table showing crash counts grouped by crash_month and weather_condition",
            "columns": list(pivot_table.columns),
            "index": pivot_table.index.tolist(),
        }

        return {"result": result, "metadata": metadata}

    except Exception as e:
        return {"result": {}, "metadata": {"error": str(e)}}
