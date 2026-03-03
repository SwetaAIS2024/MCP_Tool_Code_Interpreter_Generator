"""Generated MCP tool: analyze_intersection_injuries_anova"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np
from scipy.stats import f_oneway

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def analyze_intersection_injuries_anova(file_path: str):
    """
    Performs an ANOVA test to compare total injuries between intersection-related and non-intersection-related incidents.

    Parameters:
    - file_path (str): Path to CSV file containing the data.

    Returns:
    - dict: A dictionary with 'result' and 'metadata' keys.
    """
    try:
        # Load the dataset
        df = pd.read_csv(file_path)

        # Check for required columns
        required_columns = ["intersection_related_i", "injuries_total"]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"CSV file must contain the following columns: {required_columns}"
            )

        # Drop rows with NaN values in required columns
        df = df[required_columns].dropna()

        # Filter records where 'intersection_related_i' is either 'Y' or not 'Y'
        filtered_df = df[df["intersection_related_i"].isin(["Y", "N"])]

        # Group data by 'intersection_related_i' and calculate total injuries for each group
        grouped_data = filtered_df.groupby("intersection_related_i")["injuries_total"]
        groups = [
            grouped_data.get_group(name)
            for name in grouped_data.groups
            if len(grouped_data.get_group(name)) >= 2
        ]

        # Check if there are at least two groups with sufficient samples
        if len(groups) < 2:
            raise ValueError("Not enough valid groups to perform ANOVA test.")

        # Perform ANOVA test
        f_statistic, p_value = f_oneway(*groups)

        # Calculate effect size (eta squared)
        ss_between = sum(
            len(group) * (group.mean() - filtered_df["injuries_total"].mean()) ** 2
            for group in groups
        )
        ss_within = sum((len(group) - 1) * group.var(ddof=0) for group in groups)
        eta_squared = ss_between / (ss_between + ss_within)

        # Prepare result dictionary
        result = {
            "f_statistic": f_statistic,
            "p_value": p_value,
            "effect_size": eta_squared,
        }

        # Prepare metadata dictionary
        metadata = {"groups_count": len(groups), "total_records": len(filtered_df)}

    except Exception as e:
        return {"result": {}, "metadata": {"error": str(e)}}

    return {"result": result, "metadata": metadata}
