from fastmcp import FastMCP
import pandas as pd
import numpy as np
from scipy.stats import f_oneway

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def analyze_intersection_injuries_anova(file_path: str):
    """
    Performs an ANOVA test to compare total injuries between intersection-related and non-intersection-related incidents.

    Parameters:
    file_path (str): Path to CSV file containing the data.

    Returns:
    dict: A dictionary with 'result' and 'metadata' keys.
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
        valid_records = df[df["intersection_related_i"].isin(["Y", "N"])]

        # Group data by 'intersection_related_i' and calculate total injuries for each group
        grouped_data = valid_records.groupby("intersection_related_i")["injuries_total"]
        groups = [
            grouped_data.get_group(group).values
            for group in ["Y", "N"]
            if group in grouped_data.groups
        ]

        # Ensure there are at least two groups with more than one sample
        if len(groups) < 2 or any(len(group) < 2 for group in groups):
            raise ValueError(
                "There must be at least two groups with at least two samples each."
            )

        # Perform ANOVA test
        f_statistic, p_value = f_oneway(*groups)

        # Calculate effect size (eta squared)
        ss_between = sum(
            len(group) * (np.mean(group) - np.mean(np.concatenate(groups))) ** 2
            for group in groups
        )
        ss_total = sum(
            (value - np.mean(np.concatenate(groups))) ** 2
            for value in np.concatenate(groups)
        )
        eta_squared = ss_between / ss_total

        result = {
            "f_statistic": f_statistic,
            "p_value": p_value,
            "eta_squared": eta_squared,
        }

        metadata = {"columns_used": required_columns}

    except Exception as e:
        return {"result": {}, "metadata": {"error": str(e)}}

    return {"result": result, "metadata": metadata}
