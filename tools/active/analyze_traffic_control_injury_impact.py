from fastmcp import FastMCP
import pandas as pd
import time
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def analyze_traffic_control_injury_impact(file_path: str):
    """
    Performs an ANOVA test followed by Tukey HSD post-hoc analysis to compare total injuries across different traffic control devices.

    Parameters:
    - file_path (str): Path to CSV file containing 'traffic_control_device' and 'injuries_total'

    Returns:
    - dict: A dictionary with 'result' and 'metadata' keys
    """
    try:
        # Load data from CSV file
        df = pd.read_csv(file_path)

        # Validate required columns exist
        required_columns = ["traffic_control_device", "injuries_total"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Group data by 'traffic_control_device'
        grouped_data = df.groupby("traffic_control_device")["injuries_total"]

        # Extract groups as lists for ANOVA
        groups = [grouped_data.get_group(name).values for name in grouped_data.groups]

        # Run ANOVA to test for differences in injuries across groups
        f_statistic, p_value = f_oneway(*groups)

        # Perform post-hoc Tukey HSD testing with multiple-comparisons correction
        tukey_results = pairwise_tukeyhsd(
            endog=df["injuries_total"], groups=df["traffic_control_device"]
        )

        # Calculate eta-squared effect sizes for each comparison
        ss_between_groups = np.sum(
            [
                len(group) * (np.mean(group) - df["injuries_total"].mean()) ** 2
                for group in groups
            ]
        )
        ss_within_groups = np.sum(
            [(x - np.mean(group)) ** 2 for name, group in grouped_data for x in group]
        )
        eta_squared = ss_between_groups / (ss_between_groups + ss_within_groups)

        # Calculate mean injuries for each traffic control device group
        group_means = grouped_data.mean().to_dict()

        # Format results
        result = {
            "f_statistic": f_statistic,
            "p_value": p_value,
            "adjusted_p_values": tukey_results.pvalues.tolist(),
            "effect_sizes": eta_squared,
            "group_means": group_means,
        }

        metadata = {"input_file": file_path}

        return {"result": result, "metadata": metadata}

    except Exception as e:
        return {"result": {}, "metadata": {"error": str(e)}}
