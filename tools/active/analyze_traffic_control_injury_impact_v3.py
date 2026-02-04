from fastmcp import FastMCP
import pandas as pd
import time
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def analyze_traffic_control_injury_impact(file_path: str):
    """
    Performs an ANOVA test followed by Tukey HSD post-hoc analysis to compare total injuries across different traffic control devices.

    Parameters:
    file_path (str): Path to CSV file containing 'traffic_control_device' and 'injuries_total'.

    Returns:
    dict: A dictionary with 'result' and 'metadata' keys.
    """

    try:
        # Load data from CSV file
        df = pd.read_csv(file_path)

        # Validate required columns exist
        required_columns = ["traffic_control_device", "injuries_total"]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"CSV file must contain the following columns: {required_columns}"
            )

        # Ensure 'injuries_total' is numeric
        if not pd.api.types.is_numeric_dtype(df["injuries_total"]):
            raise TypeError("The 'injuries_total' column must be numeric.")

        # Group data by 'traffic_control_device'
        grouped = df.groupby("traffic_control_device")["injuries_total"]

        # Extract groups as lists for ANOVA
        groups = [grouped.get_group(name).values for name in grouped.groups]

        # Run ANOVA to test for differences in injuries across groups
        f_statistic, p_value = f_oneway(*groups)

        # Perform post-hoc Tukey HSD testing with multiple-comparisons correction
        tukey_results = pairwise_tukeyhsd(
            endog=df["injuries_total"], groups=df["traffic_control_device"]
        )

        # Calculate eta-squared effect sizes for each comparison
        ss_between = sum(
            [
                len(group) * (group.mean() - df["injuries_total"].mean()) ** 2
                for name, group in grouped
            ]
        )
        ss_within = sum(
            [(x - group.mean()) ** 2 for name, group in grouped for x in group]
        )
        eta_squared = ss_between / (ss_between + ss_within)

        # Format results
        result = {
            "f_statistic": f_statistic,
            "p_value": p_value,
            "adjusted_p_values": tukey_results.pvalues.tolist(),
            "effect_sizes": [eta_squared] * len(tukey_results.groupsunique),
            "group_means": grouped.mean().to_dict(),
        }

        # Format metadata
        metadata = {"input_file": file_path}

        return {"result": result, "metadata": metadata}

    except Exception as e:
        return {"result": {}, "metadata": {"error": str(e)}}
