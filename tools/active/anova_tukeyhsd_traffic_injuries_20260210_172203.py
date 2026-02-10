"""Generated MCP tool: anova_tukeyhsd_traffic_injuries"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def anova_tukeyhsd_traffic_injuries(file_path: str):
    # Load data from CSV file
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return {"result": {}, "metadata": {"error": f"Error loading file: {str(e)}"}}

    # Validate required columns
    required_columns = ["traffic_control_device", "injuries_total"]
    if not all(column in df.columns for column in required_columns):
        missing_columns = [col for col in required_columns if col not in df.columns]
        return {
            "result": {},
            "metadata": {"error": f"Missing columns: {', '.join(missing_columns)}"},
        }

    # Drop rows with NaN values in required columns
    df = df[required_columns].dropna()

    # Group data by 'traffic_control_device'
    grouped_data = df.groupby("traffic_control_device")["injuries_total"].apply(list)

    # Filter groups with at least 2 samples
    filtered_groups = {
        name: group for name, group in grouped_data.items() if len(group) >= 2
    }

    # Ensure there are at least 2 groups remaining
    if len(filtered_groups) < 2:
        return {
            "result": {},
            "metadata": {"error": "Not enough valid groups to perform ANOVA"},
        }

    # Extract lists of injuries_total for each group
    groups = list(filtered_groups.values())
    group_names = list(filtered_groups.keys())

    # Perform ANOVA
    try:
        f_statistic, p_value = f_oneway(*groups)
    except Exception as e:
        return {
            "result": {},
            "metadata": {"error": f"Error performing ANOVA: {str(e)}"},
        }

    # Compute eta-squared as effect size measure
    ss_between = sum(
        len(group) * (np.mean(group) - np.mean(np.concatenate(groups))) ** 2
        for group in groups
    )
    ss_within = sum((len(group) - 1) * np.var(group, ddof=1) for group in groups)
    eta_squared = ss_between / (ss_between + ss_within)

    # Perform Tukey HSD post-hoc test
    try:
        tukey = pairwise_tukeyhsd(
            endog=np.concatenate(groups),
            groups=np.repeat(group_names, [len(g) for g in groups]),
            alpha=0.05,
        )
    except Exception as e:
        return {
            "result": {},
            "metadata": {"error": f"Error performing Tukey HSD: {str(e)}"},
        }

    # Extract pairwise comparisons from Tukey HSD results
    tukey_table = pd.DataFrame(
        tukey.summary().data[1:], columns=tukey.summary().data[0]
    )
    tukey_pairs = tukey_table[
        ["group1", "group2", "meandiff", "p-adj", "lower", "upper", "reject"]
    ].to_dict("records")

    # Return results and metadata
    return {
        "result": {
            "f_statistic": f_statistic,
            "p_value": p_value,
            "effect_size": eta_squared,
            "tukey_pairs": tukey_pairs,
        },
        "metadata": {},
    }
