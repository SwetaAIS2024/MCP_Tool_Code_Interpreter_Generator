from fastmcp import FastMCP
import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def anova_tukeyhsd_traffic_injuries(file_path: str):
    try:
        # Load data
        df = pd.read_csv(file_path)

        # Validate required columns
        required_columns = ["traffic_control_device", "injuries_total"]
        if not all(column in df.columns for column in required_columns):
            return {
                "result": {},
                "metadata": {"error": f"Missing required columns: {required_columns}"},
            }

        # Group data by 'traffic_control_device'
        grouped = df.groupby("traffic_control_device")["injuries_total"]

        # Check if there are at least two groups
        if len(grouped) < 2:
            return {
                "result": {},
                "metadata": {
                    "error": "At least two distinct groups are required for ANOVA."
                },
            }

        # Extract each group as a list
        groups = [group.tolist() for name, group in grouped]

        # Run ANOVA
        f_statistic, anova_pvalue = f_oneway(*groups)

        # Perform Tukey HSD post-hoc test
        tukey_results = pairwise_tukeyhsd(
            endog=df["injuries_total"], groups=df["traffic_control_device"]
        )

        # Compute eta-squared as effect size measure
        ss_between = sum(
            (group.mean() - df["injuries_total"].mean()) ** 2 * len(group)
            for name, group in grouped
        )
        ss_within = sum(
            (value - group.mean()) ** 2 for name, group in grouped for value in group
        )
        eta_squared = ss_between / (ss_between + ss_within)

        # Prepare results
        tukey_summary = tukey_results.summary().as_text()
        adjusted_p_values = dict(zip(tukey_results.groupsunique, tukey_results.pvalues))

        return {
            "result": {
                "F_statistic": f_statistic,
                "ANOVA_p_value": anova_pvalue,
                "adjusted_p_values": adjusted_p_values,
                "eta_squared": eta_squared,
                "group_means": grouped.mean().to_dict(),
            },
            "metadata": {},
        }

    except Exception as e:
        return {"result": {}, "metadata": {"error": str(e)}}
