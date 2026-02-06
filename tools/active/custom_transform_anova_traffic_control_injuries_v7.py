"""Generated MCP tool: custom_transform_anova_traffic_control_injuries"""

from fastmcp import FastMCP
import pandas as pd
import time
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def custom_transform_anova_traffic_control_injuries(file_path: str):
    try:
        # Load data from CSV file
        df = pd.read_csv(file_path)

        # Validate required columns
        required_columns = {"traffic_control_device", "injuries_total"}
        if not required_columns.issubset(df.columns):
            raise ValueError(
                f"CSV must contain the following columns: {required_columns}"
            )

        # Ensure data types are correct
        if not pd.api.types.is_string_dtype(df["traffic_control_device"]):
            raise TypeError("Column 'traffic_control_device' must be of string type.")
        if not pd.api.types.is_numeric_dtype(df["injuries_total"]):
            raise TypeError("Column 'injuries_total' must be numeric.")

        # Prepare groups for ANOVA
        grouped = df.groupby("traffic_control_device")["injuries_total"]
        groups = [grouped.get_group(name) for name in grouped.groups]

        # Run ANOVA
        f_statistic, p_value = f_oneway(*groups)

        # Post-hoc test using Tukey HSD
        tukey_results = pairwise_tukeyhsd(
            endog=df["injuries_total"], groups=df["traffic_control_device"]
        )

        # Calculate effect size (eta-squared)
        ss_between = sum(
            [
                len(group) * (group.mean() - df["injuries_total"].mean()) ** 2
                for group in groups
            ]
        )
        ss_within = sum(
            [(x - group.mean()) ** 2 for name, group in grouped for x in group]
        )
        eta_squared = ss_between / (ss_between + ss_within)

        # Format results
        result = {
            "F_statistic": f_statistic,
            "p_value": p_value,
            "tukey_hsd_results": tukey_results.summary().as_text(),
            "eta_squared": eta_squared,
        }

        metadata = {
            "file_path": file_path,
            "columns_used": ["traffic_control_device", "injuries_total"],
        }

        return {"result": result, "metadata": metadata}

    except Exception as e:
        return {"result": {}, "metadata": {"error": str(e)}}
