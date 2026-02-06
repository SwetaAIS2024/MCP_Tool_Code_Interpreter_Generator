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
        if (
            "traffic_control_device" not in df.columns
            or "injuries_total" not in df.columns
        ):
            raise ValueError(
                "CSV must contain 'traffic_control_device' and 'injuries_total' columns."
            )

        # Ensure data types are correct
        if not pd.api.types.is_numeric_dtype(df["injuries_total"]):
            raise TypeError("'injuries_total' column must be numeric.")

        # Prepare groups for ANOVA
        grouped = df.groupby("traffic_control_device")["injuries_total"]
        groups = [grouped.get_group(name) for name in grouped.groups]

        # Run ANOVA
        f_statistic, p_value = f_oneway(*groups)

        # Calculate effect size (eta-squared)
        ss_between = sum(
            [
                len(group) * (group.mean() - df["injuries_total"].mean()) ** 2
                for group in groups
            ]
        )
        ss_within = sum([(len(group) - 1) * group.var(ddof=0) for group in groups])
        eta_squared = ss_between / (ss_between + ss_within)

        # Post-hoc test using Tukey HSD
        tukey_results = pairwise_tukeyhsd(
            endog=df["injuries_total"], groups=df["traffic_control_device"]
        )

        # Format results
        result = {
            "f_statistic": f_statistic,
            "p_value": p_value,
            "eta_squared": eta_squared,
            "tukey_hsd_summary": tukey_results.summary().as_text(),
        }

        metadata = {
            "file_path": file_path,
            "columns_used": ["traffic_control_device", "injuries_total"],
        }

        return {"result": result, "metadata": metadata}

    except Exception as e:
        return {"result": {}, "metadata": {"error": str(e)}}
