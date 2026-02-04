"""Generated MCP tool: custom_transform_anova_traffic_control_injuries"""

from fastmcp import FastMCP
import pandas as pd
import time
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def custom_transform_anova_traffic_control_injuries(file_path: str):
    """
    Performs ANOVA and Tukey HSD post-hoc analysis to compare injuries across traffic control devices.

    Parameters:
    file_path (str): Path to CSV file

    Returns:
    dict: A dictionary with 'result' and 'metadata' keys containing the statistical results.
    """
    try:
        # Load data
        df = pd.read_csv(file_path)

        # Validate required columns
        required_columns = ["traffic_control_device", "injuries_total"]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"CSV file must contain the following columns: {required_columns}"
            )

        # Prepare groups
        grouped_data = df.groupby("traffic_control_device")["injuries_total"].apply(
            list
        )

        # Run ANOVA
        f_statistic, p_value_anova = f_oneway(*grouped_data)

        # Post-hoc test using Tukey HSD
        tukey_results = pairwise_tukeyhsd(
            endog=df["injuries_total"], groups=df["traffic_control_device"]
        )

        # Calculate effect sizes - Eta-squared
        ss_between_groups = sum(
            (
                df.groupby("traffic_control_device")["injuries_total"].mean()
                - df["injuries_total"].mean()
            )
            ** 2
            * df.groupby("traffic_control_device").size()
        )
        ss_within_groups = sum(
            df.groupby("traffic_control_device")["injuries_total"].apply(
                lambda x: ((x - x.mean()) ** 2).sum()
            )
        )
        eta_squared = ss_between_groups / (ss_between_groups + ss_within_groups)

        # Format results
        result = {
            "f_statistic": f_statistic,
            "p_value_anova": p_value_anova,
            "tukey_results": tukey_results.summary().as_text(),
            "eta_squared": eta_squared,
        }

        metadata = {"file_path": file_path, "columns_used": required_columns}

        return {"result": result, "metadata": metadata}

    except Exception as e:
        return {"result": {}, "metadata": {"error": str(e)}}
