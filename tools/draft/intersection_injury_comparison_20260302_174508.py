from fastmcp import FastMCP
import pandas as pd
import numpy as np
from scipy.stats import shapiro, ttest_ind, mannwhitneyu
from statsmodels.stats.power import CohenEffectSize

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def intersection_injury_comparison(file_path: str):
    try:
        # Load data and select required columns
        df = pd.read_csv(file_path)
        required_columns = ["intersection_related_i", "injuries_total"]

        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"CSV file must contain the following columns: {required_columns}"
            )

        # Split data into two groups
        group_y = df[df["intersection_related_i"] == "Y"]["injuries_total"]
        group_n = df[df["intersection_related_i"] != "Y"]["injuries_total"]

        # Check normality of both groups
        _, p_value_y = shapiro(group_y)
        _, p_value_n = shapiro(group_n)

        # Perform t-test if normally distributed, otherwise Mann-Whitney U test
        if p_value_y > 0.05 and p_value_n > 0.05:
            stat, p_value = ttest_ind(group_y, group_n)
            effect_size = CohenEffectSize().cohen_d(group_y, group_n)
        else:
            stat, p_value = mannwhitneyu(group_y, group_n)
            # Effect size for Mann-Whitney U can be calculated as r
            effect_size = stat / np.sqrt(len(group_y) * len(group_n))

        significant = p_value < 0.05

        return {
            "result": {
                "statistic": stat,
                "p_value": p_value,
                "significant": significant,
                "effect_size": effect_size,
            },
            "metadata": {"columns_used": required_columns},
        }

    except Exception as e:
        return {"result": {}, "metadata": {"error": str(e)}}
