"""Generated MCP tool: crash_hour_injuries_custom_analysis"""

from fastmcp import FastMCP
import pandas as pd
import time
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

mcp = FastMCP("data_analysis_tools")


@mcp.tool()
def crash_hour_injuries_custom_analysis(file_path: str):
    try:
        # Load the data and select required columns
        df = pd.read_csv(file_path)
        required_columns = ["crash_hour", "injuries_total"]

        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"CSV file must contain the following columns: {required_columns}"
            )

        df = df[required_columns].dropna()

        # Group by crash_hour and calculate mean injuries_total
        grouped_data = df.groupby("crash_hour")["injuries_total"].mean().reset_index()

        if len(grouped_data) < 2:
            raise ValueError(
                "Insufficient data: Less than 2 unique crash hours available."
            )

        # Prepare data for regression analysis
        X = sm.add_constant(grouped_data["crash_hour"])
        y = grouped_data["injuries_total"]

        # Perform OLS regression
        model = ols("injuries_total ~ crash_hour", data=grouped_data).fit()

        # Check assumptions (normality, homoscedasticity)
        residuals = model.resid

        # Normality test using Shapiro-Wilk
        normality_test = stats.shapiro(residuals)
        is_normal = normality_test.pvalue > 0.05

        # Homoscedasticity test using Breusch-Pagan
        bp_test = stats.breuschpagan(residuals, X)
        homoscedasticity_p_value = bp_test[1]
        is_homoscedastic = homoscedasticity_p_value > 0.05

        # Compute R-squared value
        r_squared = model.rsquared

        # Summarize results
        summary_results = {
            "coefficients": model.params.to_dict(),
            "p_values": model.pvalues.to_dict(),
            "confidence_intervals": model.conf_int().to_dict(orient="index"),
        }

        result = {
            "r_squared": r_squared,
            "summary": summary_results,
            "normality_test": {
                "statistic": normality_test.statistic,
                "p_value": normality_test.pvalue,
                "is_normal": is_normal,
            },
            "homoscedasticity_test": {
                "statistic": bp_test[0],
                "p_value": homoscedasticity_p_value,
                "is_homoscedastic": is_homoscedastic,
            },
        }

        metadata = {
            "data_points": len(grouped_data),
            "unique_crash_hours": grouped_data["crash_hour"].nunique(),
        }

        return {"result": result, "metadata": metadata}

    except Exception as e:
        return {"error": str(e)}
