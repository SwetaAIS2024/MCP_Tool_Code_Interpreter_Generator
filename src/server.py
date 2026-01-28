"""MCP Server entry point for Tool Generator/Interpreter.

This module exposes the tool generation pipeline as an MCP tool using FastMCP.
"""

from typing import Dict, Any
from pathlib import Path
from pydantic import Field
from typing import Annotated

from fastmcp import FastMCP
from .pipeline import run_pipeline

# Initialize FastMCP server
mcp = FastMCP("ToolGeneratorInterpreter")


@mcp.tool()
def analyze_data(
    query: Annotated[str, Field(description="Natural language query describing the analysis to perform")],
    file_path: Annotated[str, Field(description="Path to the CSV or data file to analyze")]
) -> Dict[str, Any]:
    """Analyze data using natural language query.
    
    This tool automatically generates, validates, and executes Python code
    to answer your data analysis questions.
    
    WHEN TO USE THIS TOOL:
    - You have a data file and want to perform analysis using natural language
    - You need custom data transformations or visualizations
    - You want to create reusable analysis tools
    
    WHAT THIS TOOL DOES:
    1. Extracts intent from your query
    2. Generates a tool specification
    3. Generates Python code to perform the analysis
    4. Validates the code (syntax, schema, sandbox execution)
    5. Executes the tool on your data
    6. Returns results for human approval
    7. Promotes approved tools to the registry
    
    RETURNS: Dictionary with:
    - status: "su-ccess" or "failed"
    - tool_created: Name of the generated tool (if successful)
    - result: Execution output from the tool
    - error: Error message (if failed)
    
    EXAMPLE:
        analyze_data(
            query="Calculate the average sales by region",
            file_path="data/sales.csv"
        )
    """
    # Validate file path
    if not Path(file_path).exists():
        return {
            "status": "failed",
            "error": f"File not found: {file_path}"
        }
    
    try:
        # Run the complete pipeline
        result = run_pipeline(query, file_path)
        
        # Check if tool was successfully promoted
        if result.get("promoted_tool"):
            return {
                "status": "success",
                "tool_created": result["promoted_tool"]["name"],
                "tool_path": result["promoted_tool"]["path"],
                "result": result.get("execution_output", "Tool created successfully"),
                "validation_report": result.get("validation_report")
            }
        else:
            # Tool generation failed somewhere in the pipeline
            return {
                "status": "failed",
                "error": "Tool generation failed",
                "messages": result.get("messages", []),
                "validation_report": result.get("validation_report")
            }
            
    except Exception as e:
        return {
            "status": "failed",
            "error": f"Pipeline execution error: {str(e)}"
        }


if __name__ == "__main__":
    # Run MCP server with stdio transport
    mcp.run(transport="stdio")