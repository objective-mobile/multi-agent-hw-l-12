"""
ReportMCP server — port 8902
Tools: save_report
Resources: resource://output-dir
"""
import os
import sys

from fastmcp import FastMCP

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import settings

mcp = FastMCP("ReportMCP")


@mcp.tool()
def save_report(filename: str, content: str) -> str:
    """Save a Markdown research report to the output directory."""
    try:
        os.makedirs(settings.output_dir, exist_ok=True)
        if not filename.endswith(".md"):
            filename += ".md"
        path = os.path.join(settings.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Report saved to {path}"
    except Exception as e:
        return f"Error saving report: {str(e)}"


@mcp.resource("resource://output-dir")
def output_dir_info() -> str:
    """Returns the output directory path and list of saved reports."""
    out = settings.output_dir
    if not os.path.exists(out):
        return f"Output directory: {out}\nNo reports saved yet."
    reports = [f for f in os.listdir(out) if f.endswith(".md")]
    reports_list = "\n".join(f"  - {r}" for r in sorted(reports)) or "  (none)"
    return f"Output directory: {os.path.abspath(out)}\nSaved reports:\n{reports_list}"


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8902)
