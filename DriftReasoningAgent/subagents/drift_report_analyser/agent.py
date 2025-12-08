import json
import os
from pathlib import Path
from typing import Any, Dict, List

from google.adk.agents import Agent

# --- Constants ---
GEMINI_MODEL = "gemini-2.0-flash"


def read_drift_report() -> str:
    """Read drift_structured_log.json file from the data folder and extract drift information.

    This tool reads the drift detection report and identifies which columns have drift
    and which statistical tests detected that drift.

    Returns:
        str: Drift information including columns with drift and tests that detected it,
             or an error message if not found.
    """
    # Get the project root directory (3 levels up from this file)
    # Path: DriftReasoningAgent/subagents/drift_report_analyser/agent.py -> go up 3 levels
    current_dir = Path(__file__).parent  # .../drift_report_analyser
    project_root = current_dir.parent.parent.parent  # Go up 3 levels to root
    drift_report_path = project_root / "data" / "drift_structured_log.json"

    # Debug info
    debug_info = f"Current file: {__file__}\n"
    debug_info += f"Project root: {project_root}\n"
    debug_info += f"Drift report path: {drift_report_path}\n"
    debug_info += f"File exists: {drift_report_path.exists()}\n"

    try:
        if not drift_report_path.exists():
            return f"Drift report file not found.\n\n{debug_info}"

        with open(drift_report_path, "r", encoding="utf-8") as f:
            drift_report = json.load(f)

        # Extract tests information
        tests = drift_report.get("tests", [])

        # Group tests by column
        drift_by_column = {}

        for test in tests:
            column = test.get("column")
            test_name = test.get("test")
            drift_detected = test.get("drift_detected") == "True"

            if drift_detected:
                if column not in drift_by_column:
                    drift_by_column[column] = {
                        "column": column,
                        "drift_detected": True,
                        "tests_detecting_drift": [],
                        "test_details": [],
                    }

                drift_by_column[column]["tests_detecting_drift"].append(
                    test_name
                )
                drift_by_column[column]["test_details"].append(
                    {
                        "test": test_name,
                        "threshold": test.get("threshold"),
                        "weight": test.get("weight"),
                        "result": test.get("result", {}),
                    }
                )

        # Format the output
        if not drift_by_column:
            return "Successfully read drift report. No drift detected in any columns."

        output = (
            f"Successfully read drift report from: {drift_report_path}\n\n"
        )
        output += f"Found drift in {len(drift_by_column)} column(s):\n\n"

        for idx, (column, drift_info) in enumerate(drift_by_column.items(), 1):
            output += f"{idx}. Column: {column}\n"
            output += f"   Tests detecting drift: {', '.join(drift_info['tests_detecting_drift'])}\n"
            output += f"   Test details:\n"

            for test_detail in drift_info["test_details"]:
                output += f"     - {test_detail['test'].upper()}: threshold={test_detail['threshold']}, weight={test_detail['weight']}\n"
            output += "\n"

        return output

    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON in drift report.\n\n{debug_info}\nError details: {str(e)}"
    except Exception as e:
        return f"Error reading drift report: {str(e)}\n\n{debug_info}"


# Create the drift report analyser agent
drift_report_agent = Agent(
    name="drift_report_analyser",
    model=GEMINI_MODEL,
    instruction="""You are a Drift Report Analysis Agent.

    Start by saying if you found teh report or not 
    
    
    Your task is to read and analyze the drift_structured_log.json file to identify:
    1. Which columns have drift detected
    2. Which statistical tests detected the drift for each column
    3. Test parameters (thresholds and weights)
    
    Automatically call the read_drift_report() tool to access the drift information.
    If you find the file, analyze it and provide a clear summary of:
    - Total number of columns with detected drift
    - For each drifted column:
      * Column name
      * Statistical tests that detected drift (ks, wasserstein, psi, etc.)
      * Test thresholds and weights
    
    If the file is not found, inform the user clearly.
    
    This information will be used by the metadata agent to understand which columns
    need further investigation.
    """,
    description="Analyzes drift report to identify columns with drift and the tests that detected it.",
    output_key="drift_info",
    tools=[read_drift_report],
)
