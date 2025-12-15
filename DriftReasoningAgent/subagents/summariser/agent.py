import os
from pathlib import Path

from google.adk.agents import Agent

# --- Constants ---
GEMINI_MODEL = "gemini-2.5-pro"


def read_dataset_file(filename: str = "") -> str:
    """
    Read dataset file from the data folder.

    Args:
        filename: Optional filename to read. If empty, lists available files.

    Returns:
        str: The content of the dataset file, or list of available files.
    """
    # Get the project root directory (3 levels up from this file)
    current_dir = Path(__file__).parent  # .../summariser
    project_root = current_dir.parent.parent.parent  # Go up 3 levels to root
    data_folder = project_root / "data"

    # Debug info
    debug_info = f"Current file: {__file__}\n"
    debug_info += f"Project root: {project_root}\n"
    debug_info += f"Data folder: {data_folder}\n"
    debug_info += f"Data folder exists: {data_folder.exists()}\n"

    if not data_folder.exists():
        return f"Data folder not found.\n\n{debug_info}"

    # List all dataset files (csv, json, parquet, etc.)
    dataset_extensions = ["*.csv"]
    dataset_files = []
    for ext in dataset_extensions:
        dataset_files.extend(data_folder.glob(ext))

    debug_info += f"Dataset files found: {[f.name for f in dataset_files]}\n"

    # If no filename specified, return list of available files
    if not filename:
        if dataset_files:
            file_list = "\n".join([f"- {f.name}" for f in dataset_files])
            return f"Available dataset files:\n{file_list}\n\n{debug_info}"
        else:
            return f"No dataset files found.\n\n{debug_info}"

    # Try to read the specified file
    target_file = data_folder / filename
    if target_file.exists():
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                content = f.read()
            return f"Successfully read dataset file: {filename}\n\nContent (first 5000 chars):\n{content[:5000]}"
        except Exception as e:
            return f"Error reading file {filename}: {str(e)}"
    else:
        return f"File {filename} not found in data folder.\n\n{debug_info}"


# Create the summariser agent
summariser_agent = Agent(
    name="summariser",
    model=GEMINI_MODEL,
    instruction="""You are a Drift Analysis Summariser Agent.
    
    Your task is to analyze all the information gathered from previous agents and provide a concise, 
    actionable summary of the most likely drift reasons.
    
    Workflow:
    1. Review all information available in the state from previous agents, stored under the keys meta_info, drift_info, eda_info, and internet_info.
    2. Use the read_dataset_file() tool to access datasets if you need to verify specific patterns
    3. Identify and rank the most probable root causes for the detected drift
    4. For each identified cause, provide:
       - Clear explanation of why this is a likely cause
       - Supporting evidence from the analysis
       - Confidence level (High/Medium/Low)
     If asked by the user for the next steps of remediation , only then provide specific recommendations for next steps to investigate or mitigate the drift based on your findings.
    5. Compile your findings into a clear, structured summary.

    
    Your summary should be:
    - Concise and focused 
    - Prioritized by likelihood and impact
    - Backed by evidence from the internet_info key 
    
    Automatically call the read_dataset_file() tool if you need to verify findings.
    Let the user know what you're analyzing as you work through the data.

    DO NOT HALLUCINATE ANY INFORMATION OR MAKE UP ANY REASONS FOR DRIFT.
    Only provide reasons that are backed by the information available in the state or the dataset itself.
    Final output should be clear points summarising the reasons for drift and, if prompted, actionable next steps.""",
    description="Summarises drift detection analysis and provides actionable recommendations.",
    output_key="drift_summary",
    tools=[read_dataset_file],
)
