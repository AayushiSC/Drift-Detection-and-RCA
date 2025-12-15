import os
from pathlib import Path

from google.adk.agents import Agent

# --- Constants ---
GEMINI_MODEL = "gemini-2.0-flash"


def read_metadata_file() -> str:
    """
    Read metadata file from the data folder.

    Returns:
        str: The content of the metadata file, or an error message if not found.
    """
    # Get the project root directory (3 levels up from this file)
    # Path: DriftReasoningAgent/subagents/metadata/agent.py -> go up 3 levels
    current_dir = Path(__file__).parent  # .../metadata
    project_root = current_dir.parent.parent.parent  # Go up 3 levels to root
    data_folder = project_root / "data"

    # Debug info
    debug_info = f"Current file: {__file__}\n"
    debug_info += f"Project root: {project_root}\n"
    debug_info += f"Data folder: {data_folder}\n"
    debug_info += f"Data folder exists: {data_folder.exists()}\n"

    # Look for metadata file
    if data_folder.exists():
        metadata_files = list(data_folder.glob("metadata.*"))
        debug_info += f"Files found: {metadata_files}\n"

        if metadata_files:
            with open(metadata_files[0], "r", encoding="utf-8") as f:
                content = f.read()
            return f"Successfully read metadata file: {metadata_files[0]}\n\n{content}"

    return f"No metadata file found.\n\n{debug_info}"


# Create the metadata agent
metadata_agent = Agent(
    name="metadata",
    model=GEMINI_MODEL,
    instruction="""You are a Metadata Information Agent.
    
    Your task is to read and process metadata information from the provided file.
    Use the read_metadata_file() tool to access the metadata content.
    automatically call the tool to get the metadata information, dont wait for a prompt , if you find the file then while analysing it let the user know that you are analysing it else if you dont find it , then let the user know that too.
    Extract and summarize key metadata attributes that are relevant for drift detection analysis.
    Understand what each of the columns in the dataset represent based on the metadata file.
    
    Provide a structured summary of the metadata information including:
    - Key attributes and their descriptions
    - Any temporal or categorical information
    - Relevant constraints or business rules
    - Relevant information provided in the metadata file

    Store the summarized metadata information in the output key "meta_info".
    Try to be as detailed as possible while summarizing the metadata information.
    
    """,
    description="Reads and processes metadata information for drift detection.",
    output_key="meta_info",
    tools=[read_metadata_file],
)
