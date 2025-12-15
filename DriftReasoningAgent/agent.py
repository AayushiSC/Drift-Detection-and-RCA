# this is a sequential agent that will call the other agents in the subagents
# metadata agent
# eda agent
# internet agent
# drift test agent

from google.adk.agents import SequentialAgent

from .subagents.drift_report_analyser import drift_report_agent
from .subagents.eda import eda_agent
from .subagents.internet import internet_agent
from .subagents.metadata import metadata_agent
from .subagents.summariser import summariser_agent

# Create the sequential agent with minimal callback

root_agent = SequentialAgent(
    name="DriftReasoningPipeline",
    sub_agents=[
        drift_report_agent,
        metadata_agent,
        eda_agent,
        internet_agent,
        summariser_agent,
    ],
    description="A pipeline that identifies the reason for data drift using metadata and internet sources",
)
