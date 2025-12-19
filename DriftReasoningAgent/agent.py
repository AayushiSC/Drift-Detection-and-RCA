# this is a sequential agent that will call the other agents in the subagents
# metadata agent
# eda agent with reflection loop (3 iterations)
# internet agent
# drift test agent

from google.adk.agents import SequentialAgent

from .subagents.drift_report_analyser import drift_report_agent
from .subagents.eda import create_eda_agent
from .subagents.internet import internet_agent
from .subagents.metadata import metadata_agent
from .subagents.reflection import create_reflection_agent
from .subagents.summariser import summariser_agent

# Create the sequential agent pipeline with EDA-Reflection loop (3 iterations)
root_agent = SequentialAgent(
    name="DriftReasoningPipeline",
    sub_agents=[
        drift_report_agent,
        metadata_agent,
        # Iteration 1
        create_eda_agent(iteration=1),
        create_reflection_agent(iteration=1),
        # Iteration 2
        create_eda_agent(iteration=2),
        create_reflection_agent(iteration=2),
        # Iteration 3 (final)
        create_eda_agent(iteration=3),
        # Continue with rest of pipeline
        internet_agent,
        summariser_agent,
    ],
    description="A pipeline that identifies the reason for data drift using metadata, iterative EDA-Reflection loop (3 iterations), and internet sources",
)
