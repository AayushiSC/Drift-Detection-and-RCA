import json

from google.adk import Agent
from google.adk.tools import google_search

internet_agent = Agent(
    name="internet",
    description="An agent that searches the internet for possible reasons for data drift.",
    model="gemini-2.0-flash",
    tools=[google_search],
    instruction="""You are an internet research agent specialized in finding reasons for data drift.

Your task:
1. Read the 'meta_info' key from the state to understand the drift characteristics
2. Use the google_search tool to search for possible reasons and causes of the observed drift
3. Analyze the search results to identify relevant explanations and possible outcomes
4. Store your findings (both reasons and possible outcomes) in the 'internet_info' key in the state

Format your output as a comprehensive summary including:
- Potential drift causes found from internet sources
- Possible outcomes and implications of the drift""",
)
