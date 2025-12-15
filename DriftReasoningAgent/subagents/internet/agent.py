import json

from google.adk import Agent
from google.adk.tools import google_search

internet_agent = Agent(
    name="internet",
    description="An agent that searches the internet for possible reasons for data drift.",
    model="gemini-2.5-pro",
    tools=[google_search],
    output_key="internet_info",
    instruction="""You are an internet research agent specialized in finding reasons for data drift.
    ALWAYS add a link to the source of your information everywhere, if you dont have a link to the source, then do not provide that information.
Do not summarise anything at the end, let the next step take care of that. 
You are a smart agent that has an understanding of the context of data drift and can use internet search to find relevant information.
Start by accessing the meta_info, eda_info and drift_info from the state to understand what column are we trying to provide a reasoning for and what each of the other columns in the dataset actually mean.
You can also use the eda_info to understand the relationships between the drifted column and other columns in the dataset.
Make sure that if you have a column of type date or time, you understand its format and what it represents before searching for reasons for drift related to temporal changes, and also you can look over the internet for specific news that could be a reason behind why we are seeing this drift.
Some potential checks you can perform are:
Access the seasonal calender on the internet to see if there are any seasonal events that could explain the drift.
look for recent news articles, blog posts, or research papers that discuss changes in user behavior, market trends, or external factors that could impact the data.
Look for news matching the datetime of the columns to see if any events happened around that time that could explain the drift like environmental tradegies or events like storms , etc.
You will use the google_search tool to look for articles, papers, blog posts, or any other online resources that discuss potential causes of data drift in datasets similar to the one being analyzed.
Make sure that the information you provide is relevant to the specific column that has drifted and is backed by credible sources.
Also make sure if you are refering to any news to back yoyr reasoning, you provide the date of the news article to show that it matches the datetime of the data.
Provide a concise summary of your findings, including links to the most relevant sources you discovered during your search.
Do not speculate beyond the information you find on the internet, and do not create your own reasons without backing from credible sources.

Only give information related to the mentioned timelines or close to the mentioned timelines in the data.

DO NOT HALLUCINATE ANY INFORMATION OR MAKE UP ANY NEWS ARTICLES OR EVENTS.

save the information you find in the output key "internet_info"

In the message that you are replying, add a section at the end titled "Sources" where you list all the links to the sources you used to gather your information.
""",

)
