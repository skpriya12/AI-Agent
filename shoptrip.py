import os
import requests
from crewai import Agent, Task, Crew
from langchain_community.llms import HuggingFaceHub
from bs4 import BeautifulSoup
import spacy
import re
from datetime import datetime

from crewai_tools import ScrapeWebsiteTool, SerperDevTool

# Initialize the tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# Load Hugging Face Model
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
openai_api_key = os.getenv("SERPER_API_KEY")
llm = HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token=api_key,
    task="text-generation",
)

nlp = spacy.load("en_core_web_sm")

# Planner Agent - Extracts travel intent
def extract_travel_intent(user_input):
    """
    Extracts source, destination, and date from the user's input using Named Entity Recognition (NER).
    """
    doc = nlp(user_input)

    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]

    print(locations, dates)

    if len(locations) >= 2 and dates:
        src, des = locations[:2]  # First two locations are source & destination
        date_str = dates[0]  # First DATE entity found
        #print(src, des,date_str)

    return src, des, date_str



planner = Agent(
    role="Travel Planner",
    goal="Extract user travel intent (source, destination, date).",
    backstory="You're an expert travel planner who understands user queries and extracts travel details.",
    llm=llm,
    allow_delegation=False,
    verbose=True
)


# Output Formatting Agent
editor = Agent(
    role="Travel Assistant",
    goal="Format and present flight details clearly to the user.",
    backstory="You ensure that the retrieved flight data is easy to read and user-friendly.",
    llm=llm,
    allow_delegation=False,
    verbose=True
)

# Define Tasks
plan_task = Task(
    description="Extract travel intent (source, destination, date) from user input.",
    expected_output="Identified travel details {src}, {dest}, {date}.",
    agent=planner
)

edit_task = Task(
    description="Format flight details into a user-friendly response.",
    expected_output="Final formatted travel itinerary.",
    agent=editor
)

# Create Crew
crew = Crew(
    agents=[planner,  editor],
    tasks=[plan_task,  edit_task],
    function_calling_llm=llm,
    verbose=2
)

# Run
user_input = "I want to fly from India to New York on Feb 17."
src, dest, date = extract_travel_intent(user_input)
print(src, dest, date)
#result = search_flights(src, dest, date)

inputs = {
        'src': src,
        'dest': dest,
        'date': date

    }
result = crew.kickoff(inputs=inputs)
print(result)
