"""Gradio application for Track-Explorer."""

import os

import yaml
from dotenv import load_dotenv
from smolagents import LiteLLMModel

from track_explorer import SmolAgentsAnalyst

load_dotenv()


with open("configs/agent.yml", "r") as f:
    agent_config = yaml.safe_load(f)

os.environ["TRACK_EXPLORER_DB_URI"] = "sqlite:///gradio-demo.db"

llm = LiteLLMModel(
    model_id=agent_config["llm"]["model"],
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=agent_config["llm"]["temperature"],
)
analyst = SmolAgentsAnalyst(db_uri="sqlite:///gradio-demo.db", llm=llm, system_prompt=agent_config["system_prompt"])
analyst.launch()
