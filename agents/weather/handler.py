"""
Vercel handler pour Weather MCP Agent
"""
import sys
import os

# Ajouter le path shared au PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from .agent import WeatherMCPAgent
from fastapi import FastAPI

# Créer l'instance de l'agent
weather_agent = WeatherMCPAgent()
inner_app = weather_agent.app
app = FastAPI()
app.mount("/weather", inner_app)
app.mount("/", inner_app)