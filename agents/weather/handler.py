"""
Vercel handler pour Weather MCP Agent
"""
import sys
import os

# Ajouter le path shared au PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))

from mangum import Mangum
from agent import WeatherMCPAgent

# Créer l'instance de l'agent
weather_agent = WeatherMCPAgent()

# Adapter pour Vercel/Lambda
handler = Mangum(weather_agent.app, lifespan="off")