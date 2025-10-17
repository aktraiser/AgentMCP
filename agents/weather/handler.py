"""
Vercel handler pour Weather MCP Agent
"""
import sys
import os

# Ajouter le path shared au PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from .agent import WeatherMCPAgent

# Créer l'instance de l'agent
weather_agent = WeatherMCPAgent()
app = weather_agent.app

# Exposer le handler pour Vercel avec mounting explicite
def handler(request):
    """Handler Vercel compatible"""
    return app(request)