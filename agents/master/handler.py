"""
Vercel handler pour Master MCP Agent
"""
import sys
import os

# Ajouter le path shared au PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from .agent import MasterMCPAgent
from fastapi import FastAPI

# Créer l'instance de l'agent
master_agent = MasterMCPAgent()
inner_app = master_agent.app
app = FastAPI()
app.mount("/master", inner_app)
app.mount("/", inner_app)