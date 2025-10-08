"""
Vercel handler pour Finance MCP Agent
"""
import sys
import os

# Ajouter le path shared au PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from .agent import FinanceMCPAgent
from fastapi import FastAPI

# Créer l'instance de l'agent
finance_agent = FinanceMCPAgent()
inner_app = finance_agent.app
app = FastAPI()
app.mount("/finance", inner_app)
app.mount("/", inner_app)