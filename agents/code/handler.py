"""
Vercel handler pour Code MCP Agent
"""
import sys
import os

# Ajouter le path shared au PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from .agent import CodeMCPAgent
from fastapi import FastAPI

# Créer l'instance de l'agent
code_agent = CodeMCPAgent()
inner_app = code_agent.app
app = FastAPI()
app.mount("/code", inner_app)
app.mount("/", inner_app)