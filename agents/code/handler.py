"""
Vercel handler pour Code MCP Agent
"""
import sys
import os

# Ajouter le path shared au PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))

from mangum import Mangum
from .agent import CodeMCPAgent

# Créer l'instance de l'agent
code_agent = CodeMCPAgent()

# Adapter pour Vercel/Lambda
handler = Mangum(code_agent.app, lifespan="off")