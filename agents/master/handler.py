"""
Vercel handler pour Master MCP Agent
"""
import sys
import os

# Ajouter le path shared au PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))

from mangum import Mangum
from .agent import MasterMCPAgent

# Créer l'instance de l'agent
master_agent = MasterMCPAgent()
app = master_agent.app

# Adapter pour Vercel/Lambda
handler = Mangum(app, lifespan="off")