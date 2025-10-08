"""
Vercel handler pour Finance MCP Agent
"""
import sys
import os

# Ajouter le path shared au PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))

from mangum import Mangum
from agent import FinanceMCPAgent

# Créer l'instance de l'agent
finance_agent = FinanceMCPAgent()

# Adapter pour Vercel/Lambda
handler = Mangum(finance_agent.app, lifespan="off")