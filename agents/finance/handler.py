"""
Vercel handler pour Finance MCP Agent
"""
import sys
import os

# Ajouter le path shared au PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from .agent import FinanceMCPAgent

# Créer l'instance de l'agent
finance_agent = FinanceMCPAgent()
app = finance_agent.app

# Exposer le handler pour Vercel avec mounting explicite
def handler(request):
    """Handler Vercel compatible"""
    return app(request)