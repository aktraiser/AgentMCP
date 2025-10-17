"""
Vercel handler pour Master MCP Agent
"""
import sys
import os

# Ajouter le path shared au PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from .agent import MasterMCPAgent

# Créer l'instance de l'agent
master_agent = MasterMCPAgent()
app = master_agent.app

# Exposer le handler pour Vercel avec mounting explicite
# Vercel va chercher une fonction 'handler' ou une app FastAPI
def handler(request):
    """Handler Vercel compatible"""
    return app(request)