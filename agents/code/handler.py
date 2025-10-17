"""
Vercel handler pour Code MCP Agent
"""
import sys
import os

# Ajouter le path shared au PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from .agent import CodeMCPAgent

# Créer l'instance de l'agent
code_agent = CodeMCPAgent()
app = code_agent.app

# Exposer le handler pour Vercel avec mounting explicite
def handler(request):
    """Handler Vercel compatible"""
    return app(request)