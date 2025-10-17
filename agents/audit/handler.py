"""
Vercel handler pour Document Extraction & Chunking Audit MCP Agent
"""
import sys
import os

# Ajouter le path shared au PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from .agent import DocumentAuditMCPAgent

# Créer l'instance de l'agent
audit_agent = DocumentAuditMCPAgent()
app = audit_agent.app