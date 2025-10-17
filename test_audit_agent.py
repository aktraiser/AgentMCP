#!/usr/bin/env python3
"""
Script de test pour l'agent MCP d'audit
Usage: python test_audit_agent.py
"""
import asyncio
import json
import os
import sys
from pathlib import Path

# Ajouter le path pour import
sys.path.append(str(Path(__file__).parent / "shared"))
sys.path.append(str(Path(__file__).parent / "agents" / "audit"))

from agent import DocumentAuditMCPAgent

async def test_audit_agent():
    """Test de l'agent d'audit"""
    print("🧪 Test de l'Agent MCP d'Audit d'Extraction et Chunking")
    print("=" * 60)
    
    # Créer l'agent
    agent = DocumentAuditMCPAgent()
    
    # Tests à effectuer
    tests = [
        {
            "name": "Audit de santé système",
            "request": "audit système health status"
        },
        {
            "name": "Audit extraction général",
            "request": "audit extraction docling performance"
        },
        {
            "name": "Audit chunking",
            "request": "audit chunking strategy quality"
        },
        {
            "name": "Audit pipeline complet",
            "request": "audit full pipeline document processing"
        }
    ]
    
    results = {}
    
    for test in tests:
        print(f"\n🔍 {test['name']}")
        print("-" * 40)
        
        try:
            # Exécuter le test
            result = await agent.process(test['request'], {})
            
            # Afficher les résultats
            if result.get("error"):
                print(f"❌ Erreur: {result['error']}")
            else:
                confidence = result.get('confidence', 0.0)
                response = result.get('response', 'Pas de réponse')
                
                print(f"✅ Confiance: {confidence:.1%}")
                print(f"📄 Rapport:")
                print(response[:300] + "..." if len(response) > 300 else response)
                
                # Afficher métriques si disponibles
                audit_data = result.get('audit_data', {})
                if audit_data and not audit_data.get('error'):
                    print(f"📊 Données d'audit disponibles: {list(audit_data.keys())}")
            
            results[test['name']] = result
            
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
            results[test['name']] = {"error": str(e)}
    
    # Résumé
    print(f"\n📋 Résumé des tests")
    print("=" * 40)
    successful_tests = sum(1 for r in results.values() if not r.get("error"))
    total_tests = len(results)
    print(f"✅ Tests réussis: {successful_tests}/{total_tests}")
    
    if successful_tests < total_tests:
        print("\n⚠️ Tests échoués:")
        for name, result in results.items():
            if result.get("error"):
                print(f"  - {name}: {result['error'][:100]}")
    
    # Test de santé de l'agent
    print(f"\n🏥 Santé de l'agent d'audit")
    print("-" * 30)
    try:
        health = await agent.check_health()
        print(f"Statut: {health.get('audit_engine', 'unknown')}")
        
        connections = health.get('external_connections', {})
        for service, status in connections.items():
            emoji = "✅" if status in ["reachable", "healthy"] else "⚙️" if status == "not_configured" else "❌"
            print(f"{emoji} {service}: {status}")
        
        stats = health.get('audit_stats', {})
        if stats:
            print(f"📊 Audits effectués: {stats.get('total_audits', 0)}")
            
    except Exception as e:
        print(f"❌ Erreur santé agent: {e}")

async def test_mcp_client():
    """Test du client MCP"""
    print(f"\n🔌 Test du client MCP")
    print("-" * 30)
    
    agent = DocumentAuditMCPAgent()
    
    try:
        # Charger la config MCP
        await agent.mcp_client.load_config()
        
        # Afficher les serveurs configurés
        servers = agent.mcp_client.servers
        print(f"Serveurs MCP configurés: {len(servers)}")
        
        for name, config in servers.items():
            enabled = config.get('enabled', True)
            server_type = config.get('type', 'unknown')
            status_emoji = "✅" if enabled else "⏸️"
            print(f"{status_emoji} {name} ({server_type})")
        
        # Test de santé des serveurs
        if servers:
            print("\n🔍 Test de santé des serveurs MCP:")
            health = await agent.mcp_client.health_check()
            for server, status in health.items():
                emoji = "✅" if status == "healthy" else "❌" if "error" in status else "⚠️"
                print(f"{emoji} {server}: {status}")
        else:
            print("⚠️ Aucun serveur MCP configuré")
            
    except Exception as e:
        print(f"❌ Erreur client MCP: {e}")

if __name__ == "__main__":
    print("🚀 Lancement des tests...")
    
    # Variables d'environnement pour test
    os.environ.setdefault("CONTEXT_RAG_BASE_URL", "http://localhost:3000")
    os.environ.setdefault("DOCLING_URL", "http://localhost:8090")
    
    async def run_all_tests():
        await test_mcp_client()
        await test_audit_agent()
    
    try:
        asyncio.run(run_all_tests())
        print(f"\n✅ Tests terminés!")
    except KeyboardInterrupt:
        print(f"\n⏹️ Tests interrompus par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur générale: {e}")