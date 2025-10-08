"""
Master MCP Agent - Version cloud deployée
Orchestration intelligente des agents MCP spécialisés
"""
import os
import uuid
import asyncio
from typing import Dict, List, Optional
import sys
from datetime import datetime

# Ajouter le path shared
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))

from base_agent import CloudMCPAgentBase
import structlog

logger = structlog.get_logger()

class MasterMCPAgent(CloudMCPAgentBase):
    """Agent MCP Master pour orchestration intelligente"""
    
    def __init__(self):
        super().__init__(
            agent_id="master_agent",
            name="Master Orchestration Agent",
            capabilities=["orchestration", "agent_routing", "response_synthesis", "intent_analysis"]
        )
        
        # Configuration des agents distants
        self.agents_config = {
            "weather": {
                "url": os.getenv("WEATHER_AGENT_URL", "https://weather-agent.vercel.app"),
                "api_key": os.getenv("WEATHER_AGENT_KEY", ""),
                "capabilities": ["weather", "forecast", "climate_analysis"]
            },
            "finance": {
                "url": os.getenv("FINANCE_AGENT_URL", "https://finance-agent.vercel.app"),
                "api_key": os.getenv("FINANCE_AGENT_KEY", ""),
                "capabilities": ["stocks", "markets", "portfolio_analysis"]
            },
            "code": {
                "url": os.getenv("CODE_AGENT_URL", "https://code-agent.vercel.app"),
                "api_key": os.getenv("CODE_AGENT_KEY", ""),
                "capabilities": ["code_analysis", "syntax_check", "code_execution"]
            }
        }
        
        # Statistiques d'orchestration
        self.orchestration_stats = {
            "total_requests": 0,
            "successful_routes": 0,
            "failed_routes": 0,
            "average_confidence": 0.0,
            "agent_usage": {agent: 0 for agent in self.agents_config.keys()}
        }
    
    async def process(self, request: str, context: Dict) -> Dict:
        """Orchestration intelligente des requêtes"""
        try:
            self.orchestration_stats["total_requests"] += 1
            start_time = datetime.now()
            
            # 1. Analyse sémantique de l'intention globale
            intent_analysis = self.analyze_global_intent(request)
            
            # 2. Routage intelligent vers les agents appropriés
            routing_plan = self.create_routing_plan(intent_analysis)
            
            # 3. Exécution parallèle des appels aux agents
            agent_responses = await self.execute_agent_calls(routing_plan, request, context)
            
            # 4. Synthèse intelligente des réponses
            synthesized_response = self.synthesize_responses(
                agent_responses, intent_analysis, request
            )
            
            # 5. Métriques et logging
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_orchestration_metrics(routing_plan, agent_responses, execution_time)
            
            return {
                "agent": self.agent_id,
                "intent_analysis": intent_analysis,
                "routing_plan": routing_plan,
                "response": synthesized_response,
                "confidence": self._calculate_global_confidence(agent_responses),
                "execution_time": execution_time,
                "agents_used": list(routing_plan.keys()),
                "timestamp": datetime.now().isoformat(),
                "orchestration_id": str(uuid.uuid4())[:8]
            }
            
        except Exception as e:
            logger.error(f"Master orchestration error: {e}")
            self.orchestration_stats["failed_routes"] += 1
            return {
                "agent": self.agent_id,
                "response": f"Orchestration error: {str(e)}",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def analyze_global_intent(self, request: str) -> Dict:
        """Analyse sémantique globale pour comprendre l'intention"""
        intent = {
            "primary_domain": "general",
            "secondary_domains": [],
            "complexity": "medium",
            "requires_multi_agent": False,
            "urgency": "normal",
            "context_type": "informational"
        }
        
        request_lower = request.lower()
        
        # Détection des domaines principaux
        domain_keywords = {
            "weather": ["weather", "météo", "temperature", "rain", "forecast", "climate"],
            "finance": ["stock", "finance", "market", "price", "portfolio", "investment", "trading"],
            "code": ["code", "python", "javascript", "function", "syntax", "execute", "programming"]
        }
        
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in request_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            # Domaine principal = score le plus élevé
            intent["primary_domain"] = max(domain_scores, key=domain_scores.get)
            
            # Domaines secondaires = autres domaines avec score > 0
            intent["secondary_domains"] = [
                domain for domain, score in domain_scores.items() 
                if score > 0 and domain != intent["primary_domain"]
            ]
        
        # Détection de complexité multi-agents
        multi_agent_indicators = ["compare", "analyze both", "weather and stock", "code and market"]
        if any(indicator in request_lower for indicator in multi_agent_indicators):
            intent["requires_multi_agent"] = True
            intent["complexity"] = "high"
        
        # Détection d'urgence
        urgency_keywords = ["urgent", "quickly", "asap", "now", "immediately"]
        if any(keyword in request_lower for keyword in urgency_keywords):
            intent["urgency"] = "high"
        
        # Type de contexte
        if any(word in request_lower for word in ["explain", "what is", "how does"]):
            intent["context_type"] = "educational"
        elif any(word in request_lower for word in ["execute", "run", "do", "perform"]):
            intent["context_type"] = "action"
        
        return intent
    
    def create_routing_plan(self, intent_analysis: Dict) -> Dict:
        """Créer un plan de routage intelligent"""
        routing_plan = {}
        
        primary_domain = intent_analysis["primary_domain"]
        secondary_domains = intent_analysis["secondary_domains"]
        requires_multi_agent = intent_analysis["requires_multi_agent"]
        
        # Routage principal
        if primary_domain in self.agents_config:
            routing_plan[primary_domain] = {
                "priority": "primary",
                "agent_config": self.agents_config[primary_domain],
                "expected_confidence": 0.9
            }
        
        # Routage secondaire si multi-agent requis
        if requires_multi_agent or len(secondary_domains) > 0:
            for domain in secondary_domains:
                if domain in self.agents_config:
                    routing_plan[domain] = {
                        "priority": "secondary",
                        "agent_config": self.agents_config[domain],
                        "expected_confidence": 0.7
                    }
        
        # Fallback si aucun domaine détecté
        if not routing_plan:
            # Router vers l'agent le plus généraliste (weather par défaut)
            routing_plan["weather"] = {
                "priority": "fallback",
                "agent_config": self.agents_config["weather"],
                "expected_confidence": 0.5
            }
        
        return routing_plan
    
    async def execute_agent_calls(self, routing_plan: Dict, request: str, context: Dict) -> Dict:
        """Exécution parallèle des appels aux agents"""
        agent_responses = {}
        
        # Créer les tâches asynchrones pour chaque agent
        tasks = []
        agent_names = []
        
        for agent_name, plan in routing_plan.items():
            agent_config = plan["agent_config"]
            task = self.call_remote_agent(
                agent_config["url"],
                request,
                agent_config["api_key"]
            )
            tasks.append(task)
            agent_names.append(agent_name)
        
        # Exécution parallèle avec timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=30.0  # 30 secondes max
            )
            
            # Traitement des résultats
            for i, result in enumerate(results):
                agent_name = agent_names[i]
                if isinstance(result, Exception):
                    agent_responses[agent_name] = {
                        "error": f"Agent call failed: {str(result)}",
                        "confidence": 0.0,
                        "status": "failed"
                    }
                else:
                    agent_responses[agent_name] = result
                    agent_responses[agent_name]["status"] = "success"
                    
                # Mise à jour des statistiques
                self.orchestration_stats["agent_usage"][agent_name] += 1
            
        except asyncio.TimeoutError:
            # Gérer le timeout global
            for agent_name in agent_names:
                agent_responses[agent_name] = {
                    "error": "Agent call timeout",
                    "confidence": 0.0,
                    "status": "timeout"
                }
        
        return agent_responses
    
    def synthesize_responses(self, agent_responses: Dict, intent_analysis: Dict, original_request: str) -> str:
        """Synthèse intelligente des réponses multiples"""
        successful_responses = {
            name: resp for name, resp in agent_responses.items()
            if resp.get("status") == "success" and resp.get("confidence", 0) > 0.5
        }
        
        if not successful_responses:
            return "Sorry, I couldn't process your request. All agents encountered errors or returned low confidence results."
        
        # Synthèse basée sur le nombre de réponses
        if len(successful_responses) == 1:
            # Une seule réponse réussie
            agent_name, response = next(iter(successful_responses.items()))
            agent_response = response.get("response", "No response available")
            confidence = response.get("confidence", 0.0)
            
            return f"🎯 {agent_name.title()} Agent Response (confidence: {confidence:.0%}): {agent_response}"
        
        else:
            # Synthèse multi-agents
            synthesis = f"🔮 Multi-Agent Analysis for: '{original_request[:50]}...'\n\n"
            
            # Trier par confiance
            sorted_responses = sorted(
                successful_responses.items(),
                key=lambda x: x[1].get("confidence", 0),
                reverse=True
            )
            
            for i, (agent_name, response) in enumerate(sorted_responses):
                confidence = response.get("confidence", 0.0)
                agent_response = response.get("response", "No response")
                
                synthesis += f"📊 {agent_name.title()} Agent (confidence: {confidence:.0%}):\n"
                synthesis += f"   {agent_response}\n\n"
            
            # Conclusion synthétique
            avg_confidence = sum(r.get("confidence", 0) for r in successful_responses.values()) / len(successful_responses)
            synthesis += f"🎯 Overall Confidence: {avg_confidence:.0%} | Agents Used: {len(successful_responses)}"
            
            return synthesis
    
    def _calculate_global_confidence(self, agent_responses: Dict) -> float:
        """Calcul de la confiance globale"""
        successful_responses = [
            resp for resp in agent_responses.values()
            if resp.get("status") == "success"
        ]
        
        if not successful_responses:
            return 0.0
        
        confidences = [resp.get("confidence", 0.0) for resp in successful_responses]
        return sum(confidences) / len(confidences)
    
    def _update_orchestration_metrics(self, routing_plan: Dict, agent_responses: Dict, execution_time: float):
        """Mise à jour des métriques d'orchestration"""
        successful_routes = sum(
            1 for resp in agent_responses.values()
            if resp.get("status") == "success"
        )
        
        if successful_routes > 0:
            self.orchestration_stats["successful_routes"] += 1
        else:
            self.orchestration_stats["failed_routes"] += 1
        
        # Mise à jour de la confiance moyenne
        total_confidence = sum(
            resp.get("confidence", 0.0) for resp in agent_responses.values()
            if resp.get("status") == "success"
        )
        
        if successful_routes > 0:
            current_avg = self.orchestration_stats["average_confidence"]
            total_requests = self.orchestration_stats["total_requests"]
            
            # Moyenne mobile simple
            new_avg_confidence = total_confidence / successful_routes
            self.orchestration_stats["average_confidence"] = (
                (current_avg * (total_requests - 1) + new_avg_confidence) / total_requests
            )
    
    async def check_health(self) -> Dict:
        """Vérification de santé du master agent et des agents distants"""
        health = {
            "orchestration_engine": "healthy",
            "remote_agents": {},
            "orchestration_stats": self.orchestration_stats
        }
        
        # Test de chaque agent distant
        for agent_name, config in self.agents_config.items():
            try:
                # Test simple avec timeout court
                response = await asyncio.wait_for(
                    self.call_remote_agent(config["url"], "health check", config["api_key"]),
                    timeout=5.0
                )
                
                if response and not response.get("error"):
                    health["remote_agents"][agent_name] = "healthy"
                else:
                    health["remote_agents"][agent_name] = "unhealthy"
                    
            except asyncio.TimeoutError:
                health["remote_agents"][agent_name] = "timeout"
            except Exception as e:
                health["remote_agents"][agent_name] = f"error - {str(e)[:30]}"
        
        # Vérifier si au moins un agent est disponible
        healthy_agents = sum(
            1 for status in health["remote_agents"].values()
            if status == "healthy"
        )
        
        if healthy_agents == 0:
            health["orchestration_engine"] = "degraded - no healthy agents"
        elif healthy_agents < len(self.agents_config):
            health["orchestration_engine"] = "degraded - some agents unavailable"
        
        return health
    
    async def get_orchestration_stats(self) -> Dict:
        """API pour obtenir les statistiques d'orchestration"""
        return {
            **self.orchestration_stats,
            "agents_configured": len(self.agents_config),
            "average_confidence_percentage": f"{self.orchestration_stats['average_confidence']*100:.1f}%"
        }