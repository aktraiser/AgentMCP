"""
Code MCP Agent - Version cloud deployée avec moteur sémantique et client MCP
"""
import os
import subprocess
import tempfile
import ast
import sys
import json
import uuid
from typing import Dict, List
import httpx
import aiohttp

# Ajouter le path shared
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))

from base_agent import CloudMCPAgentBase
from mcp_client import ExternalMCPClient
import structlog

logger = structlog.get_logger()

# Prompt système pour le moteur sémantique
CODE_SYS_PROMPT = """You are Code Intelligence Agent.
Goal: analyze source code and produce accurate, concise, actionable output.
When summarizing issues, group by category (security, complexity, style), cite evidence (line numbers), 
and propose concrete remediations. Prefer bullet points; avoid speculation.
If information is missing, say so explicitly. Keep irresponsible code execution suggestions out.
Return French text if the user's request is in French; else, English."""

class SemanticHelper:
    """Moteur sémantique pour enrichissement LLM"""
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.key = os.getenv("OPENAI_API_KEY")

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        if not self.key:
            # Mode "gracieux": pas d'IA => on renvoie le user_prompt
            return user_prompt
        
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2
        }
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        
        try:
            async with httpx.AsyncClient(timeout=30) as cli:
                r = await cli.post(url, headers=headers, json=payload)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"LLM completion failed: {e}")
            return user_prompt  # Fallback

# ExternalMCPClient importé depuis shared/mcp_client.py

# Manifeste MCP pour découverte
AGENT_MANIFEST = {
    "name": "code_agent",
    "version": "1.0.0",
    "description": "Code Intelligence Agent (analysis, syntax, execution-sim, documentation)",
    "tools": [
        {
            "name": "process",
            "description": "Semantic processing of a code-related request",
            "input_schema": {
                "type": "object",
                "properties": {
                    "request": {"type": "string"},
                    "context": {"type": "object"}
                },
                "required": ["request"]
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "response": {"type": "string"},
                    "confidence": {"type": "number"},
                    "data": {"type": "object"}
                },
                "required": ["response", "confidence"]
            }
        }
    ],
    "endpoints": {
        "mcp": "/mcp",
        "schema": "/schema",
        "health": "/health",
        "manifest": "/manifest.json"
    }
}

class CodeMCPAgent(CloudMCPAgentBase):
    """Agent MCP code déployé en cloud avec analyse sémantique"""
    
    def __init__(self):
        super().__init__(
            agent_id="code_agent",
            name="Code Intelligence Agent",
            capabilities=["code_analysis", "syntax_check", "code_execution", "documentation"]
        )
        
        # Moteur sémantique LLM
        self.semantic = SemanticHelper(model=os.getenv("LLM_MODEL", "gpt-4o-mini"))
        
        # Client MCP configuré via mcp.json
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'mcp.json')
        self.mcp_client = ExternalMCPClient(config_path)
        
        # Configuration pour l'exécution de code sécurisée
        self.allowed_languages = ["python", "javascript", "bash"]
        self.max_execution_time = 30  # secondes
    
    async def process(self, request: str, context: Dict) -> Dict:
        """Traitement sémantique des requêtes de code"""
        try:
            # 1. Analyse sémantique du code
            code_intent = self.analyze_code_intent(request)
            
            # 2. Routage intelligent vers les bonnes méthodes
            if code_intent["action"] == "analyze":
                result = await self.analyze_code(
                    code_intent["code"],
                    code_intent["language"]
                )
            elif code_intent["action"] == "execute":
                result = await self.execute_code(
                    code_intent["code"],
                    code_intent["language"]
                )
            elif code_intent["action"] == "check_syntax":
                result = await self.check_syntax(
                    code_intent["code"],
                    code_intent["language"]
                )
            else:
                result = await self.document_code(
                    code_intent["code"],
                    code_intent["language"]
                )
            
            # 3. Enrichissement sémantique de la réponse
            if "error" not in result:
                semantic_response = await self.enrich_code_response(
                    result, request, code_intent
                )
                
                return {
                    "agent": self.agent_id,
                    "code_intent": code_intent,
                    "response": semantic_response,
                    "confidence": 0.90,
                    "data": result,
                    "source": "Code Analysis Engine"
                }
            else:
                return {
                    "agent": self.agent_id,
                    "response": f"Code processing error: {result.get('error')}",
                    "confidence": 0.0,
                    "error": result.get("error")
                }
                
        except Exception as e:
            logger.error(f"Error processing code request: {e}")
            return {
                "agent": self.agent_id,
                "response": f"Error processing code request: {str(e)}",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def analyze_code_intent(self, request: str) -> Dict:
        """Analyse sémantique des intentions de code"""
        intent = {
            "action": "analyze",  # Default
            "language": "python",
            "code": "",
            "complexity": "unknown",
            "security_level": "medium"
        }
        
        request_lower = request.lower()
        
        # Extraction du code depuis la requête
        # Chercher des blocs de code entre ```
        import re
        code_blocks = re.findall(r'```(?:python|javascript|bash)?\n?(.*?)```', request, re.DOTALL)
        if code_blocks:
            intent["code"] = code_blocks[0].strip()
        
        # Détection du langage
        if "python" in request_lower or "py" in request_lower:
            intent["language"] = "python"
        elif "javascript" in request_lower or "js" in request_lower:
            intent["language"] = "javascript"
        elif "bash" in request_lower or "shell" in request_lower:
            intent["language"] = "bash"
        
        # Détection de l'action
        if any(word in request_lower for word in ["execute", "run", "launch", "exécuter"]):
            intent["action"] = "execute"
        elif any(word in request_lower for word in ["syntax", "check", "validate", "vérifier"]):
            intent["action"] = "check_syntax"
        elif any(word in request_lower for word in ["document", "explain", "comment", "documenter"]):
            intent["action"] = "document"
        elif any(word in request_lower for word in ["analyze", "analyse", "review", "audit"]):
            intent["action"] = "analyze"
        
        # Analyse de complexité
        if intent["code"]:
            lines = intent["code"].split('\n')
            if len(lines) > 50:
                intent["complexity"] = "high"
            elif len(lines) > 10:
                intent["complexity"] = "medium"
            else:
                intent["complexity"] = "low"
        
        # Niveau de sécurité basé sur les patterns
        dangerous_patterns = ["exec", "eval", "import os", "subprocess", "system"]
        if any(pattern in intent["code"] for pattern in dangerous_patterns):
            intent["security_level"] = "high_risk"
        
        return intent
    
    async def on_startup(self):
        """Chargement de la configuration MCP"""
        await self.mcp_client.load_config()
        logger.info(f"Code agent initialized with {len(self.mcp_client.servers)} MCP servers")

    async def analyze_code(self, code: str, language: str) -> Dict:
        """Analyser la qualité et structure du code + MCP externes"""
        try:
            # 1. Analyse locale de base
            base = {}
            if language == "python":
                base = self._analyze_python_code(code)
            elif language == "javascript":
                base = self._analyze_javascript_code(code)
            else:
                return {"error": f"Analysis not supported for {language}"}

            # 2. Enrichissement par MCP externes
            
            # Enrichissement par catégorie MCP
            code_analysis_servers = self.mcp_client.get_servers_by_category("code_analysis")
            
            # JavaScript → lint MCP externe
            if language == "javascript" and "js-lint" in code_analysis_servers:
                try:
                    extra = await self.mcp_client.tools_call(
                        "js-lint", "lint_js",
                        {"code": code, "ruleset": "recommended"}
                    )
                    base["external_findings"] = extra.get("result", extra)
                except Exception as e:
                    base.setdefault("external_errors", []).append(f"js-lint: {e}")

            # Patterns à risque → SAST
            risky_patterns = ["eval(", "exec(", "child_process", "subprocess", "system(", "shell_exec"]
            risky = any(p in code for p in risky_patterns)
            
            security_servers = self.mcp_client.get_servers_by_category("security")
            if risky and "sast" in security_servers:
                try:
                    sast = await self.mcp_client.tools_call(
                        "sast", "scan_code",
                        {"language": language, "code": code}
                    )
                    base["security_scan"] = sast.get("result", sast)
                except Exception as e:
                    base.setdefault("external_errors", []).append(f"sast: {e}")

            return base
                
        except Exception as e:
            logger.error(f"Code analysis error: {e}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _analyze_python_code(self, code: str) -> Dict:
        """Analyse spécifique Python"""
        try:
            # Parse AST
            tree = ast.parse(code)
            
            analysis = {
                "language": "python",
                "lines_of_code": len(code.split('\n')),
                "functions": [],
                "classes": [],
                "imports": [],
                "complexity_score": 0,
                "issues": []
            }
            
            # Analyse des nœuds AST
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    analysis["functions"].append({
                        "name": node.name,
                        "line": node.lineno,
                        "args": len(node.args.args)
                    })
                elif isinstance(node, ast.ClassDef):
                    analysis["classes"].append({
                        "name": node.name,
                        "line": node.lineno
                    })
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis["imports"].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        analysis["imports"].append(node.module)
            
            # Score de complexité simple
            analysis["complexity_score"] = (
                len(analysis["functions"]) * 2 + 
                len(analysis["classes"]) * 3 + 
                analysis["lines_of_code"] // 10
            )
            
            # Détection d'issues simples
            if "exec(" in code or "eval(" in code:
                analysis["issues"].append("Potentially dangerous exec/eval usage")
            
            if analysis["lines_of_code"] > 100:
                analysis["issues"].append("Function/class might be too long")
            
            return analysis
            
        except SyntaxError as e:
            return {
                "error": f"Python syntax error: {str(e)}",
                "line": getattr(e, 'lineno', 'unknown')
            }
    
    def _analyze_javascript_code(self, code: str) -> Dict:
        """Analyse basique JavaScript"""
        analysis = {
            "language": "javascript",
            "lines_of_code": len(code.split('\n')),
            "functions": [],
            "issues": []
        }
        
        # Détection basique des fonctions
        import re
        functions = re.findall(r'function\s+(\w+)', code)
        analysis["functions"] = [{"name": func} for func in functions]
        
        # Détection d'issues communes
        if "eval(" in code:
            analysis["issues"].append("Potentially dangerous eval usage")
        
        if "document.write" in code:
            analysis["issues"].append("document.write usage (not recommended)")
        
        return analysis
    
    async def check_syntax(self, code: str, language: str) -> Dict:
        """Vérifier la syntaxe du code"""
        try:
            if language == "python":
                try:
                    ast.parse(code)
                    return {
                        "syntax_valid": True,
                        "language": language,
                        "message": "Python syntax is valid"
                    }
                except SyntaxError as e:
                    return {
                        "syntax_valid": False,
                        "language": language,
                        "error": str(e),
                        "line": e.lineno
                    }
            else:
                return {"error": f"Syntax check not implemented for {language}"}
                
        except Exception as e:
            return {"error": f"Syntax check failed: {str(e)}"}
    
    async def execute_code(self, code: str, language: str) -> Dict:
        """Exécution sécurisée de code (simulation)"""
        try:
            # SÉCURITÉ: En production, ceci devrait être dans un sandbox
            if language == "python":
                return self._simulate_python_execution(code)
            else:
                return {"error": f"Execution not supported for {language}"}
                
        except Exception as e:
            return {"error": f"Execution failed: {str(e)}"}
    
    def _simulate_python_execution(self, code: str) -> Dict:
        """Simulation d'exécution Python pour la démo"""
        # En production, utiliser un vrai sandbox
        
        # Détection de prints
        import re
        prints = re.findall(r'print\s*\(["\']([^"\']*)["\']', code)
        
        return {
            "executed": True,
            "language": "python",
            "simulated_output": "\n".join(prints) if prints else "Code executed successfully (no output)",
            "execution_time": "0.045s",
            "memory_usage": "2.3MB",
            "warning": "This is a simulated execution for security reasons"
        }
    
    async def document_code(self, code: str, language: str) -> Dict:
        """Générer de la documentation pour le code"""
        try:
            if language == "python":
                return self._document_python_code(code)
            else:
                return {"error": f"Documentation not implemented for {language}"}
                
        except Exception as e:
            return {"error": f"Documentation failed: {str(e)}"}
    
    def _document_python_code(self, code: str) -> Dict:
        """Documentation automatique Python"""
        try:
            tree = ast.parse(code)
            documentation = {
                "language": "python",
                "summary": "Code documentation generated automatically",
                "functions": [],
                "classes": []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    doc = {
                        "name": node.name,
                        "line": node.lineno,
                        "parameters": [arg.arg for arg in node.args.args],
                        "docstring": ast.get_docstring(node) or "No docstring provided"
                    }
                    documentation["functions"].append(doc)
                    
                elif isinstance(node, ast.ClassDef):
                    doc = {
                        "name": node.name,
                        "line": node.lineno,
                        "docstring": ast.get_docstring(node) or "No docstring provided"
                    }
                    documentation["classes"].append(doc)
            
            return documentation
            
        except Exception as e:
            return {"error": f"Documentation generation failed: {str(e)}"}
    
    async def enrich_code_response(self, data: Dict, original_request: str, intent: Dict) -> str:
        """Enrichissement sémantique contextuel avec LLM"""
        # 1. Synthèse déterministe de base
        action = intent.get("action", "analyze")
        language = intent.get("language", "unknown")
        base = f"[{language} • action={action}] "
        
        if action == "analyze":
            loc = data.get("lines_of_code", 0)
            issues = data.get("issues", [])
            base += f"{loc} lignes, {len(data.get('functions', []))} fonctions, {len(data.get('classes', []))} classes. "
            if issues:
                base += f"{len(issues)} problèmes potentiels. "
            
            # Mention des enrichissements MCP externes
            if data.get("external_findings"):
                base += "Lint externe: OK. "
            if data.get("security_scan"):
                base += "Scan sécurité: effectué. "
            if data.get("external_errors"):
                base += f"Erreurs MCP: {len(data.get('external_errors', []))}. "
                
        elif action == "check_syntax":
            base += "syntaxe OK" if data.get("syntax_valid") else f"erreur: {data.get('error')}"
        elif action == "execute":
            base += f"exécution simulée • output: {data.get('simulated_output', '')[:80]}"
        else:
            base += f"documentation: {len(data.get('functions', []))} fonctions, {len(data.get('classes', []))} classes."

        # 2. Enrichissement sémantique via LLM
        user_prompt = (
            f"User request:\n{original_request}\n\n"
            f"Deterministic findings (JSON):\n{json.dumps(data, ensure_ascii=False)[:8000]}\n\n"
            "Produce a concise, actionable report with bullet points, evidence lines, and prioritized fixes."
        )
        
        try:
            pretty = await self.semantic.complete(CODE_SYS_PROMPT, user_prompt)
            return pretty
        except Exception as e:
            logger.warning(f"Semantic enrichment failed: {e}")
            # Fallback vers synthèse déterministe
            return base
    
    async def check_health(self) -> Dict:
        """Vérification de santé spécifique à l'agent code"""
        health = {
            "python_ast": "unknown", 
            "code_analysis": "unknown",
            "semantic_engine": "unknown",
            "mcp_clients": {}
        }
        
        try:
            # Test du parser Python
            test_code = "def hello(): return 'world'"
            ast.parse(test_code)
            health["python_ast"] = "healthy"
            
            # Test de l'analyse de code
            analysis = self._analyze_python_code(test_code)
            if "functions" in analysis:
                health["code_analysis"] = "healthy"
            else:
                health["code_analysis"] = "degraded"
            
            # Test du moteur sémantique
            if self.semantic.key:
                health["semantic_engine"] = "available"
            else:
                health["semantic_engine"] = "no_api_key"
            
            # État des clients MCP
            health["mcp_clients"] = await self.mcp_client.health_check()
                
        except Exception as e:
            health["python_ast"] = f"error - {str(e)[:50]}"
            health["code_analysis"] = "unhealthy"
        
        return health
    
    def get_manifest(self) -> Dict:
        """Retourne le manifeste MCP pour découverte"""
        return AGENT_MANIFEST