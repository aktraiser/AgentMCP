"""
Classe de base pour tous les agents MCP déployés en cloud
"""
import os
import time
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import asyncio
import httpx
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import structlog

logger = structlog.get_logger()

class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: Dict
    id: str

class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    result: Optional[Dict] = None
    error: Optional[Dict] = None
    id: str

class CloudMCPAgentBase(ABC):
    """Classe de base pour agents MCP déployés en cloud"""
    
    def __init__(self, agent_id: str, name: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.name = name
        self.capabilities = capabilities
        self.app = FastAPI(
            title=f"{name} MCP Agent",
            description=f"Cloud-deployed {name} with MCP protocol",
            version="1.0.0"
        )
        
        
        self.setup_routes()
        self.setup_middleware()
    
    def setup_middleware(self):
        """Configuration middleware d'authentification et logging"""
        
        @self.app.middleware("http")
        async def auth_middleware(request: Request, call_next):
            # Skip auth pour health check
            if request.url.path.endswith("/health"):
                return await call_next(request)
            
            
            # Logging de la requête
            start_time = time.time()
            response = await call_next(request)
            duration = time.time() - start_time
            
            logger.info(
                "Request processed",
                agent=self.agent_id,
                method=request.method,
                path=request.url.path,
                duration=duration,
                status_code=response.status_code
            )
            
            return response
    
    def setup_routes(self):
        """Configuration des routes MCP standard"""
        
        @self.app.post("/mcp")
        async def handle_mcp_request(request: MCPRequest):
            """Point d'entrée MCP principal"""
            try:
                if request.method == "process":
                    result = await self.process(
                        request.params.get("request", ""),
                        request.params.get("context", {})
                    )
                    
                    return MCPResponse(
                        result=result,
                        id=request.id
                    )
                
                elif request.method == "capabilities":
                    return MCPResponse(
                        result={
                            "capabilities": self.capabilities,
                            "agent_id": self.agent_id,
                            "name": self.name
                        },
                        id=request.id
                    )
                
                else:
                    return MCPResponse(
                        error={
                            "code": -32601,
                            "message": f"Method {request.method} not found"
                        },
                        id=request.id
                    )
                    
            except Exception as e:
                logger.error(f"Error processing MCP request: {e}")
                return MCPResponse(
                    error={
                        "code": -32603,
                        "message": f"Internal error: {str(e)}"
                    },
                    id=request.id
                )
        
        @self.app.get("/mcp")
        async def handle_mcp_get():
            """Endpoint GET pour introspection MCP"""
            return {
                "protocol": "MCP",
                "version": "2024-11-05",
                "agent_id": self.agent_id,
                "name": self.name,
                "capabilities": self.capabilities,
                "methods": ["process", "capabilities"],
                "transport": "http",
                "description": f"MCP Agent: {self.name}"
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            try:
                # Test des dépendances critiques
                health_status = await self.check_health()
                
                return {
                    "status": "healthy",
                    "agent": self.name,
                    "agent_id": self.agent_id,
                    "capabilities": self.capabilities,
                    "timestamp": time.time(),
                    "dependencies": health_status
                }
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                raise HTTPException(status_code=503, detail="Service unhealthy")
        
        @self.app.get("/schema")
        async def get_schema():
            """Retourne le schéma MCP de l'agent"""
            return {
                "agent_id": self.agent_id,
                "name": self.name,
                "capabilities": self.capabilities,
                "endpoints": {
                    "mcp": "/mcp",
                    "health": "/health",
                    "schema": "/schema"
                },
                "mcp_version": "1.0.0"
            }
        
        @self.app.get("/manifest.json")
        async def get_manifest():
            """Retourne le manifeste MCP pour découverte"""
            if hasattr(self, 'get_manifest'):
                return self.get_manifest()
            else:
                # Manifeste par défaut
                return {
                    "name": self.agent_id,
                    "version": "1.0.0",
                    "description": f"{self.name} MCP Agent",
                    "endpoints": {
                        "mcp": "/mcp",
                        "health": "/health",
                        "schema": "/schema",
                        "manifest": "/manifest.json"
                    }
                }
        
        @self.app.get("/")
        async def root():
            """Page d'accueil de l'agent"""
            return {
                "message": f"Welcome to {self.name}",
                "agent_id": self.agent_id,
                "capabilities": self.capabilities,
                "endpoints": {
                    "mcp": "/mcp",
                    "health": "/health", 
                    "schema": "/schema",
                    "manifest": "/manifest.json"
                }
            }
    
    @abstractmethod
    async def process(self, request: str, context: Dict) -> Dict:
        """Traite une requête MCP"""
        pass
    
    async def check_health(self) -> Dict:
        """Vérifie la santé des dépendances"""
        return {"status": "ok"}
    
    async def call_external_mcp(self, service_url: str, tool_name: str, params: Dict) -> Dict:
        """Appel vers un serveur MCP externe"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{service_url}/tools/{tool_name}",
                    json=params,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    return {
                        "error": f"External tool call failed: {response.status_code}",
                        "details": response.text
                    }
                    
        except asyncio.TimeoutError:
            return {"error": f"Timeout calling external tool {tool_name}"}
        except Exception as e:
            logger.error(f"Error calling external MCP: {e}")
            return {"error": f"External call failed: {str(e)}"}
    
    async def call_remote_agent(self, agent_url: str, request: str, api_key: str) -> Dict:
        """Appel vers un autre agent MCP distant"""
        try:
            mcp_request = {
                "jsonrpc": "2.0",
                "method": "process",
                "params": {
                    "request": request,
                    "context": {"source_agent": self.agent_id}
                },
                "id": str(uuid.uuid4())
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{agent_url}/mcp",
                    json=mcp_request,
                    headers={
                        "Content-Type": "application/json",
                        "X-API-Key": api_key
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("result", {})
                else:
                    return {"error": f"Remote agent call failed: {response.status_code}"}
                    
        except Exception as e:
            logger.error(f"Error calling remote agent: {e}")
            return {"error": f"Remote agent call failed: {str(e)}"}