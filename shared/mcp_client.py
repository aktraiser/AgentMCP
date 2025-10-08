"""
MCP Client configuré via mcp.json avec support HTTP et STDIO
"""
import json
import os
import re
import uuid
import asyncio
import subprocess
from typing import Dict, List, Any, Optional
import aiohttp
import structlog

logger = structlog.get_logger()

def expand_env(value: str) -> str:
    """Remplace ${VAR_NAME} par les variables d'environnement"""
    return re.sub(r"\$\{([^}]+)\}", lambda m: os.getenv(m.group(1), ""), value)

class ExternalMCPClient:
    """Client MCP configurable via mcp.json"""
    
    def __init__(self, config_path: str = "mcp.json"):
        self.config_path = config_path
        self.servers: Dict[str, Dict[str, Any]] = {}
        self.stdio_processes: Dict[str, subprocess.Popen] = {}
        self.routing: Dict[str, List[str]] = {}
    
    async def load_config(self):
        """Charge le fichier mcp.json et initialise les connexions déclarées"""
        if not os.path.exists(self.config_path):
            logger.warning(f"MCP config file {self.config_path} not found")
            return {}
        
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
            
            servers = config.get("servers", {})
            self.routing = config.get("routing", {})
            
            # Charger et valider chaque serveur
            for name, meta in servers.items():
                if not meta.get("enabled", True):
                    logger.info(f"MCP server {name} disabled, skipping")
                    continue
                
                # Expansion des variables d'environnement
                if "url" in meta:
                    meta["url"] = expand_env(meta["url"])
                if "command" in meta:
                    meta["command"] = expand_env(meta["command"])
                if "args" in meta:
                    meta["args"] = [expand_env(arg) for arg in meta["args"]]
                if "env" in meta:
                    meta["env"] = {k: expand_env(v) for k, v in meta["env"].items()}
                
                self.servers[name] = meta
                logger.info(f"Loaded MCP server: {name} ({meta.get('type', 'unknown')})")
            
            logger.info(f"✅ MCP config loaded: {len(self.servers)} servers available")
            return self.servers
            
        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")
            return {}
    
    def get_servers_by_category(self, category: str) -> List[str]:
        """Retourne les serveurs d'une catégorie donnée"""
        return self.routing.get(category, [])
    
    def get_available_tools(self, server_name: str) -> List[str]:
        """Retourne les outils disponibles pour un serveur"""
        if server_name not in self.servers:
            return []
        return self.servers[server_name].get("tools", [])
    
    async def discover_manifest(self, server_name: str) -> Optional[Dict]:
        """Découvre le manifeste d'un serveur HTTP"""
        if server_name not in self.servers:
            return None
        
        server = self.servers[server_name]
        if server.get("type") != "http":
            return None
        
        try:
            manifest_url = f"{server['url'].rstrip('/')}/manifest.json"
            async with aiohttp.ClientSession() as session:
                async with session.get(manifest_url, timeout=10) as response:
                    if response.status == 200:
                        manifest = await response.json()
                        # Mettre à jour les outils disponibles
                        if "tools" in manifest:
                            self.servers[server_name]["tools"] = [
                                tool["name"] for tool in manifest["tools"]
                            ]
                        return manifest
        except Exception as e:
            logger.warning(f"Failed to discover manifest for {server_name}: {e}")
        
        return None
    
    async def start_stdio_server(self, server_name: str) -> bool:
        """Lance un serveur MCP en mode STDIO"""
        if server_name not in self.servers:
            return False
        
        server = self.servers[server_name]
        if server.get("type") != "stdio":
            return False
        
        if server_name in self.stdio_processes:
            # Déjà lancé
            return True
        
        try:
            command = server["command"]
            args = server.get("args", [])
            env = {**os.environ, **server.get("env", {})}
            
            process = subprocess.Popen(
                [command] + args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True
            )
            
            self.stdio_processes[server_name] = process
            logger.info(f"Started STDIO MCP server: {server_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start STDIO server {server_name}: {e}")
            return False
    
    async def tools_call(self, server_name: str, tool_name: str, arguments: Dict) -> Dict:
        """Appel générique d'un outil MCP (HTTP ou STDIO)"""
        if server_name not in self.servers:
            raise ValueError(f"Unknown MCP server '{server_name}'")
        
        server = self.servers[server_name]
        server_type = server.get("type", "http")
        
        if server_type == "http":
            return await self._call_http_tool(server, tool_name, arguments)
        elif server_type == "stdio":
            return await self._call_stdio_tool(server_name, tool_name, arguments)
        else:
            raise ValueError(f"Unsupported MCP server type: {server_type}")
    
    async def _call_http_tool(self, server: Dict, tool_name: str, arguments: Dict) -> Dict:
        """Appel d'un outil via HTTP/JSON-RPC 2.0"""
        url = f"{server['url'].rstrip('/')}/mcp"
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments}
        }
        
        headers = {"Content-Type": "application/json"}
        
        # Authentification si nécessaire
        auth = server.get("auth", {})
        if auth.get("type") == "bearer":
            headers["Authorization"] = f"Bearer {auth['token']}"
        elif auth.get("type") == "header":
            header_name = auth.get("header")
            header_value = expand_env(auth.get("value", ""))
            if header_name and header_value:
                headers[header_name] = header_value
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers, timeout=30) as response:
                    response.raise_for_status()
                    result = await response.json()
                    
                    if "error" in result:
                        return {"error": result["error"]}
                    
                    return result.get("result", result)
                    
        except Exception as e:
            logger.error(f"HTTP MCP call failed: {e}")
            return {"error": f"HTTP call failed: {str(e)}"}
    
    async def _call_stdio_tool(self, server_name: str, tool_name: str, arguments: Dict) -> Dict:
        """Appel d'un outil via STDIO/JSON-RPC 2.0"""
        # S'assurer que le processus est lancé
        if not await self.start_stdio_server(server_name):
            return {"error": f"Failed to start STDIO server {server_name}"}
        
        process = self.stdio_processes[server_name]
        
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments}
        }
        
        try:
            # Envoyer la requête
            request_line = json.dumps(payload) + "\n"
            process.stdin.write(request_line)
            process.stdin.flush()
            
            # Lire la réponse avec timeout
            response_line = await asyncio.wait_for(
                asyncio.to_thread(process.stdout.readline),
                timeout=30.0
            )
            
            if not response_line:
                return {"error": "No response from STDIO server"}
            
            result = json.loads(response_line.strip())
            
            if "error" in result:
                return {"error": result["error"]}
            
            return result.get("result", result)
            
        except asyncio.TimeoutError:
            return {"error": "STDIO call timeout"}
        except Exception as e:
            logger.error(f"STDIO MCP call failed: {e}")
            return {"error": f"STDIO call failed: {str(e)}"}
    
    async def health_check(self) -> Dict[str, str]:
        """Vérifie l'état de tous les serveurs MCP"""
        health = {}
        
        for server_name, server in self.servers.items():
            if not server.get("enabled", True):
                health[server_name] = "disabled"
                continue
            
            server_type = server.get("type", "http")
            
            if server_type == "http":
                try:
                    async with aiohttp.ClientSession() as session:
                        health_url = f"{server['url'].rstrip('/')}/health"
                        async with session.get(health_url, timeout=5) as response:
                            if response.status == 200:
                                health[server_name] = "healthy"
                            else:
                                health[server_name] = f"unhealthy-{response.status}"
                except Exception:
                    health[server_name] = "unreachable"
            
            elif server_type == "stdio":
                if server_name in self.stdio_processes:
                    process = self.stdio_processes[server_name]
                    if process.poll() is None:
                        health[server_name] = "running"
                    else:
                        health[server_name] = "crashed"
                else:
                    health[server_name] = "not_started"
        
        return health
    
    async def shutdown(self):
        """Arrête tous les processus STDIO"""
        for server_name, process in self.stdio_processes.items():
            try:
                process.terminate()
                await asyncio.wait_for(
                    asyncio.to_thread(process.wait),
                    timeout=5.0
                )
                logger.info(f"Stopped STDIO server: {server_name}")
            except Exception as e:
                logger.warning(f"Failed to stop {server_name}: {e}")
                process.kill()
        
        self.stdio_processes.clear()
    
    def __del__(self):
        """Nettoyage automatique"""
        if self.stdio_processes:
            for process in self.stdio_processes.values():
                try:
                    process.terminate()
                except:
                    pass