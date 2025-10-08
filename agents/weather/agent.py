"""
Weather MCP Agent - Version cloud deployée avec moteur sémantique et client MCP
"""
import os
import asyncio
import json
import uuid
from typing import Dict, Optional
import requests
import sys
import httpx
import aiohttp

# Ajouter le path shared
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))

from base_agent import CloudMCPAgentBase
from mcp_client import ExternalMCPClient
import structlog

logger = structlog.get_logger()

# Prompt système pour le moteur sémantique météo
WEATHER_SYS_PROMPT = """You are Weather Intelligence Agent.
Goal: analyze weather data and provide accurate, contextual, actionable weather insights.
When summarizing weather conditions, include practical advice (clothing, activities, travel), 
cite specific measurements (temperature, precipitation, wind), and highlight any weather alerts or unusual patterns.
Prefer bullet points for complex forecasts; avoid speculation about long-term climate.
If data is incomplete, say so explicitly. Always prioritize user safety for severe weather.
Return French text if the user's request is in French; else, English."""

class SemanticHelper:
    """Moteur sémantique pour enrichissement LLM"""
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.key = os.getenv("OPENAI_API_KEY")

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        if not self.key:
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
            return user_prompt

# ExternalMCPClient importé depuis shared/mcp_client.py

# Manifeste MCP pour découverte
AGENT_MANIFEST = {
    "name": "weather_agent",
    "version": "1.0.0", 
    "description": "Weather Intelligence Agent (current weather, forecasts, climate analysis)",
    "tools": [
        {
            "name": "process",
            "description": "Semantic processing of a weather-related request",
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

class WeatherMCPAgent(CloudMCPAgentBase):
    """Agent MCP météo déployé en cloud avec intelligence sémantique"""
    
    def __init__(self):
        super().__init__(
            agent_id="weather_agent",
            name="Weather Intelligence Agent", 
            capabilities=["weather", "forecast", "climate_analysis"]
        )
        
        # Moteur sémantique LLM
        self.semantic = SemanticHelper(model=os.getenv("LLM_MODEL", "gpt-4o-mini"))
        
        # Client MCP configuré via mcp.json
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'mcp.json')
        self.mcp_client = ExternalMCPClient(config_path)
        
        # Configuration OpenMeteo
        self.openmeteo_api_key = os.getenv("OPENMETEO_API_KEY")
        self.base_url = "https://api.open-meteo.com/v1/forecast"
        self.geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
    
    async def process(self, request: str, context: Dict) -> Dict:
        """Traitement sémantique des requêtes météo"""
        try:
            # 1. Analyse sémantique des intentions
            semantic_intent = self.analyze_weather_intent(request)
            
            # 2. Appel API météo selon l'intention
            if semantic_intent["type"] == "forecast":
                weather_data = await self.get_weather_forecast(
                    semantic_intent["location"],
                    days=7
                )
            else:
                weather_data = await self.get_current_weather(
                    semantic_intent["location"]
                )
            
            # 3. Enrichissement sémantique de la réponse
            if "error" not in weather_data:
                enriched_response = await self.enrich_weather_response(
                    weather_data, request, semantic_intent
                )
                
                return {
                    "agent": self.agent_id,
                    "semantic_intent": semantic_intent,
                    "response": enriched_response,
                    "confidence": 0.95,
                    "data": weather_data,
                    "source": "Open-Meteo API"
                }
            else:
                return {
                    "agent": self.agent_id,
                    "response": f"Unable to fetch weather data for {semantic_intent['location']}: {weather_data.get('error')}",
                    "confidence": 0.0,
                    "error": weather_data.get("error")
                }
                
        except Exception as e:
            logger.error(f"Error processing weather request: {e}")
            return {
                "agent": self.agent_id,
                "response": f"Error processing weather request: {str(e)}",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def analyze_weather_intent(self, request: str) -> Dict:
        """Analyse sémantique des intentions météorologiques"""
        intent = {
            "location": "Paris",  # Default
            "type": "current",
            "specific_needs": [],
            "urgency": "medium",
            "temporal_context": "now"
        }
        
        request_lower = request.lower()
        
        # Extraction sémantique de la localisation
        cities = ["paris", "london", "tokyo", "new york", "sydney", "berlin", "madrid", "amsterdam"]
        for city in cities:
            if city in request_lower:
                intent["location"] = city.title()
                break
        
        # Analyse des patterns temporels
        forecast_patterns = ["tomorrow", "demain", "forecast", "prévision", "week", "semaine", "days", "jours"]
        if any(pattern in request_lower for pattern in forecast_patterns):
            intent["type"] = "forecast"
            intent["temporal_context"] = "future"
        
        # Détection des besoins spécifiques
        if "temperature" in request_lower or "temp" in request_lower:
            intent["specific_needs"].append("temperature")
        if "rain" in request_lower or "precipitation" in request_lower or "pluie" in request_lower:
            intent["specific_needs"].append("precipitation")
        if "wind" in request_lower or "vent" in request_lower:
            intent["specific_needs"].append("wind")
        
        # Détection d'urgence contextuelle
        urgency_keywords = ["urgent", "now", "quickly", "asap", "maintenant"]
        if any(keyword in request_lower for keyword in urgency_keywords):
            intent["urgency"] = "high"
            
        return intent
    
    async def get_coordinates(self, city: str) -> Optional[Dict]:
        """Géocodage via OpenMeteo"""
        try:
            params = {"name": city, "count": 1}
            response = requests.get(self.geocoding_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("results"):
                    result = data["results"][0]
                    return {
                        "lat": result["latitude"],
                        "lon": result["longitude"],
                        "country": result.get("country", "")
                    }
            return None
        except Exception as e:
            logger.error(f"Geocoding error: {e}")
            return None
    
    async def get_current_weather(self, location: str) -> Dict:
        """Obtenir la météo actuelle"""
        try:
            coords = await self.get_coordinates(location)
            if not coords:
                return {"error": f"Location {location} not found"}
            
            params = {
                "latitude": coords["lat"],
                "longitude": coords["lon"],
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                "timezone": "auto"
            }
            
            response = requests.get(self.base_url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                current = data.get("current", {})
                
                return {
                    "city": location,
                    "coordinates": coords,
                    "temperature": current.get("temperature_2m"),
                    "humidity": current.get("relative_humidity_2m"),
                    "wind_speed": current.get("wind_speed_10m"),
                    "weather_code": current.get("weather_code"),
                    "description": self.weather_code_to_text(current.get("weather_code", 0)),
                    "timestamp": current.get("time"),
                    "source": "Open-Meteo API"
                }
            else:
                return {"error": f"Weather API returned {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Weather API error: {e}")
            return {"error": f"Failed to fetch weather: {str(e)}"}
    
    async def get_weather_forecast(self, location: str, days: int = 7) -> Dict:
        """Obtenir les prévisions météo"""
        try:
            coords = await self.get_coordinates(location)
            if not coords:
                return {"error": f"Location {location} not found"}
            
            params = {
                "latitude": coords["lat"],
                "longitude": coords["lon"],
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                "timezone": "auto",
                "forecast_days": days
            }
            
            response = requests.get(self.base_url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                daily = data.get("daily", {})
                
                forecast = []
                for i in range(len(daily.get("time", []))):
                    forecast.append({
                        "date": daily["time"][i],
                        "temp_max": daily["temperature_2m_max"][i],
                        "temp_min": daily["temperature_2m_min"][i],
                        "precipitation": daily["precipitation_sum"][i],
                        "description": self.weather_code_to_text(daily["weather_code"][i])
                    })
                
                return {
                    "city": location,
                    "forecast": forecast,
                    "source": "Open-Meteo API"
                }
            else:
                return {"error": f"Forecast API returned {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Forecast API error: {e}")
            return {"error": f"Failed to fetch forecast: {str(e)}"}
    
    def weather_code_to_text(self, code: int) -> str:
        """Conversion des codes météo OpenMeteo en texte"""
        weather_codes = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
            55: "Dense drizzle", 56: "Light freezing drizzle", 57: "Dense freezing drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            66: "Light freezing rain", 67: "Heavy freezing rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            77: "Snow grains", 80: "Slight rain showers", 81: "Moderate rain showers",
            82: "Violent rain showers", 85: "Slight snow showers", 86: "Heavy snow showers",
            95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
        }
        return weather_codes.get(code, "Unknown weather condition")
    
    async def on_startup(self):
        """Chargement de la configuration MCP"""
        await self.mcp_client.load_config()
        logger.info(f"Weather agent initialized with {len(self.mcp_client.servers)} MCP servers")

    async def get_current_weather(self, location: str) -> Dict:
        """Obtenir la météo actuelle + enrichissements MCP"""
        try:
            coords = await self.get_coordinates(location)
            if not coords:
                return {"error": f"Location {location} not found"}
            
            params = {
                "latitude": coords["lat"],
                "longitude": coords["lon"],
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                "timezone": "auto"
            }
            
            response = requests.get(self.base_url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                current = data.get("current", {})
                
                base_weather = {
                    "city": location,
                    "coordinates": coords,
                    "temperature": current.get("temperature_2m"),
                    "humidity": current.get("relative_humidity_2m"),
                    "wind_speed": current.get("wind_speed_10m"),
                    "weather_code": current.get("weather_code"),
                    "description": self.weather_code_to_text(current.get("weather_code", 0)),
                    "timestamp": current.get("time"),
                    "source": "Open-Meteo API"
                }
                
                # Enrichissement par MCP externes
                
                # Enrichissement par catégorie MCP  
                weather_servers = self.mcp_client.get_servers_by_category("weather")
                
                # Vérifier les alertes météo
                if "weather-alerts" in weather_servers:
                    try:
                        alerts = await self.mcp_client.tools_call(
                            "weather-alerts", "get_alerts",
                            {"latitude": coords["lat"], "longitude": coords["lon"]}
                        )
                        base_weather["alerts"] = alerts.get("result", alerts)
                    except Exception as e:
                        base_weather.setdefault("external_errors", []).append(f"weather-alerts: {e}")

                # Contexte historique si conditions extrêmes
                temp = base_weather.get("temperature", 0)
                if temp < -10 or temp > 35:  # Conditions extrêmes
                    if "weather-historical" in weather_servers:
                        try:
                            historical = await self.mcp_client.tools_call(
                                "weather-historical", "get_temperature_context",
                                {"location": location, "current_temp": temp}
                            )
                            base_weather["historical_context"] = historical.get("result", historical)
                        except Exception as e:
                            base_weather.setdefault("external_errors", []).append(f"weather-historical: {e}")

                return base_weather
            else:
                return {"error": f"Weather API returned {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Weather API error: {e}")
            return {"error": f"Failed to fetch weather: {str(e)}"}

    async def enrich_weather_response(self, data: Dict, original_request: str, intent: Dict) -> str:
        """Enrichissement sémantique contextuel avec LLM"""
        # 1. Synthèse déterministe de base
        intent_type = intent.get("type", "current")
        location = intent.get("location", "unknown")
        
        if intent_type == "forecast":
            forecast_days = len(data.get("forecast", []))
            base = f"Prévisions météo pour {location}: {forecast_days} jours"
            
            if data.get("forecast"):
                temps = [day.get("temp_max", 0) for day in data["forecast"][:3]]
                if len(temps) >= 2:
                    if temps[1] > temps[0]:
                        base += " - Tendance réchauffement"
                    elif temps[1] < temps[0]:
                        base += " - Tendance refroidissement"
        else:
            temp = data.get("temperature")
            desc = data.get("description", "")
            base = f"Météo actuelle à {location}: {temp}°C, {desc}"
            
            # Mention des enrichissements MCP
            if data.get("alerts"):
                base += " - Alertes disponibles"
            if data.get("historical_context"):
                base += " - Contexte historique inclus"
            if data.get("external_errors"):
                base += f" - Erreurs MCP: {len(data.get('external_errors', []))}"

        # 2. Enrichissement sémantique via LLM
        user_prompt = (
            f"User request:\n{original_request}\n\n"
            f"Weather data (JSON):\n{json.dumps(data, ensure_ascii=False)[:8000]}\n\n"
            "Provide practical weather advice with safety recommendations, clothing suggestions, and activity planning."
        )
        
        try:
            pretty = await self.semantic.complete(WEATHER_SYS_PROMPT, user_prompt)
            return pretty
        except Exception as e:
            logger.warning(f"Semantic enrichment failed: {e}")
            return base
    
    async def check_health(self) -> Dict:
        """Vérification de santé spécifique à l'agent météo"""
        health = {
            "openmeteo_api": "unknown",
            "semantic_engine": "unknown", 
            "mcp_clients": {}
        }
        
        try:
            # Test simple de l'API OpenMeteo
            response = requests.get(
                "https://api.open-meteo.com/v1/forecast?latitude=48.8566&longitude=2.3522&current=temperature_2m",
                timeout=5
            )
            if response.status_code == 200:
                health["openmeteo_api"] = "healthy"
            else:
                health["openmeteo_api"] = f"unhealthy - {response.status_code}"
                
            # Test du moteur sémantique
            if self.semantic.key:
                health["semantic_engine"] = "available"
            else:
                health["semantic_engine"] = "no_api_key"
            
            # État des clients MCP
            health["mcp_clients"] = await self.mcp_client.health_check()
                
        except Exception as e:
            health["openmeteo_api"] = f"error - {str(e)}"
        
        return health
    
    def get_manifest(self) -> Dict:
        """Retourne le manifeste MCP pour découverte"""
        return AGENT_MANIFEST