"""
Finance MCP Agent - Version cloud deployée avec moteur sémantique et client MCP
"""
import os
import json
import uuid
from typing import Dict, List
import requests
import sys
from datetime import datetime, timedelta
import httpx
import aiohttp

# Ajouter le path shared
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))

from base_agent import CloudMCPAgentBase
from mcp_client import ExternalMCPClient
import structlog

logger = structlog.get_logger()

# Prompt système pour le moteur sémantique finance
FINANCE_SYS_PROMPT = """You are Finance Intelligence Agent.
Goal: analyze financial data and provide accurate, actionable investment insights.
When summarizing market data, include risk assessment, valuation metrics, and trend analysis.
Cite specific numbers (prices, P/E ratios, volumes) and highlight any unusual market patterns.
Prefer bullet points for portfolio analysis; avoid speculation about future price movements.
If data is incomplete, say so explicitly. Always prioritize risk disclosure for investment decisions.
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
    "name": "finance_agent",
    "version": "1.0.0",
    "description": "Finance Intelligence Agent (stocks, markets, portfolio analysis)",
    "tools": [
        {
            "name": "process",
            "description": "Semantic processing of a finance-related request",
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

# Import conditionnel de yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not available, using simulation mode")

class FinanceMCPAgent(CloudMCPAgentBase):
    """Agent MCP finance déployé en cloud avec analyse sémantique"""
    
    def __init__(self):
        super().__init__(
            agent_id="finance_agent",
            name="Financial Intelligence Agent",
            capabilities=["stocks", "markets", "portfolio_analysis"]
        )
        
        # Moteur sémantique LLM
        self.semantic = SemanticHelper(model=os.getenv("LLM_MODEL", "gpt-4o-mini"))
        
        # Client MCP configuré via mcp.json
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'mcp.json')
        self.mcp_client = ExternalMCPClient(config_path)
        
        # Configuration APIs financières
        self.yahoo_finance_key = os.getenv("YAHOO_FINANCE_KEY")
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY")
    
    async def process(self, request: str, context: Dict) -> Dict:
        """Traitement sémantique des requêtes financières"""
        try:
            # 1. Analyse sémantique financière
            financial_intent = self.analyze_financial_intent(request)
            
            # 2. Routage intelligent vers les bonnes méthodes
            if financial_intent["analysis_type"] == "historical":
                market_data = await self.get_historical_data(
                    financial_intent["symbol"],
                    period="1mo"
                )
            elif financial_intent["analysis_type"] == "portfolio":
                market_data = await self.analyze_portfolio(
                    financial_intent["symbols"]
                )
            else:
                market_data = await self.get_stock_data(
                    financial_intent["symbol"]
                )
            
            # 3. Enrichissement sémantique financier
            if "error" not in market_data:
                semantic_response = await self.enrich_financial_response(
                    market_data, request, financial_intent
                )
                
                return {
                    "agent": self.agent_id,
                    "financial_intent": financial_intent,
                    "response": semantic_response,
                    "confidence": 0.92,
                    "data": market_data,
                    "source": "Yahoo Finance" if YFINANCE_AVAILABLE else "Simulation"
                }
            else:
                return {
                    "agent": self.agent_id,
                    "response": f"Unable to fetch financial data: {market_data.get('error')}",
                    "confidence": 0.0,
                    "error": market_data.get("error")
                }
                
        except Exception as e:
            logger.error(f"Error processing finance request: {e}")
            return {
                "agent": self.agent_id,
                "response": f"Error processing financial request: {str(e)}",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def analyze_financial_intent(self, request: str) -> Dict:
        """Analyse sémantique des intentions financières"""
        intent = {
            "symbol": "AAPL",  # Default
            "analysis_type": "current",
            "sector": "technology",
            "metric": "price",
            "sentiment": "neutral",
            "symbols": []
        }
        
        request_lower = request.lower()
        
        # Extraction sémantique des symboles boursiers
        symbol_mapping = {
            "apple": "AAPL", "aapl": "AAPL",
            "google": "GOOGL", "googl": "GOOGL", "alphabet": "GOOGL",
            "microsoft": "MSFT", "msft": "MSFT",
            "nvidia": "NVDA", "nvda": "NVDA",
            "tesla": "TSLA", "tsla": "TSLA",
            "amazon": "AMZN", "amzn": "AMZN",
            "meta": "META", "facebook": "META",
            "netflix": "NFLX", "nflx": "NFLX"
        }
        
        found_symbols = []
        for key, value in symbol_mapping.items():
            if key in request_lower and value not in found_symbols:
                found_symbols.append(value)
        
        if found_symbols:
            intent["symbol"] = found_symbols[0]
            intent["symbols"] = found_symbols
        
        # Détection du type d'analyse
        if any(word in request_lower for word in ["history", "historical", "trend", "past", "evolution"]):
            intent["analysis_type"] = "historical"
        elif any(word in request_lower for word in ["portfolio", "multiple", "compare", "comparison"]):
            intent["analysis_type"] = "portfolio"
        elif len(found_symbols) > 1:
            intent["analysis_type"] = "portfolio"
        
        # Analyse du sentiment sémantique
        bullish_words = ["up", "rise", "bull", "bullish", "positive", "growth", "gain", "increase"]
        bearish_words = ["down", "fall", "bear", "bearish", "negative", "decline", "loss", "decrease"]
        
        if any(word in request_lower for word in bullish_words):
            intent["sentiment"] = "bullish"
        elif any(word in request_lower for word in bearish_words):
            intent["sentiment"] = "bearish"
        
        # Détection de secteur
        if any(word in request_lower for word in ["tech", "technology", "software"]):
            intent["sector"] = "technology"
        elif any(word in request_lower for word in ["energy", "oil", "gas"]):
            intent["sector"] = "energy"
        elif any(word in request_lower for word in ["finance", "bank", "financial"]):
            intent["sector"] = "finance"
            
        return intent
    
    async def on_startup(self):
        """Chargement de la configuration MCP"""
        await self.mcp_client.load_config()
        logger.info(f"Finance agent initialized with {len(self.mcp_client.servers)} MCP servers")

    async def get_stock_data(self, symbol: str) -> Dict:
        """Obtenir les données actuelles d'une action + enrichissements MCP"""
        try:
            # 1. Données de base Yahoo Finance
            base_data = {}
            if YFINANCE_AVAILABLE:
                stock = yf.Ticker(symbol)
                info = stock.info
                
                if not info or info.get('regularMarketPrice') is None:
                    return {"error": f"No data found for symbol {symbol}"}
                
                base_data = {
                    "symbol": symbol,
                    "name": info.get("longName", symbol),
                    "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
                    "previous_close": info.get("previousClose"),
                    "open": info.get("open") or info.get("regularMarketOpen"),
                    "volume": info.get("volume") or info.get("regularMarketVolume"),
                    "market_cap": info.get("marketCap"),
                    "pe_ratio": info.get("trailingPE"),
                    "dividend_yield": info.get("dividendYield"),
                    "source": "Yahoo Finance",
                    "timestamp": datetime.now().isoformat()
                }
                
                # Calcul du changement
                if base_data["current_price"] and base_data["previous_close"]:
                    change = base_data["current_price"] - base_data["previous_close"]
                    change_pct = (change / base_data["previous_close"]) * 100
                    base_data["change"] = round(change, 2)
                    base_data["change_percent"] = round(change_pct, 2)
            else:
                base_data = self._simulate_stock_data(symbol)

            # 2. Enrichissement par MCP externes
            
            # Enrichissement par catégorie MCP
            finance_servers = self.mcp_client.get_servers_by_category("finance")
            
            # Analyse technique si mouvement significatif
            if abs(base_data.get("change_percent", 0)) > 5:  # Mouvement > 5%
                if "technical-analysis" in finance_servers:
                    try:
                        ta_data = await self.mcp_client.tools_call(
                            "technical-analysis", "analyze_movement",
                            {"symbol": symbol, "price_change": base_data.get("change_percent", 0)}
                        )
                        base_data["technical_analysis"] = ta_data.get("result", ta_data)
                    except Exception as e:
                        base_data.setdefault("external_errors", []).append(f"technical-analysis: {e}")

            # Données fondamentales pour valorisation
            if "fundamentals" in finance_servers:
                try:
                    fundamentals = await self.mcp_client.tools_call(
                        "fundamentals", "get_ratios",
                        {"symbol": symbol}
                    )
                    base_data["fundamentals"] = fundamentals.get("result", fundamentals)
                except Exception as e:
                    base_data.setdefault("external_errors", []).append(f"fundamentals: {e}")

            return base_data
                
        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {e}")
            return {"error": f"Failed to fetch data for {symbol}: {str(e)}"}
    
    async def get_historical_data(self, symbol: str, period: str = "1mo") -> Dict:
        """Obtenir l'historique des prix"""
        try:
            if YFINANCE_AVAILABLE:
                stock = yf.Ticker(symbol)
                history = stock.history(period=period, interval="1d")
                
                if history.empty:
                    return {"error": f"No historical data for {symbol}"}
                
                # Convertir en format JSON-friendly
                data = []
                for date, row in history.iterrows():
                    data.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "open": round(row["Open"], 2),
                        "high": round(row["High"], 2),
                        "low": round(row["Low"], 2),
                        "close": round(row["Close"], 2),
                        "volume": int(row["Volume"])
                    })
                
                # Calculer des métriques
                import numpy as np
                prices = history['Close'].values
                returns = np.diff(prices) / prices[:-1] * 100
                
                result = {
                    "symbol": symbol,
                    "period": period,
                    "data_points": len(data),
                    "history": data[-10:],  # Derniers 10 points
                    "stats": {
                        "avg_return": round(np.mean(returns), 3),
                        "volatility": round(np.std(returns), 3),
                        "max_price": round(prices.max(), 2),
                        "min_price": round(prices.min(), 2),
                        "price_change": round(((prices[-1] - prices[0]) / prices[0]) * 100, 2)
                    },
                    "source": "Yahoo Finance"
                }
                
                return result
            else:
                return self._simulate_historical_data(symbol, period)
                
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return {"error": f"Failed to fetch historical data: {str(e)}"}
    
    async def analyze_portfolio(self, symbols: List[str]) -> Dict:
        """Analyser un portfolio d'actions"""
        try:
            portfolio = []
            total_value = 0
            
            for symbol in symbols:
                stock_data = await self.get_stock_data(symbol)
                if "error" not in stock_data:
                    portfolio.append({
                        "symbol": symbol,
                        "price": stock_data.get("current_price", 0),
                        "name": stock_data.get("name", symbol),
                        "change_percent": stock_data.get("change_percent", 0)
                    })
                    total_value += stock_data.get("current_price", 0)
            
            if not portfolio:
                return {"error": "No valid stocks found in portfolio"}
            
            # Analyse du portfolio
            positive_stocks = [s for s in portfolio if s["change_percent"] > 0]
            negative_stocks = [s for s in portfolio if s["change_percent"] < 0]
            
            result = {
                "portfolio": portfolio,
                "total_symbols": len(portfolio),
                "total_value": round(total_value, 2),
                "performance": {
                    "positive_stocks": len(positive_stocks),
                    "negative_stocks": len(negative_stocks),
                    "avg_change": round(sum(s["change_percent"] for s in portfolio) / len(portfolio), 2)
                },
                "timestamp": datetime.now().isoformat(),
                "source": "Yahoo Finance" if YFINANCE_AVAILABLE else "Simulation"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio: {e}")
            return {"error": f"Portfolio analysis failed: {str(e)}"}
    
    def _simulate_stock_data(self, symbol: str) -> Dict:
        """Simulation de données boursières"""
        import random
        
        base_prices = {
            "AAPL": 175.0, "GOOGL": 140.0, "MSFT": 380.0,
            "NVDA": 450.0, "TSLA": 250.0, "AMZN": 145.0
        }
        
        base_price = base_prices.get(symbol, 100.0)
        change_pct = random.uniform(-3.0, 3.0)
        current_price = base_price * (1 + change_pct / 100)
        
        return {
            "symbol": symbol,
            "name": f"{symbol} Inc",
            "current_price": round(current_price, 2),
            "previous_close": base_price,
            "change_percent": round(change_pct, 2),
            "volume": random.randint(1000000, 50000000),
            "source": "Simulated"
        }
    
    def _simulate_historical_data(self, symbol: str, period: str) -> Dict:
        """Simulation de données historiques"""
        import random
        
        days = {"1mo": 30, "3mo": 90, "1y": 365}.get(period, 30)
        base_price = 150.0
        
        data = []
        for i in range(days):
            change = random.uniform(-0.05, 0.05)
            base_price *= (1 + change)
            data.append({
                "date": (datetime.now() - timedelta(days=days-i)).strftime("%Y-%m-%d"),
                "close": round(base_price, 2)
            })
        
        return {
            "symbol": symbol,
            "period": period,
            "data_points": len(data),
            "history": data[-10:],
            "stats": {
                "volatility": round(random.uniform(1.0, 8.0), 2),
                "avg_return": round(random.uniform(-2.0, 3.0), 2)
            },
            "source": "Simulated"
        }
    
    async def enrich_financial_response(self, data: Dict, original_request: str, intent: Dict) -> str:
        """Enrichissement sémantique contextuel financier avec LLM"""
        # 1. Synthèse déterministe de base
        analysis_type = intent.get("analysis_type", "current")
        
        if analysis_type == "portfolio":
            total_symbols = data.get("total_symbols", 0)
            total_value = data.get("total_value", 0)
            performance = data.get("performance", {})
            avg_change = performance.get("avg_change", 0)
            base = f"Analyse portfolio: {total_symbols} actions, valeur ${total_value:,.2f}, performance {avg_change:+.2f}%"
            
        elif analysis_type == "historical":
            symbol = data.get("symbol")
            stats = data.get("stats", {})
            volatility = stats.get("volatility", 0)
            price_change = stats.get("price_change", 0)
            base = f"Analyse historique {symbol}: évolution {price_change:+.1f}%, volatilité {volatility:.1f}%"
            
        else:
            symbol = data.get("symbol")
            name = data.get("name", symbol)
            price = data.get("current_price")
            change_percent = data.get("change_percent", 0)
            base = f"{name} ({symbol}): ${price:.2f} ({change_percent:+.2f}%)"
            
            # Mention des enrichissements MCP
            if data.get("technical_analysis"):
                base += " - Analyse technique disponible"
            if data.get("fundamentals"):
                base += " - Données fondamentales incluses"
            if data.get("external_errors"):
                base += f" - Erreurs MCP: {len(data.get('external_errors', []))}"

        # 2. Enrichissement sémantique via LLM
        user_prompt = (
            f"User request:\n{original_request}\n\n"
            f"Financial data (JSON):\n{json.dumps(data, ensure_ascii=False)[:8000]}\n\n"
            "Provide investment analysis with risk assessment, valuation metrics, and actionable recommendations."
        )
        
        try:
            pretty = await self.semantic.complete(FINANCE_SYS_PROMPT, user_prompt)
            return pretty
        except Exception as e:
            logger.warning(f"Semantic enrichment failed: {e}")
            return base
    
    async def check_health(self) -> Dict:
        """Vérification de santé spécifique à l'agent finance"""
        health = {
            "yfinance": "unknown", 
            "yahoo_api": "unknown",
            "semantic_engine": "unknown",
            "mcp_clients": {}
        }
        
        if YFINANCE_AVAILABLE:
            health["yfinance"] = "available"
            
            try:
                # Test simple avec Apple
                stock = yf.Ticker("AAPL")
                info = stock.info
                if info and info.get("regularMarketPrice"):
                    health["yahoo_api"] = "healthy"
                else:
                    health["yahoo_api"] = "no_data"
            except Exception as e:
                health["yahoo_api"] = f"error - {str(e)[:50]}"
        else:
            health["yfinance"] = "not_available"
            health["yahoo_api"] = "simulation_mode"
        
        # Test du moteur sémantique
        if self.semantic.key:
            health["semantic_engine"] = "available"
        else:
            health["semantic_engine"] = "no_api_key"
        
        # État des clients MCP
        health["mcp_clients"] = await self.mcp_client.health_check()
        
        return health
    
    def get_manifest(self) -> Dict:
        """Retourne le manifeste MCP pour découverte"""
        return AGENT_MANIFEST