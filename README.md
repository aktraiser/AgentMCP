# 🚀 MCP Agents Cloud Deployment

Déploiement serverless des agents MCP sur Vercel avec GitHub Actions automatiques.

## 📁 Structure du Projet

```
mcp-agents-deployment/
├── README.md
├── vercel.json                 # Configuration Vercel
├── requirements.txt            # Dépendances Python
├── agents/
│   ├── weather/
│   │   ├── handler.py         # Handler Vercel pour Weather Agent
│   │   ├── agent.py           # Code de l'agent météo
│   │   └── Dockerfile         # Container pour déploiement alternatif
│   ├── finance/
│   │   ├── handler.py         # Handler Vercel pour Finance Agent
│   │   ├── agent.py           # Code de l'agent finance
│   │   └── Dockerfile
│   ├── code/
│   │   ├── handler.py         # Handler Vercel pour Code Agent
│   │   ├── agent.py           # Code de l'agent d'analyse
│   │   └── Dockerfile
│   └── master/
│       ├── handler.py         # Handler Vercel pour Master Agent
│       ├── orchestrator.py    # Orchestrateur cloud-natif
│       └── Dockerfile
├── shared/
│   ├── __init__.py
│   ├── base_agent.py          # Classe de base pour tous les agents
│   ├── registry.py            # Registry cloud pour découverte d'agents
│   └── auth.py                # Middleware d'authentification
└── deploy/
    ├── docker-compose.yml     # Déploiement Docker local
    ├── k8s/                   # Manifests Kubernetes
    └── scripts/               # Scripts de déploiement
```

## 🎯 Déploiement Rapide

### 1. Vercel (Recommandé)
```bash
# Clone et setup
git clone <your-repo>
cd mcp-agents-deployment
pip install -r requirements.txt

# Déployer sur Vercel
vercel deploy
```

### 2. Docker Local
```bash
docker-compose up -d
```

### 3. Kubernetes
```bash
kubectl apply -f deploy/k8s/
```

## 🔐 Variables d'Environnement

Créer un fichier `.env` :
```env
# API Keys
OPENMETEO_API_KEY=your_key_here
YAHOO_FINANCE_KEY=your_key_here
GITHUB_TOKEN=your_token_here

# Agent Authentication
WEATHER_AGENT_KEY=secure_key_1
FINANCE_AGENT_KEY=secure_key_2
CODE_AGENT_KEY=secure_key_3
MASTER_AGENT_KEY=secure_master_key

# Vercel URLs (après déploiement)
WEATHER_AGENT_URL=https://your-project.vercel.app/weather
FINANCE_AGENT_URL=https://your-project.vercel.app/finance
CODE_AGENT_URL=https://your-project.vercel.app/code
```

## 📊 Endpoints Déployés

| Agent | Endpoint | Fonctionnalités |
|-------|----------|-----------------|
| Master | `/master/*` | Orchestration, découverte, synthèse |
| Weather | `/weather/*` | Météo actuelle, prévisions |
| Finance | `/finance/*` | Prix actions, analyse portfolio |
| Code | `/code/*` | Analyse code, debug, suggestions |

## 🔧 Test des Agents

```bash
# Test Master Agent
curl -X POST https://your-project.vercel.app/master/orchestrate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-master-key" \
  -d '{"query": "Weather in Paris and Apple stock price"}'

# Test Weather Agent direct
curl -X POST https://your-project.vercel.app/weather/mcp \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-weather-key" \
  -d '{"jsonrpc": "2.0", "method": "process", "params": {"request": "Weather in Paris"}, "id": 1}'
```

## 💰 Coûts Estimés

| Usage | Vercel Hobby | Vercel Pro |
|-------|--------------|------------|
| Dev/Test | Gratuit | $20/mois |
| Production légère | $20/mois | $20/mois |
| Production intensive | Non recommandé | $20-100/mois |

## 🚀 Next Steps

1. **Configurer les variables d'environnement**
2. **Déployer sur Vercel** 
3. **Tester les endpoints**
4. **Configurer le monitoring**
5. **Optimiser selon l'usage**