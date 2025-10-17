# Agent MCP d'Extraction & Processing de Documents

## 📋 Vue d'ensemble

L'**Agent MCP d'Extraction & Processing** est un agent spécialisé conçu pour l'extraction de documents et leur traitement complet avec sauvegarde vectorielle. Il combine les capacités d'extraction Docling, de chunking intelligent, de génération d'embeddings et de sauvegarde Supabase, tout en offrant des fonctionnalités d'audit pour vérifier la qualité du traitement.

### 🎯 Objectifs

- **Extraction de documents** : Traiter PDF, DOCX, XLSX via Docling et fallbacks
- **Chunking intelligent** : Découpage sémantique optimisé pour les embeddings
- **Génération d'embeddings** : Création d'embeddings OpenAI vectoriels
- **Sauvegarde Supabase** : Stockage documents + chunks avec recherche vectorielle
- **Audit qualité** : Vérification et contrôle de la qualité du traitement
- **Intégration MCP** : Utilisation des services MCP Docling avancés

## 🏗️ Architecture

### Composants principaux

```
┌─────────────────────────────────────────────────────────────┐
│           Agent MCP d'Extraction & Processing               │
├─────────────────────────────────────────────────────────────┤
│  • DocumentAuditMCPAgent (agent.py)                        │
│  • Extraction: MCP Docling → Docling → Context RAG API     │
│  • Chunking: Sémantique + optimisation embeddings         │
│  • Embeddings: OpenAI text-embedding-3-small              │
│  • Storage: Supabase documents + document_chunks           │
│  • CloudMCPAgentBase (classe de base)                      │
│  • FastAPI handlers (endpoints REST)                       │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                Services externes                            │
├─────────────────────────────────────────────────────────────┤
│  • Context RAG API (http://localhost:3000)                 │
│  • Docling traditionnel (http://localhost:8090)            │
│  • MCP Docling (http://localhost:8091)                     │
│  • Supabase (base vectorielle)                             │
│  • MCP Supabase (optionnel)                                │
└─────────────────────────────────────────────────────────────┘
```

### Opérations disponibles

1. **Extraction Simple** (`extract`)
   - Extraction de documents via MCP Docling/Docling/Context RAG API
   - Chunking sémantique du contenu
   - Génération d'embeddings (optionnel)
   - **Pas de sauvegarde** - analyse et rapport seulement

2. **Pipeline Complet** (`process`)
   - **Extraction** → **Chunking** → **Embeddings** → **Sauvegarde Supabase**
   - Insertion dans `documents` table
   - Insertion chunks avec embeddings dans `document_chunks` table
   - Pipeline de production complet

3. **Audit Extraction** (`audit extraction`)
   - Test APIs d'extraction Context RAG
   - Vérification Docling traditionnel et MCP
   - Analyse qualité du contenu extrait

4. **Audit Chunking** (`audit chunking`)
   - Configuration stratégies de chunking
   - Distribution des tailles de chunks
   - Cohérence sémantique et métadonnées

5. **Audit Vectoriel** (`audit vector`)
   - Connexion Supabase et index HNSW
   - Qualité des embeddings
   - Performance recherche vectorielle

6. **Audit Pipeline** (`audit full`)
   - Audit end-to-end du pipeline complet
   - Score global de performance
   - Recommandations d'amélioration

7. **Audit Santé** (`audit health`)
   - État de tous les composants
   - Métriques générales et disponibilité

## ⚙️ Configuration

### Variables d'environnement

```bash
# Application Context RAG
CONTEXT_RAG_BASE_URL=http://localhost:3000
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key

# Docling traditionnel
DOCLING_URL=http://localhost:8090

# MCP Docling (nouveau)
DOCLING_MCP_URL=http://localhost:8091
DOCLING_API_KEY=your_docling_api_key

# MCP Supabase (optionnel)
SUPABASE_MCP_URL=http://localhost:8092

# OpenAI pour enrichissement
OPENAI_API_KEY=your_openai_api_key
LLM_MODEL=gpt-4o-mini

# Context7 MCP
CONTEXT7_API_KEY=ctx7sk-99ae89ad-91ff-47db-8fe1-b0258cc8dcf5
```

### Configuration MCP (mcp.json)

```json
{
  "servers": {
    "docling": {
      "type": "http",
      "url": "${DOCLING_MCP_URL}",
      "enabled": true,
      "description": "Docling MCP Server for document extraction",
      "tools": ["extract_document", "convert_file", "analyze_structure"],
      "categories": ["document_extraction", "content_analysis"],
      "auth": {
        "type": "header",
        "header": "X-API-Key",
        "value": "${DOCLING_API_KEY}"
      }
    },
    "supabase-vector": {
      "type": "http",
      "url": "${SUPABASE_MCP_URL}",
      "enabled": false,
      "description": "Supabase Vector Operations MCP Server",
      "tools": ["search_similarity", "analyze_vectors"],
      "categories": ["vector_operations", "embedding_analysis"]
    }
  },
  "routing": {
    "document_extraction": ["docling"],
    "content_analysis": ["docling"],
    "vector_operations": ["supabase-vector"],
    "embedding_analysis": ["supabase-vector"]
  }
}
```

## 🚀 Utilisation

### Endpoints disponibles

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/mcp` | POST | Point d'entrée MCP principal |
| `/health` | GET | Vérification santé agent |
| `/schema` | GET | Schéma MCP de l'agent |
| `/manifest.json` | GET | Manifeste pour découverte |

### Exemples de requêtes

#### 1. Extraction simple (analyse sans sauvegarde)

```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "process",
    "params": {
      "request": "extract /path/to/document.pdf generate_embeddings=true",
      "context": {}
    },
    "id": "extract-001"
  }'
```

#### 2. Pipeline complet avec sauvegarde Supabase

```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "process",
    "params": {
      "request": "process pipeline /path/to/document.pdf title=MonDocument category=Docs",
      "context": {}
    },
    "id": "process-001"
  }'
```

#### 3. Audit de santé système

```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "process", 
    "params": {
      "request": "audit système health",
      "context": {}
    },
    "id": "audit-001"
  }'
```

#### 4. Audit extraction avec MCP Docling

```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "process",
    "params": {
      "request": "audit extraction docling mcp file_path /path/to/test.pdf",
      "context": {}
    },
    "id": "audit-002"
  }'
```

#### 5. Audit pipeline complet

```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "process", 
    "params": {
      "request": "audit full pipeline document_id 456",
      "context": {}
    },
    "id": "audit-003"
  }'
```

### Analyse des intentions

L'agent analyse automatiquement les requêtes pour déterminer :

- **Type d'opération** : extract, process, audit extraction, audit chunking, audit vector, audit full, audit health
- **Fichier source** : file_path extraction automatique
- **Options** : generate_embeddings, title, category, tags, etc.
- **Portée** : basic, detailed, summary  
- **Focus** : quality, performance, security, consistency

**Exemples de détection** :

```python
"extract document.pdf generate_embeddings=true" → type="extract", file_path="document.pdf", options={"generate_embeddings": True}
"process pipeline document.pdf title=Test" → type="process", file_path="document.pdf", options={"title": "Test"}
"audit extraction docling performance" → type="extraction", focus=["performance"]
"audit chunking détaillé document_id 123" → type="chunking", scope="detailed", document_id="123"
"audit full pipeline qualité" → type="full", focus=["quality"]
```

## 📊 Fonctionnalités avancées

### Pipeline d'Extraction Intelligent

L'agent utilise une cascade d'extraction avec fallback automatique :

1. **MCP Docling** (priorité 1) - Service MCP Docling externe si configuré
2. **Docling Traditionnel** (priorité 2) - API Docling locale si disponible  
3. **Context RAG API** (fallback) - API d'extraction de votre application

### Chunking Sémantique

Implémentation du chunking basée sur votre code `lib/chunking.ts` :

- **Stratégie sémantique** : Respect de la structure des paragraphes
- **Tailles configurables** : 800 caractères par défaut, 100 overlap
- **Optimisation embeddings** : Nettoyage et contextualisation automatique
- **Métadonnées enrichies** : Type, formatage, position, source

### Génération d'Embeddings

Utilisation de l'API OpenAI comme votre application Context RAG :

- **Modèle** : `text-embedding-3-small` (1536 dimensions)
- **Gestion d'erreurs** : Retry et fallback par chunk
- **Optimisation** : Limite 8000 caractères par chunk
- **Statistiques** : Compteurs succès/échec détaillés

### Sauvegarde Supabase Vectorielle

Reproduction exacte du schéma de votre application :

- **Table `documents`** : Métadonnées document, titre, description, tokens
- **Table `document_chunks`** : Chunks avec embeddings et métadonnées
- **Index HNSW** : Recherche vectorielle optimisée `vector_cosine_ops`
- **Cohérence** : Même structure que Context RAG pour compatibilité

### Audit MCP Docling

Fonctionnalités de vérification et audit :

1. **Découverte automatique** des outils MCP disponibles
2. **Tests de performance** avec métriques de temps de réponse
3. **Analyse de qualité** du contenu extrait
4. **Comparaison** entre méthodes d'extraction

```python
# Processus d'audit MCP Docling
async def _audit_via_mcp_docling(self):
    # 1. Vérification disponibilité serveur MCP
    docling_servers = self.mcp_client.get_servers_by_category("document_extraction")
    
    # 2. Test de santé
    health_check = await self.mcp_client.health_check()
    
    # 3. Découverte des outils
    manifest = await self.mcp_client.discover_manifest("docling")
    
    # 4. Tests de performance
    performance = await self._test_mcp_docling_performance()
    
    return audit_result
```

### Analyse de qualité d'extraction

Score calculé sur plusieurs critères :

- **Longueur contenu** (30%) : Vérification extraction non vide
- **Structure détectée** (50%) : Headings, paragraphes, formatage  
- **Absence d'artefacts** (10%) : Pas d'erreurs OCR ou extraction
- **Cohérence textuelle** (10%) : Ratio mots uniques, répétitions

```python
def _analyze_extraction_quality(self, extracted_content: str) -> Dict:
    quality_score = 0.0
    
    # Longueur du contenu
    if len(extracted_content) > 100:
        quality_score += 0.3
    
    # Structure détectée  
    has_headings = any(line.startswith('#') for line in extracted_content.split('\n'))
    if has_headings:
        quality_score += 0.2
    
    # Détection d'artefacts
    artifacts = ['OCR:', '[Image]', 'Error:']
    artifact_count = sum(1 for a in artifacts if a in extracted_content)
    quality_score -= 0.1 * artifact_count
    
    return {"score": max(0.0, min(1.0, quality_score)), ...}
```

### Métriques et rapports

#### Structure des rapports

```markdown
📊 **Rapport d'audit Extraction**

### 📄 Extraction de Documents

- **API Extraction**: available
- **Docling traditionnel**: healthy  
- **MCP Docling**: healthy
  - Outils disponibles: 4
  - Temps de réponse: 0.245s
- **Test MCP extraction**: ✅ Score qualité: 87%

**⚠️ Problèmes détectés** (1):
  - Docling URL non configurée - utilisation fallback uniquement
```

#### Métriques trackées

```python
self.audit_stats = {
    "total_audits": 0,
    "extraction_audits": 0, 
    "chunking_audits": 0,
    "vector_audits": 0,
    "pipeline_operations": 0,          # Nouveaux pipelines traités
    "documents_processed": 0,          # Documents extraits avec succès  
    "chunks_created": 0,               # Chunks générés au total
    "embeddings_generated": 0,         # Embeddings créés avec succès
    "issues_found": 0,
    "average_quality_score": 0.0
}
```

## 🔧 Installation et déploiement

### 1. Installation locale

```bash
# Cloner le repository
git clone <your-repo>
cd mcp-agents-deployment

# Installer les dépendances
pip install -r requirements.txt

# Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec vos configurations

# Tester l'agent
python test_audit_agent.py
```

### 2. Déploiement Vercel

```bash
# Configurer vercel.json
{
  "functions": {
    "agents/audit/handler.py": {
      "runtime": "python3.9"
    }
  },
  "routes": [
    {
      "src": "/audit/(.*)",
      "dest": "/agents/audit/handler.py"
    }
  ]
}

# Déployer
vercel deploy
```

### 3. Configuration Docker (optionnel)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "agents.audit.handler:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🧪 Tests et validation

### Script de test intégré

```bash
# Lancer tous les tests
python test_audit_agent.py

# Tests spécifiques
python -c "
import asyncio
from agents.audit.agent import DocumentAuditMCPAgent

async def test():
    agent = DocumentAuditMCPAgent()
    result = await agent.process('audit health', {})
    print(result['response'])

asyncio.run(test())
"
```

### Tests de santé

```python
# Vérification santé agent
GET /health
{
  "status": "healthy",
  "agent": "Document Extraction & Chunking Audit Agent", 
  "agent_id": "audit_agent",
  "capabilities": [
    "document_audit", "extraction_quality", "chunking_analysis", 
    "supabase_vector_audit", "document_extraction", "document_chunking", 
    "vector_storage", "pipeline_processing"
  ],
  "dependencies": {
    "audit_engine": "healthy",
    "external_connections": {
      "context_rag": "reachable",
      "supabase": "connected", 
      "docling": "not_configured"
    },
    "audit_stats": {
      "documents_processed": 15,
      "chunks_created": 245,
      "embeddings_generated": 238,
      "pipeline_operations": 8
    }
  }
}
```

## 📈 Monitoring et métriques

### Métriques exposées

- **Documents traités** : Compteur total de documents extraits
- **Chunks créés** : Nombre total de chunks générés
- **Embeddings générés** : Embeddings OpenAI créés avec succès
- **Pipelines complets** : Opérations de bout en bout réussies
- **Audits effectués** : Compteur total par type d'audit
- **Score qualité moyen** : Moyenne mobile des scores d'extraction
- **Temps de réponse** : Latence par type d'opération
- **Taux d'erreur** : Pourcentage d'échecs par étape
- **Utilisation services** : Santé des dépendances externes

### Alertes recommandées

1. **Score qualité < 70%** : Dégradation qualité extraction
2. **Temps réponse > 30s** : Performance dégradée  
3. **Taux erreur > 10%** : Problèmes systémiques
4. **Embeddings échecs > 20%** : Problème API OpenAI
5. **Chunks vides > 5%** : Problème chunking
6. **Services indisponibles** : Perte de connectivité Supabase/Docling

## 🔄 Intégration avec autres agents

### Master Agent

L'agent d'audit peut être orchestré par le Master Agent :

```python
# Dans master/agent.py
self.agents_config = {
    "audit": {
        "url": f"{base_url}/audit",
        "capabilities": [
            "document_audit", "extraction_quality", "chunking_analysis",
            "document_extraction", "pipeline_processing", "vector_storage"
        ]
    }
}
```

### Workflow automatisé

```python
# 1. Pipeline complet de traitement via l'agent
async def process_document_via_agent(file_path, options={}):
    # Traitement complet : extraction → chunking → embeddings → sauvegarde
    result = await call_audit_agent(f"process pipeline {file_path} title={options.get('title', '')}")
    
    # Vérification automatique de la qualité
    if result.get("processing_stats", {}).get("success"):
        doc_id = result.get("processing_stats", {}).get("document_id")
        return {"success": True, "document_id": doc_id}
    else:
        await send_processing_alert(file_path, result)
        return {"success": False, "error": result.get("error")}

# 2. Audit automatique d'un pipeline existant  
async def audit_existing_pipeline(doc_id):
    # Audit complet d'un document déjà traité
    audit_result = await call_audit_agent(f"audit full pipeline document_id {doc_id}")
    
    # Alertes si nécessaire
    if audit_result.get("quality_score", 0) < 0.7:
        await send_quality_alert(doc_id, audit_result)
    
    return audit_result

# 3. Traitement par batch avec audit
async def process_document_batch(file_paths):
    results = []
    for file_path in file_paths:
        # Traitement
        result = await call_audit_agent(f"process pipeline {file_path}")
        results.append(result)
        
        # Audit immédiat
        if result.get("processing_stats", {}).get("document_id"):
            doc_id = result["processing_stats"]["document_id"]
            await call_audit_agent(f"audit full pipeline document_id {doc_id}")
    
    return results
```

## 🛠️ Développement et extension

### Ajouter nouveaux types d'audit

```python
# Dans audit_document_extraction()
if audit_intent["audit_type"] == "my_new_audit":
    result = await self.audit_my_new_feature(params)
    self.audit_stats["my_new_audits"] += 1
```

### Ajouter nouveaux services MCP

```json
// Dans mcp.json
{
  "servers": {
    "my_service": {
      "type": "http",
      "url": "${MY_SERVICE_URL}",
      "tools": ["my_tool"],
      "categories": ["my_category"]
    }
  },
  "routing": {
    "my_category": ["my_service"]
  }
}
```

### Personnaliser rapports

```python
def _format_my_custom_report(self, data: Dict) -> str:
    report = "### 🔧 Mon Audit Custom\n\n"
    # Logique de formatage personnalisée
    return report
```

## 📚 Ressources

### Documentation technique

- [Protocol MCP](https://modelcontextprotocol.io/) - Standard de communication
- [FastAPI](https://fastapi.tiangolo.com/) - Framework web Python
- [Supabase](https://supabase.com/docs) - Base de données vectorielle
- [Docling](https://github.com/DS4SD/docling) - Extraction de documents

### Fichiers clés

- `agents/audit/agent.py` - Agent principal
- `agents/audit/handler.py` - Handler Vercel
- `shared/base_agent.py` - Classe de base MCP
- `shared/mcp_client.py` - Client MCP
- `mcp.json` - Configuration serveurs MCP
- `test_audit_agent.py` - Tests automatisés

### Support et contribution

Pour des questions ou contributions :

1. Consulter la documentation interne
2. Tester avec `test_audit_agent.py`
3. Vérifier les logs structurés
4. Valider la configuration MCP

## 🎯 Cas d'usage

### 1. Remplacement Context RAG API
```bash
# Au lieu de votre API Context RAG
curl -X POST /api/upload -F 'file=@document.pdf'

# Utilisez l'agent MCP
curl -X POST /mcp -d '{"method":"process","params":{"request":"process pipeline document.pdf"}}'
```

### 2. Traitement par batch
```python
files = ["/path/to/doc1.pdf", "/path/to/doc2.docx", "/path/to/doc3.xlsx"]
for file_path in files:
    result = await agent.process(f"process pipeline {file_path} category=BatchImport", {})
    print(f"Document {file_path}: {result.get('processing_stats', {}).get('document_id')}")
```

### 3. Audit qualité continue
```python
# Audit automatique tous les documents créés aujourd'hui
recent_docs = get_recent_documents(days=1)
for doc_id in recent_docs:
    audit = await agent.process(f"audit full pipeline document_id {doc_id}", {})
    if audit.get("quality_score", 0) < 0.8:
        alert_quality_issue(doc_id, audit)
```

### 4. Test de régression
```bash
# Tester l'extraction sur un ensemble de fichiers de référence
for file in test_files/*.pdf; do
    curl -X POST /mcp -d "{\"method\":\"process\",\"params\":{\"request\":\"extract $file\"}}"
done
```

---

*Agent MCP d'Extraction & Processing - Pipeline complet de traitement documentaire avec audit qualité* 🚀