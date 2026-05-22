# 🧠 AI Research Intelligence Platform

> **Bloomberg Terminal for AI Research Trends** — A production-grade platform that ingests AI ecosystem signals, ranks trends using an X-style algorithm, forecasts momentum, and displays everything in a stunning modern dashboard.

![Platform Preview](docs/preview.png)

---

## 🎯 Overview

The AI Research Intelligence Platform continuously monitors the AI ecosystem across multiple data sources:
- **arXiv** — Latest research papers
- **GitHub** — Trending repositories  
- **HuggingFace** — Trending models & papers
- **PapersWithCode** — SOTA implementations
- **Reddit** — Community sentiment & discussions

It processes this data through a sophisticated ML pipeline to:
1. **Extract topics** using BERTopic clustering
2. **Forecast momentum** using Prophet time-series models
3. **Rank everything** using an X/Twitter-inspired ranking algorithm
4. **Generate AI insights** using open-source LLMs (Qwen/DeepSeek via Ollama)
5. **Display results** in a beautiful real-time dashboard

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   FRONTEND (Next.js 15)                  │
│  Dashboard · Trends · Papers · Repos · Forecasts · Topics│
└──────────────────┬──────────────────────────────────────┘
                   │ REST API + WebSocket
┌──────────────────▼──────────────────────────────────────┐
│                  BACKEND (FastAPI)                        │
│  Ingestion API · Ranking API · Forecast API · LLM API    │
└──────┬──────────┬────────────────────────────────────────┘
       │          │
┌──────▼──┐  ┌───▼──────────────────────────────────────┐
│  Redis  │  │           ML Pipeline                     │
│  Cache  │  │  BERTopic · Embeddings · Prophet · XGB    │
└─────────┘  └──────────┬───────────────────────────────┘
                        │
┌───────────────────────▼───────────────────────────────┐
│              MCP Servers Layer                         │
│  arxiv · github · huggingface · paperswithcode · reddit│
└───────────────────────────────────────────────────────┘
                        │
┌───────────────────────▼───────────────────────────────┐
│            Data Storage                                │
│  PostgreSQL · Qdrant (vectors) · Redis (cache/streams) │
└───────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
ai-research-platform/
├── frontend/                    # Next.js 15 dashboard
│   ├── app/
│   │   ├── page.tsx            # Main dashboard
│   │   ├── trends/             # Trending topics page
│   │   ├── papers/             # Research papers page
│   │   ├── repos/              # GitHub repos page
│   │   ├── forecasts/          # Forecast charts page
│   │   └── topics/             # Topic explorer page
│   ├── components/
│   │   ├── TrendCard.tsx       # Animated trend cards
│   │   ├── ForecastChart.tsx   # ECharts time-series
│   │   ├── TopicGraph.tsx      # D3 force-directed graph
│   │   ├── LiveFeed.tsx        # WebSocket live feed
│   │   └── RankingTable.tsx    # Sortable rankings
│   └── lib/
│       ├── api.ts              # API client
│       └── types.ts            # TypeScript types
│
├── backend/                     # FastAPI backend
│   ├── main.py                 # App entrypoint
│   ├── api/                    # Route handlers
│   ├── services/               # Business logic
│   ├── models/                 # SQLAlchemy ORM models
│   ├── schemas/                # Pydantic schemas
│   └── db/                     # Database config
│
├── mcp_servers/                 # MCP data source servers
│   ├── arxiv_server.py
│   ├── github_server.py
│   ├── huggingface_server.py
│   ├── paperswithcode_server.py
│   ├── reddit_server.py
│   └── ingestion_orchestrator.py
│
├── ml_pipeline/                 # ML components
│   ├── embedding_service.py    # SentenceTransformers
│   ├── topic_extractor.py      # BERTopic
│   └── llm_service.py          # LLM insight generation
│
├── ranking_engine/              # X-style ranking
│   ├── scorer.py               # Modular scoring formula
│   ├── pipeline.py             # Multi-stage pipeline
│   └── candidate_retrieval.py  # Candidate fetching
│
├── forecasting/                 # Time-series forecasting
│   ├── prophet_forecaster.py   # Prophet model
│   ├── xgboost_forecaster.py   # XGBoost alternative
│   └── aggregator.py           # Signal aggregation
│
├── vector_db/                   # Qdrant integration
│   └── qdrant_client.py
│
├── docker/                      # Docker configs
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── nginx.conf
│
├── scripts/                     # Utility scripts
│   ├── init_db.py
│   └── run_ingestion.py
│
├── docker-compose.yml
├── requirements.txt
├── Makefile
└── .env.example
```

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- Node.js 20+
- Ollama (optional, for LLM insights)

### 1. Clone & Configure

```bash
git clone <repo-url>
cd ai-research-platform
cp .env.example .env
# Edit .env with your API keys
```

### 2. Start with Docker Compose

```bash
# Start all services
docker-compose up -d

# Initialize database
make init-db

# Run first ingestion
make ingest

# View logs
docker-compose logs -f backend
```

### 3. Access the Platform

| Service | URL |
|---------|-----|
| **Frontend Dashboard** | http://localhost:3000 |
| **Backend API** | http://localhost:8000 |
| **API Docs** | http://localhost:8000/docs |
| **Qdrant UI** | http://localhost:6333/dashboard |

---

## 🔧 Development Setup

### Backend

```bash
# Install dependencies
pip install -r requirements.txt

# Start infrastructure only
docker-compose up -d postgres redis qdrant

# Initialize DB
python scripts/init_db.py

# Run backend
uvicorn backend.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:3000
```

### Run Ingestion Manually

```bash
# Ingest all sources
python scripts/run_ingestion.py --sources all

# Ingest specific sources
python scripts/run_ingestion.py --sources arxiv github

# Available sources: arxiv, github, huggingface, paperswithcode, reddit
```

---

## 📊 Ranking Algorithm

The platform uses an **X/Twitter-inspired ranking algorithm**:

```
trend_score = 0.35 × growth_velocity
            + 0.25 × github_activity
            + 0.20 × citation_acceleration
            + 0.10 × community_engagement
            + 0.10 × novelty_score
```

### Pipeline Stages

```
Candidate Retrieval (N=500)
        ↓
Signal Filtering (quality threshold)
        ↓
Embedding Similarity (semantic relevance)
        ↓
Momentum Scoring (Prophet forecast)
        ↓
Engagement Boosting (recency + virality)
        ↓
Novelty Boosting (diversity injection)
        ↓
Re-Ranking (final sort)
        ↓
Top-K Results
```

---

## 🤖 AI Features

### Topic Extraction
Uses **BERTopic** + **SentenceTransformers** to extract emerging topics from paper abstracts and repo descriptions.

### Forecasting
Uses **Facebook Prophet** for time-series forecasting of:
- Topic growth velocity
- GitHub star momentum
- Citation acceleration

### AI Insights
Uses **Qwen 2.5** (via Ollama) to generate contextual insights like:

> *"GraphRAG is rapidly gaining momentum due to enterprise retrieval applications and multi-agent workflows. Star growth has accelerated 340% in the past 30 days, with 12 new implementations released this week."*

---

## 🔌 API Reference

### Trends
- `GET /api/v1/trends/` — Ranked trending items
- `GET /api/v1/trends/topics` — Trending topics
- `GET /api/v1/trends/dashboard` — Dashboard stats

### Papers
- `GET /api/v1/papers/` — List papers
- `GET /api/v1/papers/search?q=...` — Semantic search
- `GET /api/v1/papers/{id}/similar` — Similar papers
- `GET /api/v1/papers/{id}/insight` — AI insight

### Forecasts
- `GET /api/v1/forecasts/{entity_id}` — Get forecast
- `POST /api/v1/forecasts/compute` — Trigger computation

### WebSocket
- `WS /ws/live-feed` — Real-time updates stream

Full API documentation at: `http://localhost:8000/docs`

---

## ⚙️ Configuration

Key environment variables (see `.env.example`):

| Variable | Description |
|----------|-------------|
| `GITHUB_TOKEN` | GitHub Personal Access Token |
| `REDDIT_CLIENT_ID` | Reddit App Client ID |
| `REDDIT_CLIENT_SECRET` | Reddit App Secret |
| `HUGGINGFACE_API_KEY` | HuggingFace API token |
| `OLLAMA_MODEL` | LLM model (default: qwen2.5:7b) |
| `EMBEDDING_MODEL` | Embedding model |
| `DATABASE_URL` | PostgreSQL connection string |

---

## 🐳 Docker Services

| Service | Port | Description |
|---------|------|-------------|
| `frontend` | 3000 | Next.js dashboard |
| `backend` | 8000 | FastAPI server |
| `postgres` | 5432 | PostgreSQL database |
| `redis` | 6379 | Cache & message broker |
| `qdrant` | 6333 | Vector database |
| `kafka` | 9092 | Event streaming |
| `nginx` | 80 | Reverse proxy |

---

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific test module
pytest backend/tests/test_ranking.py -v

# Run with coverage
pytest --cov=backend --cov-report=html
```

---

## 📈 Monitoring

- **Health check**: `GET /health`
- **Metrics**: Prometheus-compatible at `/metrics` (optional)
- **Logs**: Structured JSON via `structlog`

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

- **X/Twitter** — Ranking algorithm inspiration
- **Meta Prophet** — Time-series forecasting
- **HuggingFace** — Transformers & model hub
- **MCP Protocol** — Anthropic's Model Context Protocol
- **BERTopic** — Topic modeling
