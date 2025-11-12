# MAE Complete File Structure

Production-ready Mycelial Agent Engine with full data backbone.

---

## Directory Tree

```
agent-learning-template-codebase/
│
├── README.md                          # Main documentation
├── ARCHITECTURE.md                    # Technical architecture guide
├── DEPLOYMENT.md                      # Deployment and operations guide
├── FILE_STRUCTURE.md                  # This file
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Docker image definition
├── docker-compose.yml                 # Multi-service orchestration
├── .env.example                       # Environment variables template
├── .gitignore                         # Git ignore rules
│
├── config/                            # ⚙️ Configuration
│   ├── settings.py                    # Settings loader and validator
│   ├── config.yaml                    # Main configuration file
│   └── redis.conf                     # Redis server configuration
│
├── src/                               # 📦 Source Code
│   │
│   ├── connectors/                    # 🔌 Data Backbone Connectors
│   │   ├── redis_client.py            # Redis operations (Pub/Sub, Streams, KV)
│   │   ├── sql_logger.py              # Thread-safe SQLite logger with write queue
│   │   └── vector_db.py               # Vector DB interface (ChromaDB, Milvus)
│   │
│   ├── core/                          # 🧠 Learning Engines (Abstract)
│   │   ├── model.py                   # Main Mesa model
│   │   ├── frl_base.py                # Federated Reinforcement Learning interface
│   │   ├── vdn_base.py                # Value-Decomposition Networks interface
│   │   └── haven_base.py              # HAVEN risk coordination interface
│   │
│   ├── agents/                        # 🤖 Agent Templates
│   │   ├── base_agent.py              # MycelialAgent base class
│   │   ├── data_miner_agent.py        # Data ingestion agent
│   │   ├── specialist_agent.py        # Main worker agent
│   │   └── risk_manager_agent.py      # HAVEN oversight agent
│   │
│   └── simulation/                    # 🧪 Adversarial Testing
│       ├── adversarial_model.py       # Safety test model
│       ├── toxic_agent.py             # Toxic behavior agents
│       └── simulation_runner.py       # Test orchestration
│
├── run_simulation.py                  # 🎮 Launch simulation mode
├── run_live.py                        # 🚀 Launch live/production mode
│
├── data/                              # 💾 Data Storage (gitignored)
│   ├── .gitkeep                       # Keep directory in git
│   ├── mae_system.db                  # SQLite database (auto-created)
│   └── chromadb/                      # ChromaDB persistence (auto-created)
│
├── logs/                              # 📝 Log Files (gitignored)
│   ├── .gitkeep
│   └── mae.log                        # Application logs (auto-created)
│
└── simulation_results/                # 📊 Test Results (gitignored)
    ├── .gitkeep
    ├── results_YYYYMMDD_HHMMSS.json   # Adversarial test results
    └── report_YYYYMMDD_HHMMSS.txt     # Human-readable reports
```

---

## File Purposes

### Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | Main documentation, getting started guide |
| `ARCHITECTURE.md` | Technical architecture, data flow, scaling |
| `DEPLOYMENT.md` | Deployment guide, troubleshooting, operations |
| `FILE_STRUCTURE.md` | This file - project organization |

### Configuration Files

| File | Purpose |
|------|---------|
| `config/settings.py` | Python settings loader with validation |
| `config/config.yaml` | Central YAML configuration |
| `config/redis.conf` | Production Redis configuration |
| `.env.example` | Environment variable template |

### Docker Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Multi-stage Docker image build |
| `docker-compose.yml` | Full stack deployment (Redis, ChromaDB, MAE) |

### Source Code - Connectors

| File | Purpose | Key Features |
|------|---------|--------------|
| `redis_client.py` | Redis operations | Pub/Sub, Streams, KV, connection pooling |
| `sql_logger.py` | Persistent logging | Thread-safe queue, batching, pattern archiving |
| `vector_db.py` | Collective memory | Policy embeddings, similarity search, clustering |

### Source Code - Core

| File | Purpose | Framework |
|------|---------|-----------|
| `model.py` | Main simulation model | Mesa Model |
| `frl_base.py` | P2P learning interface | Abstract base class |
| `vdn_base.py` | Credit assignment interface | Abstract base class |
| `haven_base.py` | Risk coordination interface | Abstract base class |

### Source Code - Agents

| File | Purpose | Inherits From |
|------|---------|---------------|
| `base_agent.py` | Base agent class | Mesa Agent |
| `data_miner_agent.py` | Data ingestion | MycelialAgent |
| `specialist_agent.py` | Worker agent | MycelialAgent |
| `risk_manager_agent.py` | Risk oversight | MycelialAgent |

### Source Code - Simulation

| File | Purpose |
|------|---------|
| `adversarial_model.py` | Safety testing model |
| `toxic_agent.py` | Bad behavior agents (7 types) |
| `simulation_runner.py` | Test orchestration and reporting |

### Entry Points

| File | Mode | Purpose |
|------|------|---------|
| `run_simulation.py` | Simulation | Fixed-duration testing |
| `run_live.py` | Live | Continuous production operation |

---

## File Dependencies

### Import Graph

```
settings.py
    ├─ config.yaml
    └─ Environment Variables

redis_client.py
    └─ redis (PyPI package)

sql_logger.py
    └─ sqlite3 (built-in)

vector_db.py
    ├─ chromadb (PyPI package)
    └─ numpy (PyPI package)

base_agent.py
    ├─ mesa (PyPI package)
    └─ redis_client.py

data_miner_agent.py
    └─ base_agent.py

specialist_agent.py
    └─ base_agent.py

risk_manager_agent.py
    ├─ base_agent.py
    └─ haven_base.py

model.py
    └─ mesa (PyPI package)

adversarial_model.py
    ├─ model.py
    └─ haven_base.py

toxic_agent.py
    └─ specialist_agent.py

simulation_runner.py
    ├─ adversarial_model.py
    ├─ toxic_agent.py
    ├─ specialist_agent.py
    └─ risk_manager_agent.py

run_simulation.py
    ├─ settings.py
    ├─ redis_client.py
    ├─ model.py
    └─ agents/*

run_live.py
    ├─ settings.py
    ├─ redis_client.py
    ├─ model.py
    └─ agents/*
```

---

## Data Files

### Auto-Generated Files

These files are created automatically during runtime:

```
data/
├── mae_system.db              # SQLite database
│   ├── agent_events           # Agent event log
│   ├── patterns               # Pattern archive
│   ├── performance_metrics    # Time series metrics
│   ├── system_events          # System events
│   └── risk_events            # Risk assessments

├── chromadb/                  # Vector database
│   ├── chroma.sqlite3         # ChromaDB metadata
│   └── *.parquet              # Vector data files

logs/
└── mae.log                    # Application logs

simulation_results/
├── results_*.json             # Test results (JSON)
└── report_*.txt               # Test reports (text)
```

---

## Configuration Cascade

Configuration values are resolved in this order (later overrides earlier):

1. **Code Defaults** (in `settings.py` dataclasses)
2. **config.yaml** (main configuration file)
3. **Environment Variables** (`.env` or system)

Example:

```python
# 1. Default in code
@dataclass
class RedisSettings:
    host: str = "localhost"  # Default

# 2. config.yaml overrides
redis:
  host: prod-redis.example.com  # Override

# 3. Environment variable overrides both
REDIS_HOST=staging-redis.example.com  # Final value
```

---

## Development Workflow

### 1. Initial Setup

```bash
# Clone repository
git clone <your-repo>
cd agent-learning-template-codebase

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
nano .env  # Edit configuration

# Start infrastructure
docker-compose up -d redis chromadb
```

### 2. Implement Domain Logic

```bash
# Create data producer
src/connectors/your_producer.py

# Customize agents
src/agents/my_specialist.py

# Update configuration
config/config.yaml
```

### 3. Test

```bash
# Run adversarial simulation
cd src/simulation
python simulation_runner.py

# Check results
cat ../../simulation_results/report_*.txt
```

### 4. Deploy

```bash
# Build Docker image
docker build -t mae:latest .

# Deploy full stack
docker-compose up -d

# Monitor
docker-compose logs -f mae
```

---

## File Size Estimates

Approximate file sizes (production):

| Component | Size | Notes |
|-----------|------|-------|
| Source code | ~500 KB | All Python files |
| Dependencies | ~500 MB | Python packages |
| Redis data | 10-100 MB | Varies with workload |
| SQLite DB | 100 MB - 1 GB | Grows over time |
| Vector DB | 500 MB - 5 GB | Depends on embeddings |
| Logs | 10 MB - 100 MB | With rotation |
| Docker images | ~1.5 GB | All services |

---

## Port Usage

Default ports used by MAE:

| Service | Port | Purpose |
|---------|------|---------|
| Redis | 6379 | Data backbone |
| ChromaDB | 8000 | Vector database |
| Redis Commander | 8081 | Redis web UI (optional) |

---

## Critical Files for Production

Must be properly configured before production:

- [ ] `config/config.yaml` - Production settings
- [ ] `config/redis.conf` - Redis password, persistence
- [ ] `.env` - API keys, secrets
- [ ] `docker-compose.yml` - Resource limits, restart policy
- [ ] Firewall rules - Restrict port access

---

## Backup Requirements

Files/data that should be backed up:

- **Critical**:
  - `data/mae_system.db` - All historical data
  - `data/chromadb/` - Policy embeddings
  - Redis RDB/AOF files - Current state

- **Important**:
  - `config/config.yaml` - Configuration
  - `.env` - Environment settings (encrypted)

- **Optional**:
  - `logs/` - Historical logs
  - `simulation_results/` - Test results

---

This file structure provides a production-ready foundation for building decentralized multi-agent learning systems with full observability, persistence, and scalability.
