# MAE System Architecture

Complete technical architecture documentation for the Mycelial Agent Engine.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Data Backbone](#data-backbone)
3. [Component Architecture](#component-architecture)
4. [Data Flow](#data-flow)
5. [Scaling Strategy](#scaling-strategy)
6. [Pattern Recognition System](#pattern-recognition-system)

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MAE System Stack                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      Application Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Data Miner  │  │  Specialist  │  │     Risk     │         │
│  │    Agents    │  │    Agents    │  │   Manager    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────────┐
│                     Learning Engines                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │     FRL      │  │     VDN      │  │    HAVEN     │         │
│  │   (P2P)      │  │  (Credit)    │  │   (Risk)     │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────────┐
│                      Data Backbone                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │    Redis     │  │   Vector DB  │  │    SQLite    │         │
│  │ (Real-time)  │  │  (Memory)    │  │  (Persist)   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Decentralization**: No single point of failure
2. **Resilience**: Automatic recovery and fault tolerance
3. **Scalability**: Horizontal scaling via distribution
4. **Observability**: Comprehensive logging and metrics
5. **Security**: Defense in depth with multiple layers

---

## Data Backbone

### Three-Tier Storage Strategy

MAE uses three complementary data stores, each optimized for specific workloads:

#### 1. Redis - Real-Time Operations

**Purpose**: Hot path data and communication

**Use Cases**:
- Pub/Sub: Agent messages, system events, alerts
- Streams: Data ingestion pipeline
- Key-Value: Agent state, session data
- Sorted Sets: Leaderboards, rankings

**Characteristics**:
- In-memory (fast)
- Ephemeral (can be lost)
- High throughput (100k+ ops/sec)
- Low latency (<1ms)

**Data Types Used**:
```redis
# Pub/Sub Channels
PUBLISH mae:agent_messages {"from": "agent_1", "to": "agent_2", "type": "policy_share"}

# Streams (Data Ingestion)
XADD mae:data_ingestion * field1 value1 field2 value2

# Key-Value (Agent State)
SET agent:state:agent_1 '{"step": 100, "reward": 0.5}'

# Hash (Structured Data)
HSET agent:metrics:agent_1 cumulative_reward 125.5 tasks_completed 50
```

#### 2. Vector DB - Collective Memory

**Purpose**: Policy embeddings and semantic search

**Use Cases**:
- Store policy embeddings (128-512 dimensions)
- Similarity search for peer policies
- Pattern clustering and recognition
- Collective memory retrieval

**Characteristics**:
- Optimized for vector operations
- Supports similarity search
- Persistent storage
- Moderate latency (10-100ms)

**Operations**:
```python
# Add policy embedding
vector_db.add_policy_embedding(
    policy_id="policy_123",
    agent_id="agent_1",
    embedding=np.array([0.1, 0.2, ..., 0.5]),  # 128-dim vector
    metadata={"performance": 0.85, "version": 5}
)

# Search for similar policies
results = vector_db.search_similar_policies(
    query_embedding=current_policy_vector,
    top_k=10,
    filter_criteria={"performance": {"$gt": 0.7}}
)

# Cluster policies
clusters = vector_db.cluster_policies(num_clusters=10)
```

#### 3. SQLite - Persistent Archive

**Purpose**: Long-term storage and analytics

**Use Cases**:
- Event logging (agent events, system events)
- Performance metrics time series
- Pattern archive
- Risk event history
- Audit trail

**Characteristics**:
- Persistent (survives restarts)
- Thread-safe write queue
- Batch writes for efficiency
- Queryable with SQL

**Schema**:
```sql
-- Agent Events
CREATE TABLE agent_events (
    id INTEGER PRIMARY KEY,
    timestamp REAL,
    agent_id TEXT,
    event_type TEXT,
    data TEXT,
    step INTEGER
);

-- Patterns Archive
CREATE TABLE patterns (
    id INTEGER PRIMARY KEY,
    pattern_id TEXT UNIQUE,
    pattern_type TEXT,
    description TEXT,
    frequency INTEGER,
    confidence REAL,
    metadata TEXT
);

-- Performance Metrics
CREATE TABLE performance_metrics (
    id INTEGER PRIMARY KEY,
    timestamp REAL,
    agent_id TEXT,
    metric_name TEXT,
    metric_value REAL
);
```

### Data Flow Example

Here's how data flows through the system:

```
External Data Source
        │
        ▼
   Data Producer
        │
        ├──────────────────────┐
        │                      │
        ▼                      ▼
  Redis Stream          SQLite Logger
  (real-time)           (persist)
        │
        ▼
  Data Miner Agent
        │
        ├──────────────────────┐
        │                      │
        ▼                      ▼
  Redis Pub/Sub         SQLite Logger
  (broadcast)           (events)
        │
        ▼
  Specialist Agents
        │
        ├──────────┬──────────┬──────────┐
        │          │          │          │
        ▼          ▼          ▼          ▼
    Redis KV   Vector DB   SQLite   Redis Pub/Sub
    (state)    (policy)   (metrics)  (FRL)
```

---

## Component Architecture

### 1. Redis Client (`src/connectors/redis_client.py`)

**Responsibilities**:
- Connection pooling
- Pub/Sub management
- Stream operations
- Key-Value operations
- Error handling and retries

**Key Methods**:
```python
# Pub/Sub
publish(channel, message)
subscribe(channels)
listen()

# Streams
write_to_stream(stream_name, data)
read_from_stream(stream_name, last_id)

# Key-Value
set_key_value(key, value)
get_key_value(key)
set_hash(key, mapping)
get_hash(key)
```

### 2. SQLite Logger (`src/connectors/sql_logger.py`)

**Architecture**:

```
Application Thread
        │
        ▼
  Write Queue (Thread-Safe)
        │
        ▼
  Writer Thread (Background)
        │
        ▼
  Batch Writer (100 items/batch)
        │
        ▼
  SQLite Database
```

**Thread Safety**:
- All writes go through queue
- Single writer thread prevents locks
- Batching improves performance
- Non-blocking for application

**Key Methods**:
```python
# Logging
log_agent_event(agent_id, event_type, data)
log_pattern(pattern)
log_performance_metric(agent_id, metric_name, value)
log_risk_event(agent_id, risk_level, risk_score)

# Querying
get_agent_events(agent_id, event_type)
get_patterns(pattern_type, min_frequency)
get_performance_metrics(agent_id, metric_name)
get_risk_events(agent_id, min_risk_score)
```

### 3. Vector DB Interface (`src/connectors/vector_db.py`)

**Abstract Interface**:

Allows switching between backends without code changes:

```python
# ChromaDB (development)
vector_db = ChromaDBBackend(persist_directory="data/chromadb")

# Milvus (production)
vector_db = MilvusBackend(host="milvus-cluster")

# Qdrant (alternative)
vector_db = QdrantBackend(host="qdrant-server")
```

**Key Operations**:
```python
# Write
add_policy_embedding(policy_id, agent_id, embedding, metadata)
add_policy_embeddings_batch(embeddings)

# Read
search_similar_policies(query_embedding, top_k)
get_policy_embedding(policy_id)
get_agent_policies(agent_id)

# Analysis
cluster_policies(num_clusters)
find_policy_patterns(min_cluster_size)

# Maintenance
update_policy_metadata(policy_id, metadata)
delete_policy(policy_id)
clear_collection()
```

### 4. Settings Management (`config/settings.py`)

**Configuration Hierarchy**:

1. **Defaults** (in code)
2. **config.yaml** (overrides defaults)
3. **Environment Variables** (overrides yaml)

**Example**:
```python
# Default
redis.host = "localhost"

# config.yaml overrides
redis:
  host: prod-redis.example.com

# Environment variable overrides yaml
REDIS_HOST=staging-redis.example.com
# Final value: staging-redis.example.com
```

**Usage**:
```python
from config.settings import settings

# Access settings
redis_host = settings.redis.host
num_agents = settings.agents.num_specialists
risk_threshold = settings.haven.risk_threshold

# Validate
settings.validate()

# Reload
reload_settings("config/custom.yaml")
```

---

## Data Flow

### Simulation Mode Flow

```
1. Initialize System
   ├─ Load config.yaml
   ├─ Connect to Redis
   ├─ Initialize Vector DB
   └─ Initialize SQLite Logger

2. Create Agents
   ├─ Data Miner Agent(s)
   ├─ Specialist Agent(s)
   └─ Risk Manager Agent

3. Main Loop (N steps)
   │
   ├─ Data Miner Step
   │  ├─ Read from Redis Streams
   │  ├─ Validate & Transform
   │  ├─ Publish to Pub/Sub
   │  └─ Log to SQLite
   │
   ├─ Specialist Step
   │  ├─ Receive data (Pub/Sub)
   │  ├─ Select action (Policy)
   │  ├─ Execute action
   │  ├─ Get reward
   │  ├─ Update policy
   │  ├─ Share policy (FRL)
   │  ├─ Store embedding (Vector DB)
   │  └─ Log metrics (SQLite)
   │
   └─ Risk Manager Step
      ├─ Assess agent risks
      ├─ Detect contagion
      ├─ Execute interventions
      └─ Log risk events

4. Generate Report
   ├─ Query SQLite for metrics
   ├─ Analyze patterns (Vector DB)
   └─ Export results
```

### Live Mode Flow

```
External System
      │
      ▼
Data Producer (Your Custom Code)
      │
      ├──────────────┐
      │              │
      ▼              ▼
Redis Stream    SQLite Log
      │
      ▼
Data Miner Agent
      │
      ▼
Redis Pub/Sub
      │
      ▼
Specialist Agents (Continuously)
      │
      ├──────────────┬──────────────┐
      │              │              │
      ▼              ▼              ▼
  Execute        Update         Share
  Actions        Policy        (P2P)
      │              │              │
      ▼              ▼              ▼
  External     Vector DB      Peers
   System
```

---

## Scaling Strategy

### Vertical Scaling (Single Server)

**Recommended for**: 10-50 agents

**Optimization**:
```yaml
# config.yaml
redis:
  max_connections: 100

sqlite:
  batch_size: 500      # Larger batches
  queue_size: 20000    # Bigger queue

performance:
  enable_caching: true
  cache_ttl: 600
```

### Horizontal Scaling (Distributed)

**Recommended for**: 50+ agents

**Architecture**:

```
                 Load Balancer
                      │
         ┌────────────┼────────────┐
         │            │            │
    MAE Node 1   MAE Node 2   MAE Node 3
         │            │            │
         └────────────┼────────────┘
                      │
              Redis Cluster
                      │
         ┌────────────┼────────────┐
         │            │            │
    Vector DB     SQLite       External
    (Milvus)     (Shared)       APIs
```

**Configuration**:
```yaml
# Node 1
agents:
  agent_id_start: 0
  agent_id_end: 16

# Node 2
agents:
  agent_id_start: 17
  agent_id_end: 33

# Node 3
agents:
  agent_id_start: 34
  agent_id_end: 50
```

---

## Pattern Recognition System

### Week-by-Week Implementation (8-Week Plan)

#### Weeks 1-2: Data Collection
- SQLite logger captures all events
- Vector embeddings stored for each policy
- Baseline metrics established

#### Weeks 3-4: Pattern Detection
- Cluster policies using Vector DB
- Identify recurring patterns
- Store patterns in SQLite

#### Weeks 5-6: Pattern Classification
- Classify patterns (beneficial vs toxic)
- Correlate with performance metrics
- Build pattern taxonomy

#### Weeks 7-8: Pattern Application
- Use patterns for:
  - Anomaly detection
  - Fast policy initialization
  - Transfer learning
  - Collective memory

### Pattern Recognition Flow

```python
# 1. Capture Policy
policy_embedding = encode_policy(agent.policy)

vector_db.add_policy_embedding(
    policy_id=f"policy_{agent.id}_{step}",
    agent_id=agent.id,
    embedding=policy_embedding,
    performance=agent.recent_performance
)

# 2. Find Similar Patterns (Daily)
similar_policies = vector_db.search_similar_policies(
    query_embedding=policy_embedding,
    top_k=10,
    filter_criteria={"performance": {"$gt": 0.7}}
)

# 3. Cluster Patterns (Weekly)
clusters = vector_db.cluster_policies(
    num_clusters=20,
    filter_criteria={"performance": {"$gt": 0.5}}
)

# 4. Archive Patterns
for cluster_id, policy_ids in clusters.items():
    if len(policy_ids) >= 5:  # Significant pattern
        pattern = PatternEntry(
            pattern_id=f"cluster_{cluster_id}",
            pattern_type="policy_cluster",
            description=f"Cluster with {len(policy_ids)} policies",
            discovered_by="system",
            discovered_at=time.time(),
            frequency=len(policy_ids),
            confidence=calculate_confidence(policy_ids),
            metadata={"policy_ids": policy_ids}
        )
        sql_logger.log_pattern(pattern)
```

---

## Performance Considerations

### Redis Optimization

```conf
# redis.conf

# Memory
maxmemory 512mb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000
appendonly yes
appendfsync everysec

# Performance
tcp-backlog 511
tcp-keepalive 300
```

### SQLite Optimization

```python
# Connection settings
conn = sqlite3.connect(
    db_path,
    timeout=30.0,
    isolation_level=None  # Autocommit mode
)

# Write batching
batch_size = 100  # Fewer disk writes
flush_interval = 1.0  # Balance latency vs throughput
```

### Vector DB Optimization

```python
# Batch insertions
vector_db.add_policy_embeddings_batch(embeddings)

# Index optimization
collection.create_index(
    field_name="performance",
    index_type="BTREE"
)

# Query optimization
search_similar_policies(
    query_embedding=vec,
    top_k=10,  # Limit results
    filter_criteria={"agent_id": "agent_1"}  # Use filters
)
```

---

## Monitoring Architecture

### Metrics Collection

```python
# Application Metrics
sql_logger.log_performance_metric(
    agent_id="agent_1",
    metric_name="reward",
    metric_value=0.85,
    step=current_step
)

# System Metrics
stats = sql_logger.get_statistics()
# {
#   "writes_queued": 10000,
#   "writes_committed": 9950,
#   "write_errors": 0,
#   "agent_events_count": 5000,
#   "patterns_count": 25
# }

# Vector DB Metrics
vector_stats = vector_db.get_collection_stats()
# {
#   "total_policies": 1500,
#   "embedding_dim": 128,
#   "backend": "ChromaDB"
# }
```

### Health Checks

```python
# Redis health
redis_healthy = redis_client.ping()

# Vector DB health
vector_db_healthy = vector_db.is_initialized

# SQLite health
sqlite_healthy = sql_logger.running and sql_logger.write_errors < 10
```

---

## Security Architecture

### Defense in Depth

1. **Network Layer**:
   - VPC isolation
   - Security groups
   - TLS encryption

2. **Application Layer**:
   - Redis password authentication
   - Input validation
   - Rate limiting

3. **Data Layer**:
   - Encryption at rest (SQLite)
   - Encryption in transit (Redis TLS)
   - Access control

### HAVEN Integration

```python
# Risk-aware architecture
risk_manager.assess_agent_risk(agent_id, policy, performance)
if risk_score > threshold:
    risk_manager.execute_intervention(agent_id, InterventionType.ISOLATION)
    sql_logger.log_risk_event(agent_id, "HIGH", risk_score, factors)
```

---

This architecture enables MAE to scale from a single-server prototype to a distributed production system while maintaining resilience, observability, and security.
