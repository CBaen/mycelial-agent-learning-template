# MAE Deployment Guide

Complete guide for deploying the Mycelial Agent Engine in various environments.

---

## Table of Contents

1. [Quick Start (Docker)](#quick-start-docker)
2. [Local Development Setup](#local-development-setup)
3. [Production Deployment](#production-deployment)
4. [Configuration](#configuration)
5. [Monitoring](#monitoring)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start (Docker)

The fastest way to get MAE running is with Docker Compose.

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 4GB RAM minimum
- 10GB disk space

### Steps

1. **Clone and navigate to project**:
   ```bash
   cd agent-learning-template-codebase
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

3. **Start the stack**:
   ```bash
   docker-compose up -d
   ```

4. **Verify services**:
   ```bash
   docker-compose ps
   ```

   You should see:
   - `mae_redis` - Running
   - `mae_chromadb` - Running
   - `mae_app` - Running

5. **View logs**:
   ```bash
   docker-compose logs -f mae
   ```

6. **Access services**:
   - Redis: `localhost:6379`
   - ChromaDB: `localhost:8000`
   - Application logs: `docker-compose logs mae`

7. **Stop the stack**:
   ```bash
   docker-compose down
   ```

---

## Local Development Setup

For development without Docker:

### Prerequisites

- Python 3.8+
- Redis 6.0+
- 4GB RAM minimum

### Steps

1. **Install Redis**:

   **macOS**:
   ```bash
   brew install redis
   redis-server &
   ```

   **Ubuntu/Debian**:
   ```bash
   sudo apt-get install redis-server
   sudo systemctl start redis
   ```

   **Windows**:
   - Use WSL2 or Docker
   - Or download from https://redis.io/download

2. **Install Python dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env as needed
   ```

4. **Create data directories**:
   ```bash
   mkdir -p data logs simulation_results
   ```

5. **Run simulation mode**:
   ```bash
   python run_simulation.py
   ```

6. **Or run live mode**:
   ```bash
   python run_live.py
   ```

---

## Production Deployment

### Architecture Options

#### Option 1: Single Server (Small Deployment)

For 10-50 agents, single server is sufficient.

**Server Requirements**:
- 4 vCPUs
- 8GB RAM
- 50GB SSD
- Ubuntu 22.04 LTS

**Deployment**:

1. **Install Docker**:
   ```bash
   curl -fsSL https://get.docker.com | sh
   sudo usermod -aG docker $USER
   ```

2. **Clone repository**:
   ```bash
   git clone <your-repo>
   cd agent-learning-template-codebase
   ```

3. **Configure for production**:
   ```bash
   cp .env.example .env
   nano .env  # Set production values
   ```

4. **Update config.yaml**:
   ```yaml
   # config/config.yaml
   redis:
     password: "YOUR_STRONG_PASSWORD"  # Set strong password

   logging:
     level: WARNING
     log_file_path: /var/log/mae/mae.log

   agents:
     num_specialists: 20  # Scale as needed
   ```

5. **Start with restart policy**:
   ```bash
   docker-compose up -d
   ```

6. **Set up log rotation**:
   ```bash
   sudo nano /etc/logrotate.d/mae
   ```

   Add:
   ```
   /var/log/mae/*.log {
       daily
       rotate 7
       compress
       delaycompress
       notifempty
       create 0640 mae mae
       sharedscripts
   }
   ```

#### Option 2: Distributed Deployment (Large Scale)

For 50+ agents, use distributed architecture.

**Components**:
- Load Balancer (Nginx/HAProxy)
- Redis Cluster (3+ nodes)
- Multiple MAE instances (horizontal scaling)
- Shared Vector DB (Milvus cluster)
- Centralized logging (ELK stack)

**Redis Cluster Setup**:

1. **Create cluster config**:
   ```bash
   # redis-cluster.yml
   version: '3.8'
   services:
     redis-node-1:
       image: redis:7-alpine
       command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf
       ports:
         - "7001:6379"
     # Add nodes 2-6...
   ```

2. **Initialize cluster**:
   ```bash
   redis-cli --cluster create \
     node1:7001 node2:7002 node3:7003 \
     node4:7004 node5:7005 node6:7006 \
     --cluster-replicas 1
   ```

**MAE Instance Scaling**:

```yaml
# docker-compose.prod.yml
services:
  mae-1:
    image: mae:latest
    environment:
      - REDIS_HOST=redis-cluster
      - AGENT_ID_START=0
      - AGENT_ID_END=10

  mae-2:
    image: mae:latest
    environment:
      - REDIS_HOST=redis-cluster
      - AGENT_ID_START=11
      - AGENT_ID_END=20

  # Add more instances...
```

---

## Configuration

### Environment Variables

Priority: **Environment Variables** > **config.yaml** > **Defaults**

**Critical Variables**:

```bash
# Redis
REDIS_HOST=prod-redis.example.com
REDIS_PORT=6379
REDIS_PASSWORD=your_secure_password

# Vector DB
VECTOR_DB_BACKEND=milvus  # For production scale
VECTOR_DB_HOST=vectordb-cluster.example.com

# Application
LOG_LEVEL=WARNING
MAE_CONFIG_PATH=/app/config/config.yaml
```

### Security Hardening

1. **Redis Security**:
   ```conf
   # redis.conf
   requirepass YOUR_STRONG_PASSWORD
   bind 127.0.0.1  # Only local connections
   rename-command FLUSHDB ""
   rename-command FLUSHALL ""
   rename-command CONFIG ""
   ```

2. **Network Security**:
   - Use VPC/private networks
   - Enable TLS for Redis connections
   - Firewall rules: only allow required ports
   - Use secrets management (AWS Secrets Manager, Vault)

3. **Application Security**:
   - Don't expose ChromaDB port publicly
   - Use environment variables for secrets
   - Enable audit logging
   - Regular security updates

---

## Monitoring

### Health Checks

**Docker Compose Built-in**:
```bash
docker-compose ps
# All services should show (healthy)
```

**Manual Health Checks**:

```bash
# Redis
redis-cli ping
# Expected: PONG

# ChromaDB
curl http://localhost:8000/api/v1/heartbeat
# Expected: 200 OK

# MAE Application
docker logs mae_app --tail 50
```

### Metrics Collection

**Option 1: Redis INFO**:
```bash
redis-cli INFO stats
```

Key metrics:
- `total_commands_processed`
- `used_memory`
- `connected_clients`

**Option 2: Application Metrics**:

MAE logs metrics to SQLite. Query with:

```python
from connectors.sql_logger import SQLiteLogger

logger = SQLiteLogger()
stats = logger.get_statistics()
print(stats)
```

**Option 3: Prometheus (Advanced)**:

1. Add prometheus client to requirements.txt
2. Expose metrics endpoint
3. Configure Prometheus to scrape

### Logging

**View Live Logs**:
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f mae

# Last 100 lines
docker-compose logs --tail=100 mae
```

**Log Locations**:
- Application: `logs/mae.log`
- SQLite DB: `data/mae_system.db`
- Redis: Docker logs
- ChromaDB: Docker logs

### Alerting

Set up alerts for:
- Redis connection failures
- High memory usage (>80%)
- Risk events (contagion detected)
- System errors

**Example with cron**:

```bash
# /etc/cron.hourly/mae-health-check
#!/bin/bash
if ! docker ps | grep -q mae_app; then
    echo "MAE app is down!" | mail -s "MAE Alert" admin@example.com
fi
```

---

## Troubleshooting

### Common Issues

#### 1. Redis Connection Refused

**Symptoms**:
```
redis.exceptions.ConnectionError: Error connecting to localhost:6379
```

**Solution**:
```bash
# Check if Redis is running
docker ps | grep redis
# or
ps aux | grep redis

# Start Redis
docker-compose up -d redis
# or
redis-server
```

#### 2. ChromaDB Not Starting

**Symptoms**:
```
chromadb.errors.ChromaError: Could not connect to ChromaDB
```

**Solution**:
```bash
# Check logs
docker logs mae_chromadb

# Restart ChromaDB
docker-compose restart chromadb

# Clear ChromaDB data (WARNING: deletes all embeddings)
docker-compose down
docker volume rm mae_chromadb_data
docker-compose up -d
```

#### 3. Memory Issues

**Symptoms**:
```
MemoryError: Unable to allocate array
```

**Solution**:
```bash
# Check memory usage
docker stats

# Increase Docker memory limit
# Edit docker-compose.yml:
services:
  mae:
    deploy:
      resources:
        limits:
          memory: 4G  # Increase from 2G
```

#### 4. Port Already in Use

**Symptoms**:
```
Error starting userland proxy: listen tcp 0.0.0.0:6379: bind: address already in use
```

**Solution**:
```bash
# Find process using port
sudo lsof -i :6379
# or
sudo netstat -tulpn | grep 6379

# Kill process or change port in docker-compose.yml
ports:
  - "6380:6379"  # Use different host port
```

#### 5. Agents Not Receiving Data

**Symptoms**:
- No data processing logs
- DataMinerAgent shows 0 records processed

**Solution**:
```bash
# Check Redis stream
redis-cli XLEN mae:data_ingestion
# Expected: > 0

# Check if data producer is running
docker ps | grep producer

# Verify stream name in config
# config/config.yaml should match data producer
```

### Performance Tuning

**Slow Performance**:

1. **Enable connection pooling**:
   ```yaml
   # config.yaml
   redis:
     max_connections: 100
   ```

2. **Reduce logging verbosity**:
   ```yaml
   logging:
     level: WARNING  # Instead of DEBUG
   ```

3. **Optimize batch sizes**:
   ```yaml
   agents:
     batch_size: 50  # Increase for better throughput

   sqlite:
     batch_size: 500  # Larger batches = fewer writes
   ```

4. **Use Redis pipelining** (implement in code):
   ```python
   pipe = redis_client.client.pipeline()
   for item in batch:
       pipe.set(key, value)
   pipe.execute()
   ```

### Debugging

**Enable Debug Logging**:
```bash
# Set in .env
LOG_LEVEL=DEBUG

# Or in config.yaml
logging:
  level: DEBUG

# Restart
docker-compose restart mae
```

**Interactive Debugging**:
```bash
# Attach to running container
docker exec -it mae_app bash

# Run Python REPL
python

# Import and inspect
from config.settings import settings
print(settings.to_dict())
```

**Database Inspection**:
```bash
# SQLite
sqlite3 data/mae_system.db
> SELECT * FROM agent_events LIMIT 10;

# Redis
redis-cli
> KEYS *
> XINFO STREAM mae:data_ingestion
```

---

## Backup and Recovery

### Backup

**Redis**:
```bash
# Trigger save
redis-cli BGSAVE

# Copy RDB file
cp data/dump.rdb backups/dump-$(date +%Y%m%d).rdb
```

**SQLite**:
```bash
# Backup database
sqlite3 data/mae_system.db ".backup backups/mae_system-$(date +%Y%m%d).db"
```

**ChromaDB**:
```bash
# Backup volume
docker run --rm -v mae_chromadb_data:/data -v $(pwd)/backups:/backup \
  alpine tar czf /backup/chromadb-$(date +%Y%m%d).tar.gz /data
```

### Recovery

**Restore Redis**:
```bash
# Stop Redis
docker-compose stop redis

# Replace dump file
cp backups/dump-20250111.rdb data/dump.rdb

# Start Redis
docker-compose start redis
```

**Restore SQLite**:
```bash
cp backups/mae_system-20250111.db data/mae_system.db
```

**Restore ChromaDB**:
```bash
docker-compose down chromadb
docker volume rm mae_chromadb_data
docker run --rm -v mae_chromadb_data:/data -v $(pwd)/backups:/backup \
  alpine tar xzf /backup/chromadb-20250111.tar.gz -C /data
docker-compose up -d chromadb
```

---

## Scaling Guide

### Vertical Scaling (More Resources)

Increase resources for existing servers:

```yaml
# docker-compose.yml
services:
  mae:
    deploy:
      resources:
        limits:
          cpus: '4.0'    # Increase from 2.0
          memory: 8G     # Increase from 2G
```

### Horizontal Scaling (More Instances)

Add more MAE instances:

```bash
docker-compose up -d --scale mae=3
```

**Note**: Requires:
- Redis Cluster for coordination
- Shared vector DB
- Load balancer for API (if applicable)

---

## Support

For issues:
1. Check this guide
2. Review logs: `docker-compose logs`
3. Check [README.md](README.md)
4. Open GitHub issue

---

**Production Checklist**:
- [ ] Strong Redis password set
- [ ] Firewall configured
- [ ] Backup script running
- [ ] Monitoring enabled
- [ ] Log rotation configured
- [ ] Health checks passing
- [ ] Adversarial tests passed
- [ ] Documentation updated
