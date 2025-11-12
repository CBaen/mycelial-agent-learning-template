# MAE Quick Start Guide

Get the Mycelial Agent Engine running in 5 minutes.

---

## Prerequisites

```bash
# Check versions
docker --version    # 20.10+
docker-compose --version  # 2.0+
python --version   # 3.8+
redis-cli --version  # 6.0+
```

---

## Option 1: Docker (Recommended)

### 1. Start Services

```bash
# Clone repository
git clone <your-repo>
cd agent-learning-template-codebase

# Start Redis and ChromaDB
docker-compose up -d redis chromadb

# Wait 10 seconds for services to be healthy
sleep 10
```

### 2. Run Simulation

```bash
# Install Python dependencies (local)
pip install -r requirements.txt

# Run simulation
python run_simulation.py
```

### 3. Check Results

```bash
# View logs
tail -f logs/mae.log

# Check simulation results
ls -lh simulation_results/

# View latest report
cat simulation_results/report_*.txt | tail -50
```

---

## Option 2: Local (No Docker)

### 1. Start Redis

```bash
# macOS
brew services start redis

# Ubuntu
sudo systemctl start redis

# Or manually
redis-server &
```

### 2. Install and Run

```bash
# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data logs simulation_results

# Run simulation
python run_simulation.py
```

---

## Quick Commands

### View System Status

```bash
# Check Redis
redis-cli ping
# Expected: PONG

# Check ChromaDB (if using Docker)
curl http://localhost:8000/api/v1/heartbeat
# Expected: 200 OK

# Check Python environment
python -c "from config.settings import settings; print(settings.to_dict())"
```

### Inspect Data

```bash
# Redis data
redis-cli
> KEYS *
> XINFO STREAM mae:data_ingestion
> GET agent:state:SpecialistAgent_1

# SQLite data
sqlite3 data/mae_system.db
> SELECT COUNT(*) FROM agent_events;
> SELECT * FROM patterns ORDER BY frequency DESC LIMIT 10;
> .quit
```

### Run Tests

```bash
# Run adversarial simulation
cd src/simulation
python simulation_runner.py

# Expected output:
# Test: PASSED ✓
# Contagion detected and contained
```

---

## Common Tasks

### Change Agent Count

```yaml
# Edit config/config.yaml
agents:
  num_specialists: 20  # Change from 10 to 20
```

```bash
# Restart
docker-compose restart mae
# or
python run_simulation.py
```

### Change Redis Database

```bash
# Edit .env
REDIS_DB=1  # Use DB 1 instead of 0

# Or in config.yaml
redis:
  db: 1
```

### Enable Debug Logging

```bash
# Edit .env
LOG_LEVEL=DEBUG

# Or in config.yaml
logging:
  level: DEBUG
```

### Clear All Data

```bash
# Clear Redis
redis-cli FLUSHDB

# Clear SQLite
rm data/mae_system.db

# Clear Vector DB
rm -rf data/chromadb/

# Clear logs
rm -rf logs/*.log
```

---

## Troubleshooting

### Redis Connection Error

```bash
# Error: Connection refused

# Fix: Start Redis
docker-compose up -d redis
# or
redis-server &
```

### Import Errors

```bash
# Error: ModuleNotFoundError: No module named 'mesa'

# Fix: Install dependencies
pip install -r requirements.txt
```

### Port Already in Use

```bash
# Error: Address already in use (6379)

# Fix: Use different port
# Edit docker-compose.yml:
ports:
  - "6380:6379"  # Use port 6380

# Update config
REDIS_PORT=6380
```

### Out of Memory

```bash
# Error: MemoryError

# Fix: Increase Docker memory
# Docker Desktop -> Settings -> Resources -> Memory: 4GB
```

---

## Next Steps

1. **Customize for Your Domain**:
   - Read [README.md](README.md) - Implementation guide
   - Create your data producer
   - Implement custom agents

2. **Run Safety Tests**:
   - See [README.md#adversarial-testing](README.md#running-the-adversarial-simulation)
   - Tune risk thresholds
   - Validate contagion detection

3. **Deploy to Production**:
   - Read [DEPLOYMENT.md](DEPLOYMENT.md)
   - Configure security
   - Set up monitoring
   - Run backups

---

## Quick Reference

### File Locations

| What | Where |
|------|-------|
| Configuration | `config/config.yaml` |
| Environment | `.env` |
| Data producer | `src/connectors/your_producer.py` |
| Custom agents | `src/agents/my_specialist.py` |
| Logs | `logs/mae.log` |
| Database | `data/mae_system.db` |
| Test results | `simulation_results/` |

### Important Commands

```bash
# Start everything
docker-compose up -d

# View logs
docker-compose logs -f mae

# Stop everything
docker-compose down

# Run simulation
python run_simulation.py

# Run live mode
python run_live.py

# Run adversarial test
python src/simulation/simulation_runner.py
```

### Configuration Priority

```
Environment Variables  >  config.yaml  >  Code Defaults
```

Example:
```bash
REDIS_HOST=prod.example.com  # Highest priority
```

---

## Getting Help

- **Documentation**: Read [README.md](README.md)
- **Architecture**: Read [ARCHITECTURE.md](ARCHITECTURE.md)
- **Deployment**: Read [DEPLOYMENT.md](DEPLOYMENT.md)
- **Issues**: Check logs and troubleshooting sections

---

**You're ready to build your multi-agent system! 🚀**
