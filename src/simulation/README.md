# Adversarial Simulation & Memory Engine

The simulation module provides a **self-healing testbed** and **pre-training tool** for the Mycelial Agent Engine (MAE). It validates HAVEN risk management, generates initial knowledge patterns, and ensures system resilience against adversarial behavior.

---

## 🎯 Purpose

### 1. **Adversarial Testing** (Self-Healing Testbed)
- Validate HAVEN risk detection and containment
- Test policy contagion detection mechanisms
- Verify intervention effectiveness
- Ensure system resilience against toxic agents

### 2. **Memory Building** (Pre-Training Tool)
- Generate successful and failed pattern examples
- Create initial knowledge base for agents
- Store pattern embeddings in Vector DB
- Enable agents to start with domain knowledge

---

## 📁 Module Structure

```
src/simulation/
├── adversarial_model.py     # Mesa model for safety testing
├── toxic_agent.py            # ToxicAgent with 7 behavior types
├── simulation_runner.py      # Test orchestration & reporting
├── memory_builder.py         # Knowledge base pre-training (NEW)
└── README.md                 # This file
```

---

## 🔬 Component Details

### 1. `adversarial_model.py` - Safety Testing Model

**Purpose**: Specialized Mesa model for adversarial testing

**Key Features**:
- Deterministic seeding for reproducibility
- Contagion detection and tracking
- Intervention monitoring
- Comprehensive test reporting
- Performance degradation analysis

**Usage**:
```python
from simulation.adversarial_model import AdversarialModel

model = AdversarialModel(
    redis_client=redis_client,
    num_healthy_agents=10,
    num_toxic_agents=2,
    test_scenario="policy_contagion",
    haven_coordinator=haven_coordinator,
    random_seed=42
)

# Add agents...

results = model.run(num_steps=1000)
print(model.get_summary())
```

**Tracked Metrics**:
- Contagion detection time
- Infection rate
- Containment effectiveness
- System risk timeline
- Agent performance degradation
- Intervention count

---

### 2. `toxic_agent.py` - Adversarial Agent Types

**Purpose**: Agents programmed with toxic behaviors for testing

**7 Toxic Behavior Types**:

| Behavior | Description | Testing Goal |
|----------|-------------|--------------|
| `MANIPULATIVE` | Games reward system | Test anti-gaming measures |
| `HIGH_RISK` | Takes excessive risks | Test risk detection |
| `BUGGY` | Implementation bugs | Test error handling |
| `EXPLOITATIVE` | Exploits vulnerabilities | Test security |
| `SABOTAGE` | Actively harms system | Test resilience |
| `POLICY_POISONING` | Spreads toxic policies | Test contagion detection |
| `BYZANTINE` | Random malicious behavior | Test robustness |

**Usage**:
```python
from simulation.toxic_agent import ToxicAgent, ToxicBehaviorType

# Create specific toxic agent
toxic = ToxicAgent(
    unique_id=1,
    model=model,
    redis_client=redis_client,
    toxic_behavior=ToxicBehaviorType.POLICY_POISONING,
    toxicity_level=0.8,  # 0.0 to 1.0
    data_channel="processed_data"
)

# Create mixed swarm
from simulation.toxic_agent import ToxicAgentFactory

toxic_swarm = ToxicAgentFactory.create_mixed_toxic_swarm(
    model=model,
    redis_client=redis_client,
    start_id=100,
    num_agents=5
)
```

**Key Methods**:
- `get_toxic_statistics()` - Toxic behavior metrics
- `get_detection_status()` - Detection/isolation status
- `share_policy()` - Aggressive policy sharing

---

### 3. `simulation_runner.py` - Test Orchestration

**Purpose**: Runs complete adversarial tests with reporting

**Features**:
- Automated test setup
- Redis database isolation (uses separate DB)
- HAVEN coordinator integration
- Comprehensive result reporting
- JSON and text output formats

**Usage**:
```python
from simulation.simulation_runner import SimulationRunner

runner = SimulationRunner(
    redis_host="localhost",
    redis_port=6379,
    redis_db=1,  # Separate DB for testing
    output_dir="simulation_results"
)

results = runner.run_full_test(
    num_healthy=10,
    num_toxic=2,
    toxic_behavior=ToxicBehaviorType.POLICY_POISONING,
    num_steps=1000
)

# Check results
if results["test_passed"]:
    print("✓ System successfully contained contagion")
else:
    print("✗ Contagion was not properly contained")
```

**Command Line**:
```bash
cd src/simulation
python simulation_runner.py
```

**Output Files**:
- `simulation_results/results_YYYYMMDD_HHMMSS.json` - Full metrics
- `simulation_results/report_YYYYMMDD_HHMMSS.txt` - Human-readable report

---

### 4. `memory_builder.py` - Knowledge Base Pre-Training ⭐ NEW

**Purpose**: Generate initial knowledge base for agent learning

**What It Does**:
1. Runs multiple adversarial simulations with varying parameters
2. Extracts successful and failed patterns from each simulation
3. Generates pattern embeddings (128-dim vectors)
4. Stores patterns in Vector DB for agent pre-training
5. Creates synthetic patterns for additional diversity
6. Validates memory quality

**7 Pattern Types**:

| Pattern Type | Description | Purpose |
|--------------|-------------|---------|
| `SUCCESSFUL_POLICY` | Successfully contained contagion | Learn what works |
| `FAILED_POLICY` | Failed to contain | Learn what doesn't work |
| `TOXIC_DETECTION` | Toxic agent detection signature | Learn detection patterns |
| `CONTAGION_PATTERN` | How toxicity spreads | Learn spread dynamics |
| `RECOVERY_PATTERN` | System recovery after attack | Learn resilience |
| `OPTIMAL_COORDINATION` | High-quality team coordination | Learn collaboration |
| `RISK_MITIGATION` | Successful risk reduction | Learn safety |

**Usage**:
```python
from simulation.memory_builder import MemoryBuilder
from connectors.redis_client import RedisClient
from connectors.vector_db import create_vector_db
from connectors.sql_logger import SQLiteLogger

# Initialize infrastructure
redis_client = RedisClient(host="localhost", port=6379, db=2)
vector_db = create_vector_db(
    backend="chromadb",
    collection_name="mae_patterns",
    embedding_dim=128,
    persist_directory="data/pattern_memory"
)
vector_db.initialize()
sql_logger = SQLiteLogger(db_path="data/memory_building.db")

# Create memory builder
builder = MemoryBuilder(
    redis_client=redis_client,
    vector_db=vector_db,
    sql_logger=sql_logger,
    output_dir="memory_building_results"
)

# Run pattern generation campaign
campaign_results = builder.run_pattern_generation_campaign(
    num_simulations=6,
    steps_per_simulation=300
)

# Generate synthetic patterns for diversity
builder.generate_synthetic_patterns(num_patterns=50)

# Validate quality
quality_report = builder.validate_memory_quality()
print(f"Quality Grade: {quality_report['quality_grade']}")
```

**Command Line**:
```bash
cd src/simulation
python memory_builder.py
```

**Output**:
- `memory_building_results/campaign_results_*.json` - Campaign metrics
- `memory_building_results/patterns_*.pkl` - Pickled patterns
- `data/pattern_memory/` - Vector DB with pattern embeddings
- `data/memory_building.db` - SQLite event log

**Default Scenarios Tested**:
1. Policy poisoning with high containment
2. Multiple toxic agents
3. Buggy agent detection
4. Byzantine behavior
5. Sabotage attack
6. Subtle attack (low toxicity)

---

## 🚀 Quick Start

### 1. Run Adversarial Test

```bash
# Start Redis
docker-compose up -d redis

# Run test
cd src/simulation
python simulation_runner.py
```

**Expected Output**:
```
=== Adversarial Simulation Report ===
Scenario: policy_contagion
Result: PASSED ✓

Contagion Detection:
- Detected: Yes
- Step: 45
- Infection Rate: 20.0%
- Containment Rate: 80.0%

System Response:
- Interventions: 3
- Final Risk: 0.234
```

---

### 2. Build Initial Knowledge Base

```bash
# Run memory builder
cd src/simulation
python memory_builder.py
```

**Expected Output**:
```
=== Pattern Generation Campaign ===
Running Scenario 1/6: policy_poisoning_high_containment
...
Extracted 8 patterns from simulation

Storing patterns in Vector DB...
Successfully stored 56 patterns in Vector DB

Memory Quality Report:
  Total Patterns: 56
  Pattern Types: 6
  Quality Grade: A (Excellent)
```

---

### 3. Query Patterns for Agent Learning

```python
# Query similar patterns from Vector DB
from simulation.memory_builder import MemoryBuilder

builder = MemoryBuilder(...)

# Define a query pattern
query = {
    "pattern_type": "SUCCESSFUL_POLICY",
    "metrics": {"containment_rate": 0.9},
    "outcome": "success"
}

# Find similar patterns
similar = builder.query_similar_patterns(query, top_k=5)

for pattern in similar:
    print(f"Pattern: {pattern['policy_id']}")
    print(f"Similarity: {pattern['similarity']:.3f}")
```

---

## 📊 Test Scenarios

### Default Adversarial Scenarios

All scenarios test different aspects of system resilience:

```python
# Scenario configurations
scenarios = [
    {
        "name": "policy_poisoning_high_containment",
        "num_healthy": 15,
        "num_toxic": 1,
        "toxic_behavior": ToxicBehaviorType.POLICY_POISONING,
        "toxicity_level": 0.7
    },
    {
        "name": "multiple_toxic_agents",
        "num_healthy": 12,
        "num_toxic": 3,
        "toxic_behavior": ToxicBehaviorType.HIGH_RISK,
        "toxicity_level": 0.8
    },
    # ... more scenarios
]
```

---

## 🧪 Testing Workflow

### Complete Testing & Pre-Training Pipeline

```bash
# 1. Run adversarial tests (validation)
python simulation_runner.py

# 2. Build knowledge base (pre-training)
python memory_builder.py

# 3. Verify patterns are stored
python -c "from connectors.vector_db import create_vector_db; \
           vdb = create_vector_db('chromadb', 'mae_patterns', 128, 'data/pattern_memory'); \
           vdb.initialize(); \
           print(f'Patterns stored: {vdb.collection.count()}')"

# 4. Run production system with pre-trained agents
cd ../..
python run_simulation.py
```

---

## 📈 Interpreting Results

### Test Pass Criteria

A test **PASSES** if:
1. ✅ Contagion is detected
2. ✅ Interventions are triggered
3. ✅ Infection rate < 30%
4. ✅ System risk decreases after interventions

### Quality Grades for Memory Building

| Grade | Score | Interpretation |
|-------|-------|----------------|
| **A** | >0.8 | Excellent diversity and balance |
| **B** | >0.6 | Good knowledge base |
| **C** | >0.4 | Adequate for basic learning |
| **D** | ≤0.4 | Needs more patterns |

---

## 🔧 Configuration

### Environment Variables

```bash
# Redis configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=1  # Use separate DB for testing

# Vector DB configuration
VECTOR_DB_BACKEND=chromadb
VECTOR_DB_COLLECTION=mae_patterns
VECTOR_DB_PERSIST_DIR=data/pattern_memory

# Simulation configuration
SIM_NUM_HEALTHY_AGENTS=10
SIM_NUM_TOXIC_AGENTS=2
SIM_NUM_STEPS=1000
```

---

## 🎓 How Agents Use Pre-Trained Patterns

### 1. During Initialization

```python
class SpecialistAgent(MycelialAgent):
    def __init__(self, ...):
        super().__init__(...)

        # Load similar patterns from Vector DB
        if self.vector_db:
            self._load_initial_patterns()

    def _load_initial_patterns(self):
        # Query Vector DB for relevant patterns
        patterns = self.vector_db.search_similar_policies(
            query_embedding=self.policy_embedding,
            filter_dict={"pattern_type": "SUCCESSFUL_POLICY"}
        )

        # Initialize policy based on successful patterns
        for pattern in patterns[:3]:
            self._integrate_pattern(pattern)
```

### 2. During Learning

Agents continuously query the Vector DB to:
- Find successful patterns similar to current situation
- Avoid known failure patterns
- Learn from toxic detection signatures
- Recover using proven recovery patterns

---

## 🛡️ Safety Features

### Built-in Safeguards

1. **Database Isolation**: Tests use separate Redis DB (default: 1)
2. **Data Cleanup**: Automatic flush before each test
3. **Deterministic Seeds**: Reproducible test results
4. **Comprehensive Logging**: Full audit trail
5. **Automatic Rollback**: Test failures don't affect production

### HAVEN Integration

- All toxic agents are monitored
- Risk assessments run every N steps
- Interventions trigger automatically
- Isolation prevents contagion spread
- Recovery mechanisms tested

---

## 📝 Example: Custom Test Scenario

```python
from simulation.simulation_runner import SimulationRunner
from simulation.toxic_agent import ToxicBehaviorType

runner = SimulationRunner(output_dir="custom_tests")

# Custom scenario: Stealthy attack
results = runner.run_full_test(
    num_healthy=20,
    num_toxic=1,
    toxic_behavior=ToxicBehaviorType.MANIPULATIVE,
    num_steps=2000  # Longer simulation
)

# Analyze results
if results["contagion_detected"]:
    detection_time = results["contagion_detection_step"]
    print(f"Detected manipulative agent at step {detection_time}")

    if results["test_passed"]:
        print("✓ System successfully contained subtle attack")
    else:
        print("✗ Manipulative agent evaded detection too long")
```

---

## 🔍 Debugging

### View Simulation Logs

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python simulation_runner.py
```

### Inspect Generated Patterns

```python
import pickle

# Load patterns from file
with open("memory_building_results/patterns_*.pkl", "rb") as f:
    patterns = pickle.load(f)

# Analyze patterns
for pattern in patterns[:10]:
    print(f"Pattern: {pattern['pattern_id']}")
    print(f"Type: {pattern['pattern_type']}")
    print(f"Outcome: {pattern['outcome']}")
    print(f"Confidence: {pattern['confidence']}")
    print("---")
```

### Query Vector DB Directly

```python
from connectors.vector_db import create_vector_db

vdb = create_vector_db("chromadb", "mae_patterns", 128, "data/pattern_memory")
vdb.initialize()

# Get all patterns
all_patterns = vdb.collection.get()
print(f"Total patterns: {len(all_patterns['ids'])}")

# Query by metadata
toxic_patterns = vdb.collection.get(
    where={"pattern_type": "TOXIC_DETECTION"}
)
print(f"Toxic detection patterns: {len(toxic_patterns['ids'])}")
```

---

## 🎯 Best Practices

### 1. **Run Tests Before Production**
Always validate HAVEN configuration with adversarial tests before deploying.

### 2. **Build Rich Knowledge Base**
Run memory builder with diverse scenarios to create comprehensive patterns.

### 3. **Monitor Quality Grades**
Aim for Grade A or B memory quality for best agent performance.

### 4. **Use Separate Databases**
Keep test data isolated from production (use different Redis DB numbers).

### 5. **Version Control Patterns**
Save pattern files with timestamps for reproducibility.

### 6. **Validate After Changes**
Re-run adversarial tests after any HAVEN configuration changes.

---

## 📚 Related Documentation

- [ARCHITECTURE.md](../../ARCHITECTURE.md) - System architecture
- [README.md](../../README.md) - Main documentation
- [QUICKSTART.md](../../QUICKSTART.md) - Getting started guide
- [src/core/haven_base.py](../core/haven_base.py) - HAVEN interface
- [src/core/builder_base.py](../core/builder_base.py) - Agent lifecycle

---

## 🤝 Contributing

When adding new toxic behaviors or test scenarios:

1. Add new behavior to `ToxicBehaviorType`
2. Implement behavior in `ToxicAgent._get_toxic_action()`
3. Add scenario to `MemoryBuilder._generate_default_scenarios()`
4. Run tests to validate
5. Update this README

---

**The simulation module is your safety net and training ground. Use it liberally! 🛡️**
