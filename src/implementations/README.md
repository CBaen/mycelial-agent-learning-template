# Reference Implementations

This directory contains **simple, production-ready reference implementations** of the three core MAE engines:

1. **SimpleFRL** - Federated Reinforcement Learning (P2P policy sharing)
2. **SimpleVDN** - Value-Decomposition Networks (credit assignment)
3. **SimpleHAVEN** - Risk coordination and contagion control

---

## Quick Start

### Using SimpleFRL

```python
from implementations import create_frl_engine
from connectors.redis_client import RedisClient

redis_client = RedisClient(host="localhost", port=6379)

frl = create_frl_engine(
    agent_id="agent_001",
    redis_client=redis_client,
    config={
        "policy_update_strategy": "performance_based",
        "aggregation_method": "weighted_average",
        "max_peers": 10,
        "trust_threshold": 0.5
    }
)

# Share policy with peers
num_peers = frl.share_policy_update(
    policy_state={"param1": 0.5, "param2": [1, 2, 3]},
    metadata={"performance": 0.85}
)

# Aggregate peer updates
updated_policy = frl.aggregate_policy_updates(
    local_policy=my_policy,
    peer_updates=received_updates
)
```

### Using SimpleVDN

```python
from implementations import create_vdn_engine
from connectors.redis_client import RedisClient

redis_client = RedisClient(host="localhost", port=6379)

vdn = create_vdn_engine(
    agent_id="agent_001",
    redis_client=redis_client,
    config={
        "decomposition_method": "additive",
        "credit_strategy": "difference_rewards",
        "learning_rate": 0.01
    }
)

# Compute local value
q_value = vdn.compute_local_value(
    state=current_state,
    action=my_action,
    local_observation=my_observation
)

# Assign credit from global reward
individual_credit = vdn.assign_credit(
    global_reward=100.0,
    state=state,
    joint_action=all_actions,
    next_state=next_state
)
```

### Using SimpleHAVEN

```python
from implementations import create_haven_coordinator
from connectors.redis_client import RedisClient
from connectors.sql_logger import SQLiteLogger

redis_client = RedisClient(host="localhost", port=6379)
sql_logger = SQLiteLogger(db_path="data/logs.db")

haven = create_haven_coordinator(
    coordinator_id="haven_001",
    redis_client=redis_client,
    sql_logger=sql_logger,
    config={
        "risk_threshold": 0.7,
        "contagion_threshold": 0.3,
        "auto_intervention": True
    }
)

# Register agents
haven.register_agent("agent_001", {"team_id": "team_a"})

# Assess agent risk
risk_assessment = haven.assess_agent_risk(
    agent_id="agent_001",
    policy_state=agent_policy,
    recent_performance=[0.8, 0.75, 0.7],  # Declining
    behavioral_metrics={"exploration_rate": 0.2}
)

# Check for contagion
contagion_report = haven.detect_policy_contagion()

if contagion_report.contagion_detected:
    print(f"CONTAGION DETECTED! Infection rate: {contagion_report.infection_rate:.2%}")
```

---

## Implementation Details

### SimpleFRL (simple_frl.py)

**Features:**
- Federated averaging aggregation
- Trust-based peer selection
- Byzantine resistance through validation
- Multiple sharing strategies (broadcast, selective, performance-based, etc.)
- Multiple aggregation methods (simple average, weighted average, median, trust-weighted)

**Suitable For:**
- Small to medium deployments (10-100 agents)
- Discrete or continuous policy spaces
- P2P learning without central server

**Limitations:**
- No differential privacy
- Simple anomaly detection (not ML-based)
- No compression/communication optimization

**Extension Points:**
- Replace validation logic with ML-based detector
- Add differential privacy mechanisms
- Implement advanced aggregation (e.g., Krum, Trimmed Mean)

---

### SimpleVDN (simple_vdn.py)

**Features:**
- Additive value decomposition (Q_tot = sum Q_i)
- Simple tabular Q-learning
- Multiple credit assignment strategies (difference rewards, Shapley, counterfactual)
- TD learning for value updates
- Marginal contribution estimation

**Suitable For:**
- Small to medium state/action spaces
- Problems with clear team reward structure
- Prototyping multi-agent credit assignment

**Limitations:**
- Tabular (not scalable to large state spaces)
- No deep neural networks
- Simple decomposition (no QMIX mixing network)

**Extension Points:**
- Replace Q-table with neural network
- Implement QMIX or QTRAN decomposition
- Add attention mechanisms for dynamic weighting

---

### SimpleHAVEN (simple_haven.py)

**Features:**
- Threshold-based risk assessment
- Graph-based contagion tracking
- Multiple intervention types (monitoring, freeze, isolation, rollback, etc.)
- Performance monitoring and trend detection
- Automated interventions

**Suitable For:**
- Risk monitoring in production systems
- Adversarial robustness testing
- Policy contagion prevention

**Limitations:**
- Statistical anomaly detection (not ML-based)
- Simple graph analysis (no GNN)
- Rule-based intervention logic

**Extension Points:**
- Add ML-based anomaly detection (e.g., Isolation Forest, Autoencoders)
- Implement Graph Neural Networks for contagion detection
- Add predictive risk modeling
- Implement adaptive intervention strategies

---

## When to Use vs. Custom Implementation

### Use These Implementations When:
- Prototyping a new MAE system
- Learning how the engines work
- Testing with small-scale deployments
- You need something functional quickly

### Build Custom Implementation When:
- Large-scale production deployment (1000+ agents)
- Need advanced ML features (deep learning, attention, transformers)
- Domain-specific optimizations required
- Strict performance/latency requirements
- Advanced security requirements (differential privacy, secure aggregation)

---

## Testing

Each implementation includes comprehensive unit tests:

```bash
# Test SimpleFRL
pytest tests/unit/test_simple_frl.py

# Test SimpleVDN
pytest tests/unit/test_simple_vdn.py

# Test SimpleHAVEN
pytest tests/unit/test_simple_haven.py
```

---

## Performance Characteristics

### SimpleFRL
- **Latency**: ~10-50ms per policy share (depends on policy size)
- **Throughput**: 100-1000 shares/sec
- **Memory**: ~10MB per agent
- **Scalability**: 10-100 agents

### SimpleVDN
- **Latency**: ~1-5ms per value computation
- **Throughput**: 1000+ computations/sec
- **Memory**: ~1MB per agent (tabular), grows with state space
- **Scalability**: Works well with 5-20 agents

### SimpleHAVEN
- **Latency**: ~5-20ms per risk assessment
- **Throughput**: 100-500 assessments/sec
- **Memory**: ~5MB base + 1MB per monitored agent
- **Scalability**: 10-1000 agents

---

## Configuration Options

### SimpleFRL Config

```python
{
    "policy_update_strategy": "performance_based",  # Or: broadcast, selective, neighborhood, random
    "aggregation_method": "weighted_average",       # Or: simple_average, median, trust_weighted
    "max_peers": 10,                                # Maximum peer connections
    "trust_threshold": 0.5,                         # Minimum trust to accept updates
    "share_frequency": 10,                          # Share every N steps
    "validation_enabled": True                      # Enable policy validation
}
```

### SimpleVDN Config

```python
{
    "decomposition_method": "additive",             # Or: monotonic, weighted
    "credit_strategy": "difference_rewards",        # Or: shapley_value, counterfactual, equal_split
    "state_dim": 10,                                # State space dimension
    "action_dim": 5,                                # Number of actions
    "learning_rate": 0.01,                          # TD learning rate
    "discount_factor": 0.99,                        # Gamma
    "history_size": 1000                            # Experience buffer size
}
```

### SimpleHAVEN Config

```python
{
    "risk_threshold": 0.7,                          # Threshold for high risk
    "contagion_threshold": 0.3,                     # Infection rate alarm threshold
    "performance_window": 50,                       # Window for performance tracking
    "intervention_cooldown": 10,                    # Steps between interventions
    "auto_intervention": True                       # Enable automatic interventions
}
```

---

## FAQ

**Q: Can I use these in production?**
A: Yes, for small-medium deployments. For large-scale production, extend or replace with optimized implementations.

**Q: Do I need all three engines?**
A: No. FRL is for P2P learning, VDN for credit assignment, HAVEN for safety. Use what you need.

**Q: Can I mix these with custom implementations?**
A: Yes! They implement the abstract interfaces from `src/core/`, so they're interchangeable.

**Q: How do I extend these?**
A: Inherit from the classes and override specific methods. See the abstract interfaces in `src/core/` for all methods.

**Q: Are there neural network versions?**
A: Not in the base template. These use simple algorithms (tabular Q-learning, statistical tests). For neural networks, extend or create new implementations using PyTorch/TensorFlow.

---

## Example: Integrating with Agents

```python
from agents.specialist_agent import SpecialistAgent
from implementations import create_frl_engine, create_vdn_engine
from connectors.redis_client import RedisClient

redis_client = RedisClient()

class MySpecialist(SpecialistAgent):
    def __init__(self, unique_id, model, redis_client):
        super().__init__(unique_id, model, redis_client)

        # Add FRL engine
        self.frl_engine = create_frl_engine(
            agent_id=self.agent_id,
            redis_client=redis_client
        )

        # Add VDN engine
        self.vdn_engine = create_vdn_engine(
            agent_id=self.agent_id,
            redis_client=redis_client
        )

    def step(self):
        # Standard agent step
        super().step()

        # Use FRL for peer learning
        self.frl_engine.share_policy_update(self.policy, {"performance": self.performance})

        # Use VDN for credit assignment
        individual_credit = self.vdn_engine.assign_credit(
            global_reward=self.model.global_reward,
            state=self.current_state,
            joint_action=self.model.joint_actions,
            next_state=self.next_state
        )
```

---

## Contributing

If you develop improvements to these implementations:

1. Fork the repository
2. Create a new implementation file (e.g., `advanced_frl.py`)
3. Add tests
4. Submit a pull request

We welcome:
- Neural network versions
- Optimized algorithms
- Additional strategies/methods
- Performance improvements

---

## License

MIT License - Same as the main MAE template

---

## Support

For questions about these implementations:
- Read the docstrings in the source code
- Check the abstract interfaces in `src/core/`
- Review the main README
- Open a GitHub issue

---

**These reference implementations demonstrate the power and flexibility of the MAE architecture. Start here, learn the patterns, then build what you need for your domain.**
