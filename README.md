# Mycelial Agent Engine (MAE) Template

**MAE** is a generic, production-ready framework for building **decentralized, resilient, and self-learning multi-agent systems**. It is based on a "mycelial" P2P learning architecture that avoids the single points of failure found in traditional hierarchical systems.

Unlike conventional multi-agent frameworks that rely on centralized coordination, MAE implements a **fungal-network-inspired architecture** where:
- Agents share knowledge peer-to-peer (no central server bottleneck)
- Learning propagates organically through the network
- The system is resilient to individual agent failures
- Risk management prevents "toxic" policies from spreading

---

## Quick Links

- **[PILLARS.md](PILLARS.md)** - 7 Pillars Fundamentals Guide (START HERE)
- **[Reference Implementations](src/implementations/)** - SimpleFRL, SimpleVDN, SimpleHAVEN
- **[Examples](examples/)** - Complete domain-specific implementations
- **[Requirements](requirements.txt)** - Installation dependencies

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Key Features](#key-features)
3. [Project Structure](#project-structure)
4. [How to Use This Template](#how-to-use-this-template)
5. [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)
6. [Running the Adversarial Simulation](#running-the-adversarial-simulation)
7. [Production Deployment](#production-deployment)
8. [API Reference](#api-reference)
9. [Troubleshooting](#troubleshooting)
10. [Contributing](#contributing)

---

## Architecture Overview

MAE combines three cutting-edge multi-agent learning paradigms:

### 1. **Federated Reinforcement Learning (FRL)** - P2P Policy Sharing
Agents share policy updates in a peer-to-peer "mycelial" network without a central aggregator. This enables:
- Decentralized collaborative learning
- No single point of failure
- Byzantine-resistant trust systems
- Privacy-preserving policy exchange

**Core Component**: `/src/core/frl_base.py`
**Reference Implementation**: `/src/implementations/simple_frl.py`

### 2. **Value-Decomposition Networks (VDN)** - Credit Assignment
Solves the multi-agent credit assignment problem by decomposing global rewards into individual agent contributions:
- Fair credit distribution
- Individual agent accountability
- Scalable to large agent populations
- Supports both additive (VDN) and monotonic (QMIX) decomposition

**Core Component**: `/src/core/vdn_base.py`
**Reference Implementation**: `/src/implementations/simple_vdn.py`

### 3. **HAVEN Framework** - Risk Coordination
Lightweight oversight layer that prevents "policy contagion" (bad policies spreading through the network):
- Real-time risk assessment
- Contagion detection and containment
- Automated interventions
- Adversarial robustness testing

**Core Component**: `/src/core/haven_base.py`
**Reference Implementation**: `/src/implementations/simple_haven.py`

### System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Redis Backbone                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Pub/Sub    │  │   Streams    │  │  Key-Value   │      │
│  │ (Messages)   │  │   (Data)     │  │   (State)    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
           │                  │                  │
           ▼                  ▼                  ▼
┌──────────────────────────────────────────────────────────────┐
│                    Agent Ecosystem                           │
│                                                              │
│  ┌─────────────┐      ┌──────────────────┐                  │
│  │ DataMiner   │─────▶│ SpecialistAgent  │◀─┐               │
│  │   Agent     │      │   (Worker)       │  │ FRL           │
│  └─────────────┘      └──────────────────┘  │ Policy        │
│        │                      │              │ Sharing       │
│        │ Streams              │ VDN          │               │
│        │                      │ Credit       │               │
│        ▼                      ▼              ▼               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │          RiskManagerAgent (HAVEN)                    │   │
│  │  - Monitors all agents for toxic behavior            │   │
│  │  - Detects policy contagion                          │   │
│  │  - Executes interventions                            │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

---

## Key Features

### Core Capabilities
- ✅ **Decentralized Learning**: No central coordinator required
- ✅ **Redis-Backed**: All state, communication, and data flow through Redis
- ✅ **Byzantine Resilient**: Trust-based peer selection and validation
- ✅ **Risk-Aware**: Automated detection and containment of toxic policies
- ✅ **Production-Ready**: State persistence, error handling, graceful shutdown
- ✅ **Adversarial Testing**: Built-in framework for safety validation

### Technical Stack
- **Mesa**: Agent-based modeling framework
- **Redis**: Distributed data backbone (Streams, Pub/Sub, KV)
- **Python 3.8+**: Core implementation language
- **NumPy**: Numerical computations

### Scalability
- Horizontal scaling via Redis clustering
- Agents can be distributed across multiple machines
- Lightweight coordination overhead
- Supports 10s to 1000s of agents

---

## Project Structure

```
agent-learning-template-codebase/
│
├── README.md                       # This file
├── requirements.txt                # Python dependencies (you need to create this)
│
├── run_simulation.py               # Launch production simulation
├── run_live.py                     # Launch live/production mode
│
├── src/
│   ├── core/                       # Core learning engines (ABSTRACT)
│   │   ├── model.py                # Main Mesa model
│   │   ├── frl_base.py             # Federated RL interface
│   │   ├── vdn_base.py             # Value decomposition interface
│   │   └── haven_base.py           # Risk coordination interface
│   │
│   ├── connectors/                 # Data I/O
│   │   └── redis_client.py         # Redis operations wrapper
│   │
│   ├── agents/                     # Agent templates
│   │   ├── base_agent.py           # MycelialAgent base class
│   │   ├── data_miner_agent.py     # Data ingestion agent
│   │   ├── specialist_agent.py     # Main worker agent
│   │   └── risk_manager_agent.py   # HAVEN oversight agent
│   │
│   └── simulation/                 # Adversarial testing
│       ├── adversarial_model.py    # Safety test model
│       ├── toxic_agent.py          # Toxic behavior agents
│       └── simulation_runner.py    # Test orchestration script
│
└── simulation_results/             # Test outputs (auto-generated)
```

---

## How to Use This Template

MAE is a **template**, not a finished product. To deploy this engine for your specific domain (e.g., finance, logistics, medical research, cybersecurity, etc.), you must implement **4 key areas**:

### Quick Start Checklist

- [ ] **Step 1**: Define Your Data Connectors
- [ ] **Step 2**: Implement Your Custom Agents
- [ ] **Step 3**: Define Your Reward Function
- [ ] **Step 4**: Run the Adversarial Simulation (CRITICAL)

---

## Step-by-Step Implementation Guide

### Prerequisites

1. **Install Redis**:
   ```bash
   # macOS
   brew install redis
   redis-server

   # Ubuntu/Debian
   sudo apt-get install redis-server
   sudo systemctl start redis

   # Windows
   # Download from https://redis.io/download
   # Or use WSL/Docker
   ```

2. **Install Python Dependencies**:
   ```bash
   pip install mesa redis numpy
   ```

3. **Verify Redis Connection**:
   ```bash
   redis-cli ping
   # Should return: PONG
   ```

---

### Step 1: Define Your Data Connectors (`/src/connectors/`)

Your agents need data. By default, they read from a Redis Stream.

**You must build your own "producer" script** that fetches your domain-specific data and writes it into the ingestion stream in Redis.

#### Example: Stock Market Data Connector

Create `src/connectors/stock_market_producer.py`:

```python
"""
Example: Stock market data producer for finance domain
"""
import time
import requests
from redis_client import RedisClient

def fetch_stock_data(symbol):
    """Fetch real-time stock data from API"""
    # Replace with your actual API
    response = requests.get(f"https://api.example.com/quote/{symbol}")
    return response.json()

def main():
    redis_client = RedisClient(host="localhost", port=6379)

    symbols = ["AAPL", "GOOGL", "MSFT"]

    while True:
        for symbol in symbols:
            data = fetch_stock_data(symbol)

            # Write to Redis Stream
            redis_client.write_to_stream(
                stream_name="stock_data_stream",
                data={
                    "symbol": symbol,
                    "price": data["price"],
                    "volume": data["volume"],
                    "timestamp": time.time()
                }
            )

        time.sleep(1)  # Update every second

if __name__ == "__main__":
    main()
```

#### Other Domain Examples

**Logistics Domain**:
```python
# src/connectors/delivery_tracker.py
# Stream: "delivery_events_stream"
# Data: {truck_id, location, eta, cargo_weight, ...}
```

**Scientific Research Domain**:
```python
# src/connectors/sensor_reader.py
# Stream: "sensor_readings_stream"
# Data: {sensor_id, temperature, pressure, measurement, ...}
```

**Cybersecurity Domain**:
```python
# src/connectors/network_monitor.py
# Stream: "network_events_stream"
# Data: {src_ip, dst_ip, protocol, packet_size, flags, ...}
```

---

### Step 2: Implement Your Custom Agents (`/src/agents/`)

Inherit from the provided templates and define your agent's "brain".

#### 2.1: Customize the DataMinerAgent

Create `src/agents/my_data_miner.py`:

```python
from agents.data_miner_agent import DataMinerAgent

class StockDataMiner(DataMinerAgent):
    """
    Custom data miner for stock market data
    """

    def __init__(self, unique_id, model, redis_client):
        super().__init__(
            unique_id=unique_id,
            model=model,
            redis_client=redis_client,
            source_streams=["stock_data_stream"],  # Your stream name
            output_channel="processed_stock_data",
            agent_config={"batch_size": 10}
        )

    def _validate_data(self, data):
        """Validate stock data"""
        required_fields = ["symbol", "price", "volume"]
        return all(field in data for field in required_fields)

    def _transform_data(self, record):
        """Transform raw stock data into features"""
        data = record["data"]

        # Calculate additional features
        price = float(data["price"])
        volume = float(data["volume"])

        return {
            "symbol": data["symbol"],
            "price": price,
            "volume": volume,
            "value": price * volume,  # Market value
            "timestamp": record["timestamp"]
        }
```

#### 2.2: Customize the SpecialistAgent (MOST IMPORTANT)

This is your core agent that makes decisions. **Define its goal and logic here.**

Create `src/agents/my_specialist.py`:

```python
from agents.specialist_agent import SpecialistAgent
import numpy as np

class TradingSpecialist(SpecialistAgent):
    """
    Example: Stock trading agent
    Goal: Maximize profit by buying/selling stocks
    """

    def __init__(self, unique_id, model, redis_client):
        super().__init__(
            unique_id=unique_id,
            model=model,
            redis_client=redis_client,
            data_channel="processed_stock_data",
            specialization="trading",
            agent_config={
                "learning_rate": 0.01,
                "exploration_rate": 0.2
            }
        )

        # Domain-specific state
        self.portfolio = {}  # {symbol: quantity}
        self.cash = 10000.0  # Starting capital

    def _extract_state_from_data(self, task_data):
        """Extract trading state from market data"""
        return {
            "symbol": task_data["symbol"],
            "price": task_data["price"],
            "volume": task_data["volume"],
            "portfolio": self.portfolio.copy(),
            "cash": self.cash
        }

    def _get_policy_action(self, state):
        """
        YOUR CORE LOGIC HERE
        Decide: BUY (1), SELL (-1), or HOLD (0)
        """
        symbol = state["symbol"]
        price = state["price"]

        # Example simple strategy (replace with your ML model)
        if price < 100 and self.cash > price:
            return 1  # BUY
        elif symbol in self.portfolio and self.portfolio[symbol] > 0:
            return -1  # SELL
        else:
            return 0  # HOLD

    def _execute_action(self, action):
        """Execute trading action and calculate reward"""
        # Implement your trading logic
        # action: 1=BUY, -1=SELL, 0=HOLD

        # Calculate profit/loss as reward
        reward = 0.0

        # Your trading execution logic here...

        return reward
```

#### Other Domain Examples

**Logistics Specialist**:
```python
class RouteOptimizerAgent(SpecialistAgent):
    """
    Goal: Minimize delivery time and fuel costs
    Actions: Route selection, speed optimization
    """
    pass
```

**Medical Research Specialist**:
```python
class DrugDiscoveryAgent(SpecialistAgent):
    """
    Goal: Find promising drug compounds
    Actions: Select molecules to synthesize/test
    """
    pass
```

---

### Step 3: Define Your Reward Function (`/src/core/model.py`)

The "win condition" for your agents. Edit `model.py` to define what success means.

Open `/src/core/model.py` and modify the reward calculation:

```python
class RedisBackedModel(Model):
    # ... existing code ...

    def step(self):
        """Execute one step of the model"""
        self.current_step += 1
        self.schedule.step()

        # ========================================
        # YOUR CUSTOM REWARD LOGIC HERE
        # ========================================

        global_reward = self._calculate_global_reward()

        # Distribute reward to agents using VDN
        for agent in self.schedule.agents:
            if hasattr(agent, 'get_local_reward'):
                local_reward = agent.get_local_reward(global_reward)
                # Agent will use this for learning

        self._save_model_state()

    def _calculate_global_reward(self):
        """
        Define your system-level success metric

        Examples by domain:
        - Finance: Total portfolio value increase
        - Logistics: Number of on-time deliveries
        - Science: Number of valid discoveries
        - Cybersecurity: Threats detected / false positives
        """

        # Example for trading domain:
        total_portfolio_value = 0.0
        for agent in self.schedule.agents:
            if hasattr(agent, 'portfolio'):
                # Calculate agent's portfolio value
                pass

        # Reward = increase in total value
        reward = total_portfolio_value - self.previous_value
        self.previous_value = total_portfolio_value

        return reward
```

#### Domain-Specific Reward Examples

**Logistics**:
```python
def _calculate_global_reward(self):
    on_time_deliveries = sum(a.deliveries_on_time for a in self.agents)
    fuel_cost = sum(a.fuel_used for a in self.agents)
    return on_time_deliveries * 100 - fuel_cost
```

**Medical Research**:
```python
def _calculate_global_reward(self):
    valid_compounds = sum(a.compounds_discovered for a in self.agents)
    synthesis_cost = sum(a.experiments_run * 1000 for a in self.agents)
    return valid_compounds * 10000 - synthesis_cost
```

**Cybersecurity**:
```python
def _calculate_global_reward(self):
    threats_detected = sum(a.threats_found for a in self.agents)
    false_positives = sum(a.false_alarms for a in self.agents)
    return threats_detected * 10 - false_positives * 2
```

---

### Step 4: Implement the Learning Engines (ADVANCED)

The three core engines (`frl_base.py`, `vdn_base.py`, `haven_base.py`) are **abstract base classes**. For production use, you need concrete implementations.

#### Option A: Use Simple Implementations (Recommended for Prototyping)

Create simplified versions that implement the abstract methods with basic logic. See the `MockHavenCoordinator` in `simulation_runner.py` as an example.

#### Option B: Implement Full Machine Learning (Production)

Implement the abstract methods with:
- **FRL**: Federated averaging, gradient aggregation, differential privacy
- **VDN**: Neural network value functions, Q-learning, policy gradients
- **HAVEN**: Anomaly detection, graph neural networks for contagion detection

This requires ML expertise and is beyond the scope of this README. Consider starting with Option A and upgrading incrementally.

---

## Running the Adversarial Simulation

### ⚠️ CRITICAL: DO NOT DEPLOY TO PRODUCTION UNTIL YOU PASS THIS TEST ⚠️

The adversarial simulation validates that your system can detect and contain toxic agents.

### Step 1: Define Your Domain-Specific Toxic Agent

Open `/src/simulation/toxic_agent.py` and create a custom toxic behavior for your domain:

```python
# Add your domain-specific toxic behavior
class ToxicBehaviorType:
    # ... existing behaviors ...

    FLASH_CRASH = "flash_crash"        # Finance: Causes market crash
    ROUTE_SABOTAGE = "route_sabotage"  # Logistics: Creates delivery chaos
    DATA_CORRUPTION = "data_corruption" # Science: Corrupts experimental data
```

Then implement the toxic action:

```python
class ToxicAgent(SpecialistAgent):
    # ... existing code ...

    def _get_toxic_action(self, state):
        if self.toxic_behavior == "flash_crash":
            # Sell everything at once (causes price crash)
            return -1000  # Massive sell order

        elif self.toxic_behavior == "route_sabotage":
            # Send trucks to wrong locations
            return "wrong_destination"

        # ... etc
```

### Step 2: Run the Simulation

```bash
cd src/simulation
python simulation_runner.py
```

This will:
1. Create 10 healthy specialist agents
2. Inject 2 toxic agents with bad strategies
3. Run for 1,000 steps
4. Monitor for policy contagion
5. Generate a detailed report

### Step 3: Interpret the Results

The simulation will output a report like this:

```
=== Adversarial Simulation Report ===
Scenario: policy_contagion
Result: PASSED ✓  or  FAILED ✗

Contagion Detection:
- Detected: Yes/No
- Infection Rate: 20.0%
- Containment Rate: 80.0%

System Response:
- Interventions: 3
- Response Time: 127 steps
```

#### Success Criteria (Test PASSED)
✅ Contagion was **detected** (HAVEN is working)
✅ Interventions were **triggered** (Risk manager responded)
✅ Infection rate **< 30%** (System contained the threat)

#### Failure Criteria (Test FAILED)
❌ Contagion was **not detected** (HAVEN blind to threats)
❌ No interventions **triggered** (Risk manager inactive)
❌ Infection rate **> 30%** (Toxic policy spread too widely)

### Step 4: Tune Your System

If the test **FAILED**, you need to improve your `RiskManagerAgent`:

1. **Lower the risk threshold** in `/src/agents/risk_manager_agent.py`:
   ```python
   self.risk_threshold = 0.5  # Was 0.7, now more sensitive
   ```

2. **Increase monitoring frequency**:
   ```python
   self.monitoring_interval = 3  # Was 5, now checks more often
   ```

3. **Enable more aggressive interventions**:
   ```python
   self.auto_intervention = True
   ```

4. **Re-run the simulation** until it passes.

### Step 5: Test Different Scenarios

Run multiple adversarial scenarios to ensure robustness:

```python
# Test different toxic behaviors
runner.run_full_test(toxic_behavior=ToxicBehaviorType.HIGH_RISK)
runner.run_full_test(toxic_behavior=ToxicBehaviorType.MANIPULATIVE)
runner.run_full_test(toxic_behavior=ToxicBehaviorType.BYZANTINE)

# Test with more toxic agents
runner.run_full_test(num_healthy=10, num_toxic=5)  # 50% toxic

# Test longer simulations
runner.run_full_test(num_steps=5000)
```

**Your system is production-ready when it passes ALL scenarios.**

---

## Production Deployment

### Running in Simulation Mode

For controlled testing and training:

```bash
python run_simulation.py
```

This runs the system for a fixed number of steps and exits.

### Running in Live/Production Mode

For continuous operation with real data:

```bash
python run_live.py
```

This runs indefinitely, processing real-time data streams.

### Production Checklist

Before deploying to production:

- [ ] All adversarial tests **PASS**
- [ ] Redis is configured with persistence (RDB or AOF)
- [ ] Monitoring and logging are configured
- [ ] Error handling is tested
- [ ] Graceful shutdown works (Ctrl+C)
- [ ] You have a backup/recovery plan
- [ ] You can rollback agents to safe policies

### Scaling to Production

#### Single Machine
```bash
# Start Redis
redis-server

# Start your data producer
python src/connectors/your_producer.py &

# Start the main system
python run_live.py
```

#### Distributed Deployment

1. **Redis Cluster**: Set up Redis in cluster mode for high availability
2. **Multiple Agent Processes**: Run agents on different machines, all connecting to the same Redis cluster
3. **Load Balancing**: Distribute agents across machines based on load
4. **Monitoring**: Use Redis monitoring tools and log aggregation

---

## API Reference

### RedisClient

```python
from connectors.redis_client import RedisClient

client = RedisClient(host="localhost", port=6379)

# Pub/Sub
client.publish(channel="my_channel", message={"data": "value"})
client.subscribe(["channel1", "channel2"])

# Streams
client.write_to_stream(stream_name="my_stream", data={"field": "value"})
entries = client.read_from_stream(stream_name="my_stream", last_id="0-0")

# Key-Value
client.set_key_value(key="agent:state:1", value={"state": "data"})
state = client.get_key_value(key="agent:state:1")
```

### MycelialAgent

```python
from agents.base_agent import MycelialAgent

class MyAgent(MycelialAgent):
    def step(self):
        # Your agent logic
        pass

    def share_policy(self):
        # Share with peers
        return super().share_policy()

    def get_local_reward(self, global_reward):
        # Get individual credit
        return super().get_local_reward(global_reward)
```

### SpecialistAgent

```python
from agents.specialist_agent import SpecialistAgent

class MySpecialist(SpecialistAgent):
    def _select_action(self, state):
        # Your decision logic
        return action

    def _execute_action(self, action):
        # Your action execution
        return reward
```

---

## Troubleshooting

### Issue: Redis Connection Refused

**Solution**:
```bash
# Check if Redis is running
redis-cli ping

# If not, start Redis
redis-server

# Check Redis logs
tail -f /var/log/redis/redis-server.log
```

### Issue: Agents Not Receiving Data

**Solution**:
1. Verify your data producer is writing to the correct stream:
   ```bash
   redis-cli XLEN your_stream_name
   ```
2. Check that DataMinerAgent is subscribed to the correct stream
3. Verify channel names match between publisher and subscriber

### Issue: Adversarial Simulation Always Fails

**Solution**:
1. Check that RiskManagerAgent is being created and added to the model
2. Verify `monitoring_interval` is not too large
3. Lower `risk_threshold` to make detection more sensitive
4. Add logging to see what's happening:
   ```python
   logger.setLevel(logging.DEBUG)
   ```

### Issue: Performance Degradation

**Solution**:
1. Limit history sizes (already implemented in agents)
2. Use Redis pipelining for bulk operations
3. Reduce monitoring frequency if not needed
4. Consider Redis clustering for horizontal scaling

### Issue: Policy Contagion Not Detected

**Solution**:
1. Ensure toxic agents are actually sharing policies (`share_policy()` is called)
2. Verify healthy agents have FRL engines configured
3. Check that peer connections are established
4. Increase toxicity level of toxic agents for testing

---

## Advanced Topics

### Implementing Custom FRL Engine

```python
from core.frl_base import FederatedLearningEngine

class MyFRLEngine(FederatedLearningEngine):
    def share_policy_update(self, policy_state, metadata):
        # Implement federated averaging
        pass

    def aggregate_policy_updates(self, local_policy, peer_updates):
        # Implement aggregation logic
        pass

    # ... implement all abstract methods
```

### Implementing Custom VDN Engine

```python
from core.vdn_base import ValueDecompositionEngine

class MyVDNEngine(ValueDecompositionEngine):
    def compute_local_value(self, state, action, local_observation):
        # Neural network forward pass
        pass

    def assign_credit(self, global_reward, state, joint_action, next_state):
        # Credit assignment algorithm
        pass

    # ... implement all abstract methods
```

### Implementing Custom HAVEN Coordinator

```python
from core.haven_base import HavenRiskCoordinator

class MyHavenCoordinator(HavenRiskCoordinator):
    def assess_agent_risk(self, agent_id, policy_state, recent_performance, behavioral_metrics):
        # Risk assessment algorithm
        pass

    def detect_policy_contagion(self, time_window):
        # Contagion detection using graph analysis
        pass

    # ... implement all abstract methods
```

---

## Example Use Cases

### 1. Algorithmic Trading System
- **Data**: Stock prices, volumes, order books
- **Agents**: Trading specialists with different strategies
- **Reward**: Portfolio value increase
- **Risk**: Flash crash detection, market manipulation prevention

### 2. Autonomous Logistics Network
- **Data**: Delivery requests, traffic, vehicle locations
- **Agents**: Route optimization, load balancing, ETA prediction
- **Reward**: On-time deliveries, fuel efficiency
- **Risk**: Route sabotage detection, collision avoidance

### 3. Distributed Scientific Research
- **Data**: Experimental results, sensor readings
- **Agents**: Hypothesis generation, experiment design
- **Reward**: Valid discoveries, experimental efficiency
- **Risk**: Data corruption detection, spurious correlation prevention

### 4. Cybersecurity Defense
- **Data**: Network traffic, logs, threat intelligence
- **Agents**: Intrusion detection, response automation
- **Reward**: Threats detected, false positives minimized
- **Risk**: Adversarial evasion detection, coordinated attack response

---

## Performance Benchmarks

Typical performance on modest hardware (tested on MacBook Pro M1):

| Metric | Value |
|--------|-------|
| Agents | 10-50 |
| Steps/second | 100-500 |
| Redis operations/sec | 10,000+ |
| Memory per agent | ~10 MB |
| Startup time | < 5 seconds |

For larger deployments (100+ agents), use Redis clustering and distributed agent processes.

---

## Contributing

This is a template project. Contributions that improve the base framework are welcome:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

**Areas for contribution**:
- Concrete implementations of FRL/VDN/HAVEN engines
- Additional toxic agent behaviors
- Performance optimizations
- Documentation improvements
- Example domain implementations

---

## License

MIT License - Feel free to use this template for commercial or personal projects.

---

## Citation

If you use MAE in research, please cite:

```bibtex
@software{mycelial_agent_engine,
  title = {Mycelial Agent Engine: A Decentralized Multi-Agent Learning Framework},
  year = {2025},
  author = {Your Name},
  url = {https://github.com/yourusername/mae}
}
```

---

## Support

For questions and support:
- GitHub Issues: [Report bugs or request features]
- Documentation: [Read the full docs]
- Community: [Join the discussion]

---

## Acknowledgments

MAE builds on research in:
- Federated Learning (McMahan et al., 2017)
- Value Decomposition Networks (Sunehag et al., 2018)
- Multi-Agent Reinforcement Learning (MARL)
- Byzantine-Resilient Distributed Systems

---

**Ready to build your multi-agent system? Start with Step 1! 🚀**
