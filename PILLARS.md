# The 7 Pillars of MAE: Fundamentals Guide

**Mycelial Agent Engine (MAE)** - A Decentralized Multi-Agent Learning Framework

---

## What is MAE?

MAE is a **fungal-network-inspired** multi-agent system where agents learn and collaborate peer-to-peer, without central control. Like a mycelial network in nature, agents share knowledge organically, adapt to threats, and maintain resilience through decentralization.

---

## The 7 Foundational Pillars

### Overview Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                   MYCELIAL AGENT ENGINE (MAE)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   PILLAR 1   │      │   PILLAR 2   │      │   PILLAR 3   │
│  Rule of 3   │◄────►│     FRL      │◄────►│    HAVEN     │
│   Agents     │      │ P2P Learning │      │ Risk Control │
└──────────────┘      └──────────────┘      └──────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   PILLAR 4   │      │   PILLAR 5   │      │   PILLAR 6   │
│    SQLite    │      │  Vector DB   │      │     VDN      │
│   Logging    │      │Shared Memory │      │Credit Assign │
└──────────────┘      └──────────────┘      └──────────────┘
                              │
                              ▼
                      ┌──────────────┐
                      │   PILLAR 7   │
                      │  Adversarial │
                      │  Simulator   │
                      └──────────────┘
```

---

## PILLAR 1: Rule of 3 Collaborative Agent Templates

### What It Is
A set of **specialized agent templates** that work together in teams of 3+ to accomplish tasks.

### Why It Matters
Individual agents are limited. Teams of specialized agents can tackle complex problems through collaboration and knowledge sharing.

### Core Concept: "Rule of 3"
Agents organize into teams and share policies with teammates. No agent works alone - they learn from peers and contribute to collective intelligence.

### Key Components
- **MycelialAgent**: Base class for all agents
- **SpecialistAgent**: Task execution agents (workers)
- **DataMinerAgent**: Data ingestion and preprocessing
- **RiskManagerAgent**: Safety monitoring and intervention
- **BuilderAgent**: Dynamic agent lifecycle management

### How It Works
```python
# Agents discover teammates with same team_id
teammates = agent.get_teammates()

# Share successful policies with team
policy_id = agent.share_policy_with_team()

# Learn from teammate policies
teammate_policies = agent.retrieve_teammate_policies(top_k=5)
```

### Location
`src/agents/`

---

## PILLAR 2: FRL (Federated Reinforcement Learning) - P2P Networking

### What It Is
**Peer-to-peer policy sharing** without a central coordinator. Agents share what they've learned directly with trusted peers.

### Why It Matters
Traditional federated learning requires a central server (single point of failure). FRL enables truly decentralized learning where the network continues even if individual nodes fail.

### Core Concept: "Mycelial Network"
Like fungal networks that share nutrients between trees, agents share policies through a web of peer connections. Knowledge propagates organically through the network.

### Key Features
- **Selective Sharing**: Share only with trusted peers
- **Byzantine Resistance**: Validate policies before accepting
- **Privacy-Preserving**: Share policy updates, not raw data
- **Trust Scoring**: Track peer reliability over time

### How It Works
```python
# Agent shares policy update with selected peers
peers_reached = frl_engine.share_policy_update(
    policy_state=agent.policy,
    metadata={"performance": 0.85}
)

# Aggregate updates from multiple peers
updated_policy = frl_engine.aggregate_policy_updates(
    local_policy=agent.policy,
    peer_updates=received_updates
)
```

### Location
`src/core/frl_base.py` (interface)
`src/implementations/simple_frl.py` (reference implementation)

---

## PILLAR 3: HAVEN (Risk Coordination & Contagion Control)

### What It Is
**Lightweight oversight layer** that prevents "toxic policies" (bad strategies) from spreading through the agent network.

### Why It Matters
In P2P learning, a single bad agent can poison the entire network. HAVEN detects and contains toxic behavior before it spreads.

### Core Concept: "Immune System"
HAVEN acts like an immune system, constantly monitoring for threats and intervening when risk levels spike.

### Key Features
- **Risk Assessment**: Score each agent's policy risk
- **Contagion Detection**: Track policy spread through network
- **Automated Interventions**: Freeze, isolate, or rollback risky agents
- **Adversarial Robustness**: Test system resilience to attacks

### How It Works
```python
# Assess agent risk
risk_assessment = haven.assess_agent_risk(
    agent_id="agent_042",
    policy_state=agent.policy,
    recent_performance=[0.3, 0.2, 0.1]  # Declining performance
)

# Detect contagion
contagion_report = haven.detect_policy_contagion(time_window=100)

# Execute intervention if needed
if risk_assessment.risk_level == RiskLevel.CRITICAL:
    haven.execute_intervention(
        agent_id="agent_042",
        intervention=InterventionType.FREEZE_POLICY
    )
```

### Location
`src/core/haven_base.py` (interface)
`src/implementations/simple_haven.py` (reference implementation)

---

## PILLAR 4: SQLite (Persistent Logging)

### What It Is
**Thread-safe, persistent database** for logging all agent events, patterns, performance metrics, and risk events.

### Why It Matters
Production systems need audit trails, debugging capabilities, and historical analysis. SQLite provides persistent, queryable storage without heavy infrastructure.

### Core Concept: "Black Box Recorder"
Every significant event is logged to SQLite, creating a complete record of system behavior for analysis, debugging, and compliance.

### Key Features
- **5 Database Tables**: Agent events, patterns, performance, system events, risk events
- **Thread-Safe**: Non-blocking write queue with batching
- **Auto-Flushing**: Configurable batch intervals
- **Query Interface**: Rich API for data retrieval

### How It Works
```python
# Log agent events
sql_logger.log_agent_event(
    agent_id="agent_042",
    event_type="policy_updated",
    data={"performance": 0.85}
)

# Query historical data
events = sql_logger.get_agent_events(
    agent_id="agent_042",
    event_type="policy_updated",
    limit=100
)
```

### Location
`src/connectors/sql_logger.py` (fully implemented)

---

## PILLAR 5: Vector DB (Shared Memory)

### What It Is
**Vector database for policy embeddings** that enables semantic similarity search and pattern recognition across the agent collective.

### Why It Matters
Agents need to find similar policies quickly without searching through raw data. Vector embeddings enable "semantic memory" - finding policies that work in similar situations.

### Core Concept: "Collective Memory"
All successful (and failed) policies are stored as vectors. Agents query this collective memory to learn from the experiences of others.

### Key Features
- **Semantic Search**: Find similar policies by meaning, not exact match
- **Pattern Clustering**: Identify recurring successful strategies
- **Fast Retrieval**: Sub-millisecond similarity search
- **Multiple Backends**: ChromaDB (embedded), Milvus, Qdrant

### How It Works
```python
# Store policy embedding
vector_db.add_policy_embedding(
    policy_id="policy_123",
    agent_id="agent_042",
    embedding=policy_vector,  # 128-dim vector
    metadata={"performance": 0.85}
)

# Search for similar policies
similar = vector_db.search_similar_policies(
    query_embedding=current_policy_vector,
    top_k=5
)
```

### Location
`src/connectors/vector_db.py` (interface + ChromaDB implementation)

---

## PILLAR 6: VDN (Value-Decomposition Networks) - Credit Assignment

### What It Is
**Multi-agent credit assignment** that decomposes global rewards into individual agent contributions.

### Why It Matters
In team settings, how do you know which agent contributed most to success? VDN solves the "credit assignment problem" by fairly distributing rewards.

### Core Concept: "Fair Credit"
Global system reward is decomposed into individual contributions. Each agent knows exactly how much they helped (or hurt) the team goal.

### Key Methods
- **Additive (VDN)**: Q_total = Q_1 + Q_2 + ... + Q_n
- **Monotonic (QMIX)**: Q_total = mix(Q_1, Q_2, ..., Q_n)
- **Shapley Value**: Game-theoretic fair allocation
- **Counterfactual**: What would have happened without this agent?

### How It Works
```python
# Compute individual contribution
local_reward = vdn_engine.assign_credit(
    global_reward=100.0,  # System reward
    state=current_state,
    joint_action=all_agent_actions,
    next_state=next_state
)

# Agent uses local reward for learning
agent.update_policy(local_reward)
```

### Location
`src/core/vdn_base.py` (interface)
`src/implementations/simple_vdn.py` (reference implementation)

---

## PILLAR 7: Adversarial Simulator (with Memory Builder)

### What It Is
**Safety testing framework** that injects toxic agents and validates that HAVEN can detect and contain them.

### Why It Matters
You CANNOT deploy a multi-agent learning system without adversarial testing. This pillar ensures your system is resilient to attacks.

### Core Concept: "Safety First"
Before production, run simulations with toxic agents (policy poisoning, Byzantine behavior, sabotage). If HAVEN fails to contain them, fix the system.

### Key Components
- **ToxicAgent**: Agents with malicious behaviors
- **AdversarialModel**: Test environment
- **MemoryBuilder**: Generates initial knowledge base from simulations
- **SimulationRunner**: Orchestrates safety tests

### Toxic Behaviors
- **Policy Poisoning**: Shares bad policies
- **Byzantine**: Random/unpredictable behavior
- **Sabotage**: Actively harms system
- **High Risk**: Takes excessive risks
- **Manipulative**: Subtle attacks

### How It Works
```python
# Run adversarial test
runner = SimulationRunner(
    num_healthy=10,
    num_toxic=2,
    toxic_behavior=ToxicBehaviorType.POLICY_POISONING
)

results = runner.run_full_test(num_steps=1000)

# Check if system passed
if results["test_passed"]:
    print("✓ System contained toxic agents")
else:
    print("✗ System failed - toxic policies spread")
```

### Location
`src/simulation/` (fully implemented)

---

## How The 7 Pillars Work Together

### Data Flow

```
1. External Data → Redis Streams
                    ↓
2. DataMinerAgent processes data → Redis Pub/Sub
                    ↓
3. SpecialistAgents receive tasks
                    ↓
4. Agents execute actions and learn
                    ↓
5. FRL: Agents share policies P2P
                    ↓
6. Vector DB: Policies stored as embeddings
                    ↓
7. VDN: Global reward → Individual credits
                    ↓
8. HAVEN: Risk assessment and intervention
                    ↓
9. SQLite: All events logged persistently
```

### Example: Single Agent Step

```python
def agent_step():
    # 1. Observe environment
    observation = agent.observe()

    # 2. Select action using policy
    action = agent.select_action(observation)

    # 3. Execute action
    reward = agent.execute_action(action)

    # 4. Share policy with team (FRL + Vector DB)
    agent.share_policy_with_team()

    # 5. Learn from teammates (Vector DB)
    teammate_policies = agent.retrieve_teammate_policies()
    agent.incorporate_teammate_policies(teammate_policies)

    # 6. Get individual credit (VDN)
    local_reward = agent.get_local_reward(global_reward)

    # 7. Update policy
    agent.update_policy(local_reward)

    # 8. Risk assessment (HAVEN)
    risk_score = haven.assess_agent_risk(agent.id, agent.policy)

    # 9. Log everything (SQLite)
    sql_logger.log_agent_event(agent.id, "step_complete", {...})
```

---

## Implementation Checklist

When building your MAE system, ensure all 7 pillars are functional:

- [ ] **Pillar 1**: Agent templates customized for your domain
- [ ] **Pillar 2**: FRL engine implemented (or use SimpleFRL)
- [ ] **Pillar 3**: HAVEN coordinator implemented (or use SimpleHAVEN)
- [ ] **Pillar 4**: SQLite logger initialized and logging events
- [ ] **Pillar 5**: Vector DB initialized with policy embeddings
- [ ] **Pillar 6**: VDN engine implemented (or use SimpleVDN)
- [ ] **Pillar 7**: Adversarial tests PASS before production

---

## Quick Reference Table

| Pillar | Purpose | Status | Location |
|--------|---------|--------|----------|
| **1. Agents** | Task execution & collaboration | Template | `src/agents/` |
| **2. FRL** | P2P policy sharing | Interface | `src/core/frl_base.py` |
| **3. HAVEN** | Risk & contagion control | Interface | `src/core/haven_base.py` |
| **4. SQLite** | Persistent logging | Implemented | `src/connectors/sql_logger.py` |
| **5. Vector DB** | Shared memory | Implemented | `src/connectors/vector_db.py` |
| **6. VDN** | Credit assignment | Interface | `src/core/vdn_base.py` |
| **7. Simulator** | Safety testing | Implemented | `src/simulation/` |

---

## Architecture Philosophy

MAE is designed around **3 core principles**:

### 1. Decentralization
No single point of failure. Agents communicate P2P. System continues even if individual components fail.

### 2. Safety First
Adversarial testing is mandatory. HAVEN prevents toxic policies from spreading. Every action is logged for audit.

### 3. Adaptability
Template design allows customization for any domain (finance, logistics, science, etc.) while maintaining core safety guarantees.

---

## Next Steps

1. **Read**: Full README for detailed implementation guide
2. **Explore**: Example implementations in `examples/` directory
3. **Implement**: Follow Step 1-4 in README to customize for your domain
4. **Test**: Run adversarial simulations before production
5. **Deploy**: Launch with monitoring and logging enabled

---

## Further Reading

- **README.md**: Complete implementation guide
- **ARCHITECTURE.md**: Deep dive into system design
- **examples/**: Complete domain-specific implementations
- **src/implementations/**: Reference FRL/VDN/HAVEN implementations

---

**The 7 Pillars work together to create a resilient, decentralized, self-learning multi-agent system. Master these fundamentals, and you can build agent systems for any domain.**
