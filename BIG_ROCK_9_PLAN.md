# Big Rock 9: Episodic Memory & Replay

**Project:** Mycelial Agent Engine (MAE) v3.0
**Phase:** Phase 2 - Week 6 (Days 36-42) - Intelligence
**Author:** MAE Development Team
**Date:** 2025-11-12
**Status:** Planning Phase

---

## Executive Summary

Big Rock 9 implements **Episodic Memory with Prioritized Experience Replay**, inspired by human hippocampal replay and memory consolidation. This enables agents to:
1. Store important experiences for repeated learning
2. Perform offline learning during "sleep" phases
3. Prevent catastrophic forgetting through strategic replay
4. Achieve better data efficiency through prioritized sampling

**Key Innovation:** Unlike simple experience buffers, our episodic memory uses priority-based sampling, semantic indexing via Vector DB, and memory consolidation phases to mimic human learning. This complements Big Rock 8 (Transfer Learning) by providing the memory substrate for continual learning.

**Performance Target:**
- 2-5x better data efficiency through prioritized replay
- 90%+ retention of important experiences
- Prevent catastrophic forgetting on sequential tasks
- Support 100K+ episodes with fast retrieval (<10ms)

---

## The Memory Problem in RL

### Why Episodic Memory Matters

Standard RL agents learn from each experience once, then discard it. This leads to:

1. **Data Inefficiency**: Important experiences (rare events, high rewards) are only seen once
2. **Catastrophic Forgetting**: Learning new tasks overwrites knowledge from old tasks
3. **No Offline Learning**: Agents can't improve when not interacting with environment
4. **Poor Sample Complexity**: Requires millions of samples for simple tasks

**Example:**
```
Agent finds rare high-reward state → Learns from it once → Never sees it again ❌
With Replay: Stores experience → Replays it 100x → Strong learning ✅
```

### Human Episodic Memory

Humans have specialized memory systems:
- **Hippocampus**: Stores episodic memories (specific events)
- **Memory Consolidation**: Replay during sleep strengthens important memories
- **Prioritization**: Emotional/important events remembered better
- **Semantic Indexing**: Retrieve similar past experiences

**Our Goal:** Implement computational equivalents of these mechanisms.

---

## Research Foundation

### 1. Prioritized Experience Replay (PER)

**Paper:** "Prioritized Experience Replay" (Schaul et al., 2016, ICLR)

**Core Idea:** Sample experiences based on their importance (TD error) rather than uniformly.

**Priority Calculation:**
```python
# TD error as priority
priority = |r + γ max_a' Q(s', a') - Q(s, a)| + ε

# Probability of sampling experience i:
P(i) = priority_i^α / Σ_k priority_k^α

# Importance-sampling weights (correct bias):
w_i = (1/N · 1/P(i))^β
```

**Key Parameters:**
- `α`: How much prioritization (0 = uniform, 1 = greedy)
- `β`: Importance sampling correction (0 = none, 1 = full)
- `ε`: Small constant to ensure non-zero probability

**Benefits:**
- **2-3x faster convergence** in DQN
- Rare/important experiences replayed more
- Better use of limited memory

**Implementation Detail:** Use sum-tree data structure for O(log N) sampling.

### 2. Hindsight Experience Replay (HER)

**Paper:** "Hindsight Experience Replay" (Andrychowicz et al., 2017, NeurIPS)

**Core Idea:** Learn from failures by relabeling goals.

**Example:**
```
Original goal: Reach (10, 10)
Actual outcome: Reached (7, 8) - FAILURE

HER: Store as success for goal "Reach (7, 8)"
→ Learn from every experience, even failures!
```

**Relevance:** Shows power of creative replay strategies.

### 3. Episodic Control

**Paper:** "Model-Free Episodic Control" (Blundell et al., 2016, arXiv)

**Core Idea:** Store successful state-action-value tuples, retrieve via nearest-neighbor search.

**Memory Structure:**
```
Episodic Buffer: [(s, a, r, s'), ...]
   ↓ Nearest-neighbor lookup
Q(s, a) ≈ max_i∈neighbors r_i
```

**Benefits:**
- **Instant learning**: No gradient updates needed
- **Never forgets**: Experiences stored permanently
- Leverages Vector DB for fast similarity search

**Our Integration:** Use Vector DB (Big Rock 8) for nearest-neighbor retrieval.

### 4. Memory Consolidation During Sleep

**Paper:** "The Role of Sleep in Memory Consolidation" (Stickgold, 2005, Nature)

**Biological Inspiration:**
- Brain replays experiences during sleep
- Strengthens important memories
- Integrates new knowledge with old

**Computational Equivalent:**
```python
# Offline learning phase ("sleep")
for _ in range(consolidation_steps):
    batch = memory.replay_batch(priority='high')
    agent.learn(batch)
```

**Benefits:**
- Offline improvement without environment interaction
- Strengthens weak memories
- Prevents catastrophic forgetting

### 5. Semantic Memory Organization

**Paper:** "Building machines that learn and think like people" (Lake et al., 2017, BBS)

**Core Idea:** Organize memories semantically, not just chronologically.

**Our Approach:**
- Store episode embeddings in Vector DB
- Retrieve by similarity (not just recency)
- Enable "what-if" queries: "What happened in similar states?"

**Integration with Big Rock 8:**
- Reuse TaskEmbedding for episode encoding
- Leverage existing Vector DB infrastructure
- Semantic search for experience retrieval

---

## Architecture Design

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                   Continual Learning System                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────┐  ┌────────────────────┐                │
│  │  Task Memory       │  │  Forgetting        │                │
│  │  Manager           │  │  Detector          │                │
│  │                    │  │                    │                │
│  │  - Episode buffer  │  │  - Monitor perf    │                │
│  │  - Replay strategy │  │  - Trigger replay  │                │
│  │  - Prioritization  │  │  - Alert on drop   │                │
│  └────────────────────┘  └────────────────────┘                │
│           │                        │                            │
│           └────────────┬───────────┘                            │
│                        ▼                                        │
│  ┌──────────────────────────────────────────┐                  │
│  │     Continual Learner                    │                  │
│  │                                           │                  │
│  │  - EWC regularization                    │                  │
│  │  - Memory replay integration             │                  │
│  │  - Multi-task loss balancing             │                  │
│  │  - Task performance tracking             │                  │
│  └──────────────────────────────────────────┘                  │
│                        │                                        │
│                        ▼                                        │
│  ┌──────────────────────────────────────────┐                  │
│  │     Knowledge Consolidation              │                  │
│  │                                           │                  │
│  │  - Compute Fisher information            │                  │
│  │  - Store task-specific parameters        │                  │
│  │  - Manage importance weights             │                  │
│  └──────────────────────────────────────────┘                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
         ┌──────────────────────────────────┐
         │     MycelialAgent (Enhanced)      │
         │  - Sequential task learning       │
         │  - Automatic forgetting detection │
         │  - Replay-enhanced training       │
         └──────────────────────────────────┘
```

### Component 1: Task Memory Manager

**Purpose:** Manage stored experiences for replay-based continual learning.

**Key Responsibilities:**
1. Store representative episodes from each learned task
2. Implement intelligent replay strategies (random, prioritized, balanced)
3. Manage memory budget (finite storage for infinite tasks)
4. Integrate with KnowledgeBase (Big Rock 8)

**API:**
```python
class TaskMemoryManager:
    def __init__(self, knowledge_base, max_episodes_per_task=100):
        self.knowledge_base = knowledge_base
        self.max_episodes_per_task = max_episodes_per_task
        self.task_buffers = {}  # task_id -> episode buffer

    def store_task_memory(self, task_id, episodes):
        """Store representative episodes for task"""
        # Core-set selection (diversity-based sampling)
        representative = self._select_core_set(episodes, k=self.max_episodes_per_task)
        self.task_buffers[task_id] = representative

    def sample_replay_batch(self, strategy='balanced', batch_size=32):
        """Sample experiences for replay"""
        # Strategy options:
        # - 'balanced': Equal samples from all tasks
        # - 'prioritized': Sample based on task importance/recency
        # - 'similarity': Sample tasks similar to current task
        pass

    def get_task_coverage(self):
        """Get distribution of stored episodes across tasks"""
        return {tid: len(buffer) for tid, buffer in self.task_buffers.items()}
```

### Component 2: Forgetting Detector

**Purpose:** Monitor agent performance on previously learned tasks and detect catastrophic forgetting.

**Key Responsibilities:**
1. Periodically evaluate performance on old tasks
2. Compare current vs. original performance
3. Trigger interventions (replay, re-training) when forgetting detected
4. Maintain performance history

**API:**
```python
class ForgettingDetector:
    def __init__(self, threshold=0.1):
        self.threshold = threshold  # Max acceptable performance drop
        self.performance_history = {}  # task_id -> [performances over time]

    def record_performance(self, task_id, performance, timestamp=None):
        """Record performance on task at given time"""
        if task_id not in self.performance_history:
            self.performance_history[task_id] = []
        self.performance_history[task_id].append({
            'performance': performance,
            'timestamp': timestamp or time.time()
        })

    def detect_forgetting(self, task_id):
        """Check if performance has degraded significantly"""
        history = self.performance_history.get(task_id, [])
        if len(history) < 2:
            return False, 0.0

        baseline = max(h['performance'] for h in history[:-1])
        current = history[-1]['performance']
        drop = baseline - current

        return drop > self.threshold, drop

    def get_forgotten_tasks(self):
        """Get list of tasks showing forgetting"""
        forgotten = []
        for task_id in self.performance_history:
            is_forgotten, drop = self.detect_forgetting(task_id)
            if is_forgotten:
                forgotten.append((task_id, drop))
        return forgotten
```

### Component 3: Continual Learner

**Purpose:** Orchestrate continual learning across sequential tasks using EWC + memory replay.

**Key Responsibilities:**
1. Implement EWC regularization
2. Integrate memory replay into training loop
3. Balance losses across current task and replayed tasks
4. Track task sequence and importance

**API:**
```python
class ContinualLearner:
    def __init__(self, agent, memory_manager, forgetting_detector, ewc_lambda=1000):
        self.agent = agent
        self.memory_manager = memory_manager
        self.forgetting_detector = forgetting_detector
        self.ewc_lambda = ewc_lambda

        # Task sequence
        self.task_sequence = []  # Ordered list of learned tasks
        self.fisher_information = {}  # task_id -> Fisher matrix
        self.optimal_parameters = {}  # task_id -> optimal θ*

    def learn_new_task(self, task_descriptor, episodes, num_iterations=1000):
        """Learn new task with continual learning safeguards"""
        task_id = task_descriptor.task_id
        self.task_sequence.append(task_id)

        # Training loop
        for iteration in range(num_iterations):
            # Sample from current task
            current_batch = sample_from_episodes(episodes)

            # Sample from memory (replay)
            if len(self.task_sequence) > 1:
                replay_batch = self.memory_manager.sample_replay_batch()
            else:
                replay_batch = []

            # Compute losses
            current_loss = compute_task_loss(current_batch)
            replay_loss = compute_replay_loss(replay_batch) if replay_batch else 0
            ewc_loss = self._compute_ewc_loss()

            # Combined loss
            total_loss = current_loss + replay_loss + self.ewc_lambda * ewc_loss

            # Update agent
            self.agent.update(total_loss)

            # Periodically check for forgetting
            if iteration % 100 == 0:
                self._check_forgetting()

        # After learning, consolidate knowledge
        self._consolidate_task(task_id)

        # Store memories
        self.memory_manager.store_task_memory(task_id, episodes)

    def _compute_ewc_loss(self):
        """Compute EWC regularization term"""
        loss = 0.0
        current_params = self.agent.get_parameters()

        for task_id in self.task_sequence[:-1]:  # All previous tasks
            F = self.fisher_information[task_id]
            θ_star = self.optimal_parameters[task_id]

            # L_EWC = (λ/2) Σ F_i (θ_i - θ*_i)²
            for param_name in current_params:
                diff = current_params[param_name] - θ_star[param_name]
                loss += (F[param_name] * diff ** 2).sum()

        return 0.5 * loss

    def _consolidate_task(self, task_id):
        """Compute and store Fisher information for task"""
        # Compute Fisher information matrix
        F = self._compute_fisher_information(task_id)
        self.fisher_information[task_id] = F

        # Store optimal parameters
        self.optimal_parameters[task_id] = self.agent.get_parameters()

    def _compute_fisher_information(self, task_id):
        """Compute diagonal Fisher information matrix"""
        # F_i = E[(∂log p(y|x,θ)/∂θ_i)²]
        # Approximation: Use gradient on sample data
        pass

    def _check_forgetting(self):
        """Check all previous tasks for forgetting"""
        for task_id in self.task_sequence[:-1]:
            # Evaluate on task
            perf = self.agent.evaluate_on_task(task_id)
            self.forgetting_detector.record_performance(task_id, perf)

            # Check if forgotten
            is_forgotten, drop = self.forgetting_detector.detect_forgetting(task_id)
            if is_forgotten:
                logger.warning(f"Forgetting detected on task {task_id}: {drop:.2%} drop")
                # Trigger intervention (extra replay, etc.)
                self._intervention(task_id)
```

### Component 4: Knowledge Consolidation

**Purpose:** Identify and protect important parameters for long-term retention.

**Methods:**
1. **Fisher Information Computation:** Identify important weights
2. **Synaptic Intelligence:** Track parameter importance during training
3. **Importance Weighting:** Assign protection levels to parameters

---

## Implementation Plan

### Week 7 (Days 43-49)

#### Day 43: Planning & Research
- [x] Research continual learning methods
- [ ] Design system architecture
- [ ] Write implementation plan
- [ ] Set performance targets

#### Day 44: Task Memory Manager
- [ ] Implement TaskMemoryManager class
- [ ] Core-set selection algorithm
- [ ] Replay sampling strategies (balanced, prioritized, similarity-based)
- [ ] Integration with KnowledgeBase (Big Rock 8)
- [ ] Tests: Memory storage, sampling, budget management (15 tests)

#### Day 45: Forgetting Detector
- [ ] Implement ForgettingDetector class
- [ ] Performance tracking and history
- [ ] Forgetting detection algorithm
- [ ] Alert and intervention triggers
- [ ] Tests: Detection accuracy, false positives/negatives (10 tests)

#### Day 46-47: Continual Learner (Core)
- [ ] Implement ContinualLearner class
- [ ] EWC loss computation
- [ ] Fisher information computation
- [ ] Multi-task loss balancing
- [ ] Integration with memory replay
- [ ] Tests: EWC correctness, loss balancing (20 tests)

#### Day 48: Knowledge Consolidation
- [ ] Implement consolidation algorithms
- [ ] Parameter importance tracking
- [ ] Optimal parameter storage
- [ ] Tests: Fisher computation, consolidation (10 tests)

#### Day 49: Integration with Base Agent
- [ ] Add `learn_sequential_tasks()` method to MycelialAgent
- [ ] Add `evaluate_all_tasks()` method
- [ ] Add `detect_forgetting()` method
- [ ] Configuration options (ewc_lambda, replay_ratio, etc.)
- [ ] Tests: End-to-end integration (15 tests)

### Week 8 (Days 50-56)

#### Day 50-51: Advanced Replay Strategies
- [ ] Task similarity-based replay
- [ ] Curriculum-based replay (easy→hard)
- [ ] Adaptive replay (based on forgetting)
- [ ] Tests: Strategy comparisons (10 tests)

#### Day 52-53: Performance Validation
- [ ] Experiment 1: Sequential task learning (10 tasks)
- [ ] Experiment 2: Catastrophic forgetting prevention
- [ ] Experiment 3: Long-term retention (100 tasks)
- [ ] Experiment 4: Comparison vs. baseline (no CL)
- [ ] Generate validation report

#### Day 54: Optimization & Tuning
- [ ] EWC lambda tuning
- [ ] Replay ratio optimization
- [ ] Memory budget optimization
- [ ] Performance profiling

#### Day 55: Documentation
- [ ] Create BIG_ROCK_9_API_GUIDE.md
- [ ] Usage examples
- [ ] Best practices guide
- [ ] Troubleshooting guide

#### Day 56: Final Testing & Polish
- [ ] Full test suite run (80+ tests)
- [ ] Integration testing
- [ ] Performance benchmarking
- [ ] Update PROGRESS.md

---

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Task Retention | >90% | Performance on old tasks vs. baseline |
| Forgetting Detection Accuracy | >95% | Precision/recall on forgetting events |
| Sequential Task Capacity | 100+ tasks | Tasks learned without degradation |
| Memory Overhead | <10MB per task | Storage for Fisher info + exemplars |
| Training Overhead | <20% | Additional time vs. no CL |
| Zero Forgetting Guarantee | 100% | On critical/flagged tasks |

---

## Integration with Existing Big Rocks

### Big Rock 8 (Transfer Learning)
- **Synergy:** Use KnowledgeBase for memory storage
- **Enhancement:** Transfer + continual learning = complete lifelong learning
- **API:** Shared TaskDescriptor, Episode, KnowledgeBase

### Big Rock 7 (GNN Communication)
- **Synergy:** Agents can share continual learning strategies
- **Enhancement:** Collective knowledge consolidation
- **API:** Broadcast forgetting alerts, share Fisher information

### Big Rock 5 (Electrical Signaling)
- **Synergy:** Fast forgetting alerts via signal bus
- **Enhancement:** Real-time intervention triggers
- **API:** FORGETTING_DETECTED signal type

---

## Risk Mitigation

### Risk 1: EWC Over-Constraining
**Problem:** Too much regularization prevents learning new tasks

**Mitigation:**
- Adaptive λ (decay over time)
- Task-specific λ based on importance
- Hybrid approach (EWC + replay)

### Risk 2: Memory Explosion
**Problem:** Storing experiences for 100+ tasks

**Mitigation:**
- Fixed memory budget per task
- Core-set selection (most informative examples)
- Compression techniques
- Generative replay (synthesize examples)

### Risk 3: Computational Overhead
**Problem:** Fisher computation and replay slow down training

**Mitigation:**
- Diagonal Fisher approximation (vs. full matrix)
- Batch Fisher computation
- Asynchronous replay
- Caching and optimization

### Risk 4: Forgetting Detection Lag
**Problem:** Detection happens after significant degradation

**Mitigation:**
- Continuous monitoring (every N iterations)
- Proactive replay (before forgetting occurs)
- Early warning thresholds

---

## Success Criteria

Big Rock 9 is successful if:

1. ✅ Agent learns 100+ sequential tasks without catastrophic forgetting
2. ✅ Performance on all tasks stays within 10% of original
3. ✅ Forgetting detector achieves 95%+ accuracy
4. ✅ Memory overhead <1GB for 100 tasks
5. ✅ Training overhead <20% vs. baseline
6. ✅ Integration with existing Big Rocks (4-8) seamless
7. ✅ API Guide and documentation complete
8. ✅ Test suite >80 tests, >90% coverage

---

## Next Steps

After Big Rock 9, agents will have:
- ✅ Transfer learning (Big Rock 8)
- ✅ Continual learning (Big Rock 9)
- ✅ Meta-learning (Big Rock 8 - MAML)

This completes the **Advanced Intelligence** phase. The next logical phase would be:

**Phase 4: Production Readiness**
- Big Rock 10: Monitoring & Observability
- Big Rock 11: Auto-Scaling & Load Balancing
- Big Rock 12: Security & Access Control

---

## Conclusion

Big Rock 9 transforms MAE from a transfer learning system into a true **lifelong learning system**. Agents will be able to:

1. Learn indefinitely without forgetting
2. Accumulate knowledge across 100+ tasks
3. Maintain performance on all learned capabilities
4. Detect and prevent catastrophic forgetting automatically
5. Operate in production environments with evolving task distributions

This is the final piece needed for "self-educating" agents that continuously improve over their lifetime.

**Estimated Timeline:** 14 days (2 weeks)
**LOC Estimate:** ~2,500 lines (implementation) + ~1,500 (tests) + ~1,000 (docs) = ~5,000 total
**Complexity:** High (novel research integration)
**Dependencies:** Big Rocks 4-8 (all complete)

**Ready to begin implementation immediately!** 🚀
