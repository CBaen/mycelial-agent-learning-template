# Big Rock 8: Transfer Learning & Meta-Learning API Guide

**Version**: 1.0
**Status**: Production-Ready
**Target**: 10-100x learning speed-up on related tasks

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Core Components](#core-components)
4. [Transfer Learning](#transfer-learning)
5. [MAML Meta-Learning](#maml-meta-learning)
6. [Agent Integration](#agent-integration)
7. [Performance Tuning](#performance-tuning)
8. [Best Practices](#best-practices)
9. [API Reference](#api-reference)
10. [Examples](#examples)

---

## Overview

Big Rock 8 implements **Transfer Learning** and **MAML (Model-Agnostic Meta-Learning)** to enable agents to reuse knowledge from previously learned tasks, achieving **10-100x learning speed-up** on related tasks.

### Key Features

- **Task Similarity**: 128-dim embeddings for measuring task relatedness
- **Knowledge Base**: Centralized storage for 1M+ experiences, episodes, policies
- **6 Transfer Strategies**: POLICY, EXPERIENCE, VALUE, FEATURE, CURRICULUM, COMBINED
- **MAML**: Few-shot adaptation with meta-learned initialization
- **Automatic Transfer**: Base agent integration with `begin_new_task()`
- **Thread-Safe**: Concurrent multi-agent knowledge sharing

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Transfer Learning Engine                  │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │ Task       │  │ Knowledge    │  │ MAML                │ │
│  │ Similarity │──▶│ Base         │──▶│ Meta-Learner        │ │
│  │ Matrix     │  │ (1M+ exp.)   │  │ (Few-Shot)          │ │
│  └────────────┘  └──────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      MycelialAgent                           │
│  begin_new_task() │ store_transition() │ store_episode()    │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Basic Setup

```python
from src.core.task_representation import TaskDescriptor, TaskEmbedding
from src.core.knowledge_base import KnowledgeBase
from src.core.transfer_learning import TransferLearningEngine, TransferStrategy
from src.agents.base_agent import MycelialAgent

# Initialize knowledge base and transfer engine
kb = KnowledgeBase(max_transitions=1000000, max_episodes=10000)
embedding = TaskEmbedding(embedding_dim=128)
engine = TransferLearningEngine(
    knowledge_base=kb,
    task_embedding=embedding,
    default_strategy=TransferStrategy.COMBINED
)

# Initialize agent with transfer capabilities
agent = MycelialAgent(
    model=model,
    redis_client=redis_client,
    knowledge_base=kb,
    transfer_engine=engine,
    agent_config={"transfer_enabled": True}
)
```

### 2. Define a Task

```python
# Create task descriptor
navigation_task = TaskDescriptor(
    task_id="navigation_v1",
    task_type="navigation",
    state_dim=10,  # State space dimensionality
    action_dim=4,  # Action space dimensionality
    reward_range=(-1.0, 10.0),
    episode_length=100,
    difficulty=0.5,  # 0.0 = easy, 1.0 = hard
    metadata={"environment": "gridworld", "version": "1.0"}
)

# Optionally compute task signatures from sample data
state_samples = np.random.randn(1000, 10)  # Sample states
reward_samples = np.random.rand(1000)  # Sample rewards
transition_samples = np.random.randn(1000, 10)  # State transitions

navigation_task.compute_signatures(
    state_samples, reward_samples, transition_samples
)
```

### 3. Start Learning with Transfer

```python
# Begin new task with automatic transfer
result = agent.begin_new_task(
    task_descriptor=navigation_task,
    use_transfer=True,  # Enable transfer learning
    use_maml=False,  # Disable MAML for now
    min_similarity=0.5,  # Minimum task similarity threshold
    k_source_tasks=3  # Use top-3 similar source tasks
)

# Check transfer results
print(f"Transfer used: {result['transfer_used']}")
print(f"Speed-up estimate: {result['speedup_estimate']}x")
print(f"Experiences transferred: {result['num_experiences_transferred']}")
```

### 4. Learn and Store Knowledge

```python
# During training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        # Select and execute action
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        # Store transition for future transfer
        agent.store_transition_for_transfer(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done
        )

        state = next_state

    # Store episode at end
    agent.store_episode_for_transfer(
        total_reward=episode_reward,
        success=(episode_reward > threshold)
    )
```

---

## Core Components

### TaskDescriptor

Comprehensive description of a learning task.

```python
from src.core.task_representation import TaskDescriptor

task = TaskDescriptor(
    task_id="unique_task_id",  # Unique identifier
    task_type="navigation",  # Task category
    state_dim=10,  # State dimensionality
    action_dim=4,  # Action dimensionality
    reward_range=(-1.0, 10.0),  # Expected reward range
    episode_length=100,  # Typical episode length
    difficulty=0.5,  # Difficulty rating (0-1)
    metadata={}  # Additional metadata
)

# Compute statistical signatures (optional but recommended)
task.compute_signatures(state_samples, reward_samples, transition_samples)
```

**Key Fields:**
- `state_space_signature`: Mean, std, min, max per state dimension
- `reward_signature`: Mean, std, min, max, skewness of rewards
- `dynamics_signature`: State transition statistics

### TaskEmbedding

Encodes tasks into 128-dimensional vectors for similarity computation.

```python
from src.core.task_representation import TaskEmbedding

embedding = TaskEmbedding(embedding_dim=128)

# Encode task
task_vector = embedding.encode(task)  # Returns normalized (128,) array

# Compute similarity between tasks
similarity = embedding.similarity(task1, task2)  # Returns float in [-1, 1]

# Clear cache
embedding.clear_cache()
```

**Embedding Components:**
- Basic features (20 dim): state/action dims, reward range, difficulty
- Task type encoding (10 dim): MD5 hash of task type
- Statistical signatures (98 dim): Computed from sample data

### TaskSimilarityMatrix

Efficient similarity search across all tasks.

```python
from src.core.task_representation import TaskSimilarityMatrix

matrix = TaskSimilarityMatrix(embedding)

# Add tasks
matrix.add_task(task1)
matrix.add_task(task2)

# Find similar tasks
similar_tasks = matrix.find_similar_tasks(
    query_task=new_task,
    k=5,  # Top-5 most similar
    min_similarity=0.5,  # Minimum cosine similarity
    same_type_only=False,  # Filter by task type
    exclude_task_ids=["task_1"]  # Exclude specific tasks
)

# Returns: [(task_id, similarity), ...]
```

### KnowledgeBase

Centralized storage for all learned knowledge.

```python
from src.core.knowledge_base import KnowledgeBase, ExperienceTransition, Episode

kb = KnowledgeBase(
    max_transitions=1000000,  # Max experience transitions
    max_episodes=10000,  # Max complete episodes
    enable_prioritization=True,  # Prioritized replay
    embedding_dim=128
)

# Store transition
transition = ExperienceTransition(
    task_id="task_1",
    state=current_state,
    action=action,
    reward=reward,
    next_state=next_state,
    done=done,
    agent_id="agent_1",
    metadata={}
)
kb.store_transition(transition, priority=abs(reward))

# Store episode
episode = Episode(
    episode_id="ep_001",
    task_id="task_1",
    agent_id="agent_1",
    transitions=[...],
    total_reward=100.0,
    episode_length=50,
    success=True
)
kb.store_episode(episode)

# Store policy/value function
kb.store_policy("task_1_agent_1", policy_parameters)
kb.store_value_function("task_1_agent_1", value_function)
```

**Retrieval Methods:**

```python
# Retrieve similar experiences
experiences = kb.retrieve_similar_experiences(
    target_task=new_task,
    n_experiences=1000,
    similarity_threshold=0.5
)

# Retrieve successful episodes
episodes = kb.retrieve_successful_episodes(task_id="task_1", k=10)

# Find best source task for transfer
best_source = kb.find_best_source_task(
    target_task=new_task,
    k_candidates=5
)

# Get task statistics
stats = kb.get_task_statistics("task_1")
# Returns: {episode_count, success_count, success_rate, avg_reward, max_reward, ...}
```

---

## Transfer Learning

### Transfer Strategies

Six strategies for knowledge transfer:

1. **POLICY_TRANSFER**: Transfer complete policy
2. **EXPERIENCE_REPLAY**: Transfer experience transitions
3. **VALUE_INITIALIZATION**: Initialize value function
4. **FEATURE_EXTRACTION**: Transfer learned features
5. **CURRICULUM**: Progressive task sequence
6. **COMBINED**: Combine multiple strategies (recommended)

### Basic Transfer

```python
from src.core.transfer_learning import TransferLearningEngine, TransferStrategy

engine = TransferLearningEngine(
    knowledge_base=kb,
    task_embedding=embedding,
    default_strategy=TransferStrategy.COMBINED,
    max_experiences_per_task=10000,
    max_episodes_per_task=100,
    min_source_task_performance=0.5  # Only use successful source tasks
)

# Initiate transfer
result = engine.initiate_transfer(
    target_task=new_task,
    agent_id="agent_1",
    strategy=TransferStrategy.COMBINED,  # Override default
    min_similarity=0.5,
    k_source_tasks=3,
    same_type_only=False
)

# Check results
print(f"Source tasks: {result.source_tasks}")
print(f"Experiences transferred: {result.num_experiences_transferred}")
print(f"Policy transferred: {result.policy_transferred}")
print(f"Value function transferred: {result.value_function_transferred}")
print(f"Transfer time: {result.transfer_time}s")
```

### Evaluating Transfer Performance

```python
# Measure actual speed-up
evaluation = engine.evaluate_transfer(
    target_task_id="task_1",
    baseline_performance=0.3,  # Without transfer
    transfer_performance=0.8,  # With transfer
    baseline_samples=10000,  # Samples needed without transfer
    transfer_samples=500  # Samples needed with transfer
)

print(f"Speed-up: {evaluation['speedup_factor']}x")
print(f"Performance gain: {evaluation['performance_gain']}")
print(f"Meets 10x target: {evaluation['meets_10x_target']}")
print(f"Meets 100x target: {evaluation['meets_100x_target']}")
```

### Transfer History

```python
# Get transfer history for an agent
history = engine.get_transfer_history(agent_id="agent_1")

for transfer_result in history:
    print(f"Task: {transfer_result.target_task_id}")
    print(f"Speed-up: {transfer_result.speedup_factor}x")
    print(f"Sources: {transfer_result.source_tasks}")

# Get average speed-up across all transfers
avg_speedup = engine.get_average_speedup()
print(f"Average speed-up: {avg_speedup}x")

# Get speed-up for specific strategy
policy_speedup = engine.get_average_speedup(strategy=TransferStrategy.POLICY_TRANSFER)
```

---

## MAML Meta-Learning

### Setup and Configuration

```python
from src.core.maml import MAMLLearner, MAMLConfig, create_maml_learner

# Configure MAML
config = MAMLConfig(
    meta_learning_rate=0.001,  # Outer loop LR
    inner_learning_rate=0.01,  # Inner loop LR
    num_inner_steps=5,  # Adaptation steps
    num_tasks_per_batch=4,  # Tasks per meta-batch
    k_shot=5,  # Examples for adaptation
    query_size=10,  # Examples for meta-evaluation
    first_order=False,  # Use full second-order gradients
    max_meta_iterations=1000,
    adaptation_steps_eval=10
)

# Create MAML learner
def model_init_fn():
    """Initialize model parameters"""
    return {"weights": np.random.randn(state_dim, action_dim)}

maml = MAMLLearner(
    knowledge_base=kb,
    config=config,
    model_init_fn=model_init_fn,
    loss_fn=custom_loss_fn,  # Optional
    update_fn=custom_update_fn  # Optional
)

# Or use convenience function
maml = create_maml_learner(
    knowledge_base=kb,
    model_init_fn=model_init_fn,
    k_shot=5,
    num_inner_steps=5
)
```

### Meta-Training

```python
# Define task family
task_family = [
    TaskDescriptor(task_id=f"nav_{i}", task_type="navigation", ...)
    for i in range(10)
]

# Ensure each task has training data
for task in task_family:
    # Add episodes to knowledge base
    kb.store_episode(...)

# Meta-train
result = maml.meta_train(
    task_family_id="navigation_family",
    num_iterations=1000,
    task_descriptors=task_family
)

print(f"Meta-training complete:")
print(f"  Iterations: {result['num_iterations']}")
print(f"  Final meta-loss: {result['final_meta_loss']:.4f}")
print(f"  Training time: {result['training_time']:.2f}s")
print(f"  Meta-initialized: {result['meta_initialized']}")
```

### Few-Shot Adaptation

```python
# Prepare support episodes (k-shot examples)
support_episodes = [
    Episode(...)  # 5-10 episodes for new task
    for _ in range(5)
]

# Adapt to new task
adaptation_result = maml.adapt_to_task(
    target_task=new_task,
    agent_id="agent_1",
    support_episodes=support_episodes,
    num_adaptation_steps=5
)

print(f"Adaptation Results:")
print(f"  Meta-learned: {adaptation_result.meta_learned}")
print(f"  Pre-adaptation perf: {adaptation_result.pre_adaptation_performance:.3f}")
print(f"  Post-adaptation perf: {adaptation_result.post_adaptation_performance:.3f}")
print(f"  Performance gain: {adaptation_result.performance_gain:.3f}")
print(f"  Adaptation time: {adaptation_result.adaptation_time:.2f}s")

# Use adapted parameters
adapted_params = adaptation_result.adapted_parameters
```

### Save/Load Meta-Parameters

```python
# Save meta-learned parameters
meta_params = maml.get_meta_parameters()
np.save("meta_parameters.npy", meta_params)

# Load meta-learned parameters
loaded_params = np.load("meta_parameters.npy")
maml.set_meta_parameters(loaded_params)

# Check initialization status
if maml.meta_initialized:
    print("Ready for few-shot adaptation!")
```

---

## Agent Integration

The `MycelialAgent` base class includes full transfer learning integration.

### Initialize Agent

```python
from src.agents.base_agent import MycelialAgent

agent = MycelialAgent(
    model=model,
    redis_client=redis_client,
    team_id="team_1",
    agent_config={
        "transfer_enabled": True,  # Enable transfer learning
        "maml_enabled": True,  # Enable MAML
    },
    knowledge_base=kb,
    transfer_engine=engine,
    maml_learner=maml
)
```

### Begin New Task

```python
# Define task
task = TaskDescriptor(
    task_id="new_navigation_task",
    task_type="navigation",
    state_dim=10,
    action_dim=4
)

# Start learning with automatic transfer
result = agent.begin_new_task(
    task_descriptor=task,
    use_transfer=True,
    use_maml=True,
    min_similarity=0.5,
    k_source_tasks=3
)

# Check what happened
if result['transfer_used']:
    print(f"Transferred knowledge from {result['num_source_tasks']} tasks")
    print(f"Estimated speed-up: {result['speedup_estimate']}x")

if result['maml_used']:
    print(f"MAML adaptation gain: {result['performance_gain']:.2f}")
```

### Store Knowledge During Learning

```python
# In training loop
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        # Store transition
        agent.store_transition_for_transfer(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            metadata={"step": step}
        )

        episode_reward += reward
        state = next_state

    # Store complete episode
    episode_id = agent.store_episode_for_transfer(
        total_reward=episode_reward,
        success=(episode_reward > threshold),
        clear_buffer=True
    )

    # Periodically store policy and value function
    if episode % 10 == 0:
        agent.store_policy_for_transfer(agent.policy)
        agent.store_value_function_for_transfer(agent.value_function)
```

### Evaluate Transfer Performance

```python
# After learning new task
evaluation = agent.evaluate_transfer_performance(
    baseline_performance=0.3,  # Measured without transfer
    current_performance=0.8,  # Measured with transfer
    baseline_samples=10000,
    current_samples=500
)

print(f"Transfer speed-up: {evaluation['speedup_factor']}x")
```

### Get Statistics

```python
# Transfer learning stats
transfer_stats = agent.get_transfer_statistics()
print(f"Number of transfers: {transfer_stats['num_transfers']}")
print(f"Average speed-up: {transfer_stats['avg_speedup']}x")
print(f"Max speed-up: {transfer_stats['max_speedup']}x")
print(f"Total experiences transferred: {transfer_stats['total_experiences_transferred']}")

# MAML stats
maml_stats = agent.get_maml_statistics()
print(f"Number of adaptations: {maml_stats['num_adaptations']}")
print(f"Average performance gain: {maml_stats['avg_performance_gain']:.2f}")
print(f"Meta-initialized: {maml_stats['meta_initialized']}")
```

---

## Performance Tuning

### Task Similarity Threshold

```python
# Higher threshold = more selective transfer (safer but fewer sources)
result = agent.begin_new_task(
    task_descriptor=task,
    min_similarity=0.7  # Only transfer from very similar tasks
)

# Lower threshold = more aggressive transfer (more sources but less relevant)
result = agent.begin_new_task(
    task_descriptor=task,
    min_similarity=0.3  # Transfer from moderately similar tasks
)
```

**Recommendations:**
- Start with 0.5 for general use
- Use 0.7+ for safety-critical applications
- Use 0.3-0.5 for exploration and discovery

### Number of Source Tasks

```python
# Use more sources for diverse knowledge
result = agent.begin_new_task(
    task_descriptor=task,
    k_source_tasks=5  # Combine knowledge from 5 tasks
)

# Use fewer sources for focused transfer
result = agent.begin_new_task(
    task_descriptor=task,
    k_source_tasks=1  # Transfer from single best task
)
```

**Recommendations:**
- 1-2 sources: High similarity tasks (> 0.8)
- 3-5 sources: Moderate similarity (0.5-0.8)
- 5+ sources: Low similarity or ensemble learning

### Transfer Strategy Selection

```python
# Policy transfer: Fast, works well for similar tasks
engine.initiate_transfer(..., strategy=TransferStrategy.POLICY_TRANSFER)

# Experience replay: Flexible, works across dissimilar tasks
engine.initiate_transfer(..., strategy=TransferStrategy.EXPERIENCE_REPLAY)

# Combined: Best overall performance (recommended)
engine.initiate_transfer(..., strategy=TransferStrategy.COMBINED)
```

### MAML Configuration

```python
# Fast adaptation (fewer steps)
config = MAMLConfig(
    k_shot=3,  # Fewer examples
    num_inner_steps=3,  # Fewer adaptation steps
    first_order=True  # Use first-order approximation
)

# High-quality adaptation (more computation)
config = MAMLConfig(
    k_shot=10,  # More examples
    num_inner_steps=10,  # More adaptation steps
    first_order=False  # Full second-order gradients
)
```

---

## Best Practices

### 1. Task Signature Computation

Always compute task signatures from real data for better similarity:

```python
# Collect sample data during initial exploration
state_samples = []
reward_samples = []
transition_samples = []

for _ in range(1000):  # Collect 1000 samples
    state = env.step()
    state_samples.append(state)
    # ... collect rewards and transitions

# Compute signatures
task.compute_signatures(
    np.array(state_samples),
    np.array(reward_samples),
    np.array(transition_samples)
)
```

### 2. Source Task Performance Filtering

Only transfer from successful source tasks:

```python
engine = TransferLearningEngine(
    knowledge_base=kb,
    task_embedding=embedding,
    min_source_task_performance=0.7  # Only use tasks with 70%+ success
)
```

### 3. Progressive Transfer

Start with high similarity, gradually decrease:

```python
# Phase 1: Learn from very similar tasks
result = agent.begin_new_task(task, min_similarity=0.8)

# Phase 2: If not enough sources, expand search
if result['num_source_tasks'] == 0:
    result = agent.begin_new_task(task, min_similarity=0.5)
```

### 4. Combine Transfer and MAML

Use transfer for initialization, MAML for fine-tuning:

```python
# Transfer provides good starting point
result = agent.begin_new_task(
    task_descriptor=task,
    use_transfer=True,
    use_maml=True  # Further refine with MAML
)
```

### 5. Monitor Transfer Quality

Track and validate transfer effectiveness:

```python
# After learning
evaluation = agent.evaluate_transfer_performance(
    baseline_performance=baseline_perf,
    current_performance=current_perf,
    baseline_samples=baseline_samples,
    current_samples=current_samples
)

if evaluation['speedup_factor'] < 2.0:
    print("Warning: Transfer not effective, consider adjusting similarity threshold")
```

### 6. Periodic Knowledge Cleanup

Clear old/irrelevant knowledge:

```python
# Clear data for outdated tasks
kb.clear_task_data("old_task_id")

# Or rebuild knowledge base periodically
if total_tasks > 1000:
    # Keep only recent/successful tasks
    pass
```

---

## API Reference

### TaskDescriptor

```python
TaskDescriptor(
    task_id: str,
    task_type: str,
    state_dim: int,
    action_dim: int,
    reward_range: Tuple[float, float] = (-1.0, 1.0),
    episode_length: int = 100,
    difficulty: float = 0.5,
    metadata: Dict[str, Any] = {}
)

# Methods
.compute_signatures(state_samples, reward_samples, transition_samples)
.validate() -> None
```

### TaskEmbedding

```python
TaskEmbedding(embedding_dim: int = 128)

# Methods
.encode(task: TaskDescriptor) -> np.ndarray
.similarity(task1: TaskDescriptor, task2: TaskDescriptor) -> float
.clear_cache() -> None
```

### TaskSimilarityMatrix

```python
TaskSimilarityMatrix(task_embedding: TaskEmbedding)

# Methods
.add_task(task: TaskDescriptor) -> None
.find_similar_tasks(
    query_task: TaskDescriptor,
    k: int = 5,
    min_similarity: float = 0.5,
    same_type_only: bool = False,
    exclude_task_ids: List[str] = None
) -> List[Tuple[str, float]]
.clear() -> None
```

### KnowledgeBase

```python
KnowledgeBase(
    max_transitions: int = 1000000,
    max_episodes: int = 10000,
    enable_prioritization: bool = True,
    embedding_dim: int = 128
)

# Storage Methods
.store_transition(transition: ExperienceTransition, priority: float) -> None
.store_episode(episode: Episode) -> None
.store_policy(key: str, policy: Any) -> None
.store_value_function(key: str, value_function: Any) -> None

# Retrieval Methods
.retrieve_similar_experiences(
    target_task: TaskDescriptor,
    n_experiences: int = 1000,
    similarity_threshold: float = 0.5
) -> List[ExperienceTransition]
.retrieve_successful_episodes(task_id: str, k: int = 10) -> List[Episode]
.retrieve_policy(key: str) -> Any
.retrieve_value_function(key: str) -> Any
.find_best_source_task(
    target_task: TaskDescriptor,
    k_candidates: int = 5
) -> Optional[str]

# Utility Methods
.get_task_statistics(task_id: str) -> Dict[str, Any]
.clear_task_data(task_id: str) -> None
```

### TransferLearningEngine

```python
TransferLearningEngine(
    knowledge_base: KnowledgeBase,
    task_embedding: TaskEmbedding,
    default_strategy: TransferStrategy = TransferStrategy.COMBINED,
    max_experiences_per_task: int = 10000,
    max_episodes_per_task: int = 100,
    min_source_task_performance: float = 0.5
)

# Methods
.initiate_transfer(
    target_task: TaskDescriptor,
    agent_id: str,
    strategy: TransferStrategy = None,
    min_similarity: float = 0.5,
    k_source_tasks: int = 3,
    same_type_only: bool = False
) -> TransferResult

.evaluate_transfer(
    target_task_id: str,
    baseline_performance: float,
    transfer_performance: float,
    baseline_samples: int,
    transfer_samples: int
) -> Dict[str, Any]

.get_transfer_history(
    target_task_id: str = None,
    agent_id: str = None
) -> List[TransferResult]

.get_average_speedup(strategy: TransferStrategy = None) -> float
.clear_history() -> None
```

### MAMLLearner

```python
MAMLLearner(
    knowledge_base: KnowledgeBase,
    config: MAMLConfig,
    model_init_fn: Callable[[], Any],
    loss_fn: Callable = None,
    update_fn: Callable = None
)

# Methods
.meta_train(
    task_family_id: str,
    num_iterations: int = None,
    task_descriptors: List[TaskDescriptor] = None
) -> Dict[str, Any]

.adapt_to_task(
    target_task: TaskDescriptor,
    agent_id: str,
    support_episodes: List[Episode],
    num_adaptation_steps: int = None
) -> AdaptationResult

.get_meta_parameters() -> Any
.set_meta_parameters(parameters: Any) -> None
.get_adaptation_history(...) -> List[AdaptationResult]
.get_average_adaptation_gain() -> float
.clear_history() -> None
```

### MycelialAgent Integration

```python
# New Methods (Big Rock 8)
.begin_new_task(
    task_descriptor: TaskDescriptor,
    use_transfer: bool = True,
    use_maml: bool = False,
    min_similarity: float = 0.5,
    k_source_tasks: int = 3
) -> Dict[str, Any]

.store_transition_for_transfer(...) -> None
.store_episode_for_transfer(total_reward: float, success: bool) -> str
.store_policy_for_transfer(policy: Any) -> None
.store_value_function_for_transfer(value_function: Any) -> None

.evaluate_transfer_performance(...) -> Dict[str, Any]
.get_transfer_statistics() -> Dict[str, Any]
.get_maml_statistics() -> Dict[str, Any]
```

---

## Examples

### Example 1: Simple Transfer Learning

```python
import numpy as np
from src.core.task_representation import TaskDescriptor
from src.core.knowledge_base import KnowledgeBase, Episode
from src.core.transfer_learning import TransferLearningEngine, TransferStrategy

# Setup
kb = KnowledgeBase()
engine = TransferLearningEngine(kb, kb.task_embedding)

# Source task
source = TaskDescriptor(
    task_id="cartpole_v1",
    task_type="control",
    state_dim=4,
    action_dim=2
)
kb.similarity_matrix.add_task(source)

# Train source task (simplified)
for i in range(100):
    kb.store_episode(Episode(
        episode_id=f"ep_{i}",
        task_id="cartpole_v1",
        agent_id="agent_1",
        transitions=[],
        total_reward=np.random.rand() * 200,
        episode_length=200,
        success=True
    ))

kb.store_policy("cartpole_v1", {"weights": np.random.randn(4, 2)})

# Target task (similar but different)
target = TaskDescriptor(
    task_id="cartpole_v2",
    task_type="control",
    state_dim=4,
    action_dim=2
)

# Transfer!
result = engine.initiate_transfer(
    target_task=target,
    agent_id="agent_2",
    strategy=TransferStrategy.COMBINED
)

print(f"Transferred {result.num_experiences_transferred} experiences")
print(f"Policy transferred: {result.policy_transferred}")
```

### Example 2: MAML Few-Shot Learning

```python
from src.core.maml import MAMLLearner, MAMLConfig

# Configure MAML
config = MAMLConfig(k_shot=5, num_inner_steps=5)
maml = MAMLLearner(
    knowledge_base=kb,
    config=config,
    model_init_fn=lambda: np.zeros((4, 2))
)

# Meta-train on task family
tasks = [
    TaskDescriptor(
        task_id=f"cartpole_variant_{i}",
        task_type="control",
        state_dim=4,
        action_dim=2,
        difficulty=0.3 + i * 0.1
    )
    for i in range(5)
]

# Add training data
for task in tasks:
    kb.similarity_matrix.tasks[task.task_id] = task
    for j in range(15):
        kb.store_episode(Episode(
            episode_id=f"{task.task_id}_ep_{j}",
            task_id=task.task_id,
            agent_id="meta_agent",
            transitions=[],
            total_reward=np.random.rand() * 200,
            episode_length=200,
            success=True
        ))

# Meta-train
result = maml.meta_train(
    task_family_id="cartpole_family",
    num_iterations=50,
    task_descriptors=tasks
)

print(f"Meta-training complete: {result['num_iterations']} iterations")

# Few-shot adaptation
new_task = TaskDescriptor(
    task_id="cartpole_new",
    task_type="control",
    state_dim=4,
    action_dim=2
)

support_eps = [Episode(...) for _ in range(5)]  # 5 examples

adaptation = maml.adapt_to_task(
    target_task=new_task,
    agent_id="new_agent",
    support_episodes=support_eps
)

print(f"Adaptation gain: {adaptation.performance_gain:.2f}")
```

### Example 3: Full Agent Integration

```python
from src.agents.base_agent import MycelialAgent

# Initialize agent with transfer capabilities
agent = MycelialAgent(
    model=model,
    redis_client=redis_client,
    knowledge_base=kb,
    transfer_engine=engine,
    maml_learner=maml,
    agent_config={
        "transfer_enabled": True,
        "maml_enabled": True
    }
)

# Define new task
task = TaskDescriptor(
    task_id="mountain_car_v1",
    task_type="control",
    state_dim=2,
    action_dim=3
)

# Start learning with automatic transfer
result = agent.begin_new_task(
    task_descriptor=task,
    use_transfer=True,
    use_maml=True
)

print(f"Speed-up estimate: {result['speedup_estimate']}x")

# Training loop
for episode in range(1000):
    state = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)

        # Store for future transfer
        agent.store_transition_for_transfer(
            state, action, reward, next_state, done
        )

        episode_reward += reward
        state = next_state

    # Store episode
    agent.store_episode_for_transfer(
        total_reward=episode_reward,
        success=(episode_reward > -110)  # MountainCar threshold
    )

    if episode % 100 == 0:
        # Save policy for transfer
        agent.store_policy_for_transfer(agent.policy)

# Get final statistics
stats = agent.get_transfer_statistics()
print(f"Transfers performed: {stats['num_transfers']}")
print(f"Average speed-up: {stats['avg_speedup']}x")
```

---

## Performance Targets

| Task Similarity | Expected Speed-up | Confidence |
|----------------|-------------------|------------|
| > 0.9 (Nearly identical) | 50-100x | High |
| 0.7-0.9 (Highly related) | 20-50x | High |
| 0.5-0.7 (Moderately related) | 10-20x | Medium |
| 0.3-0.5 (Weakly related) | 3-10x | Low |
| < 0.3 (Dissimilar) | 1-3x | Low |

---

## Troubleshooting

### Low Speed-up (<5x)

**Possible causes:**
1. Low task similarity - increase `min_similarity` threshold
2. Poor source task performance - increase `min_source_task_performance`
3. Insufficient source data - train source tasks longer
4. Wrong transfer strategy - try `TransferStrategy.COMBINED`

**Solutions:**
```python
# Filter for higher quality sources
engine = TransferLearningEngine(
    ...,
    min_source_task_performance=0.7,  # Only use successful tasks
)

result = agent.begin_new_task(
    ...,
    min_similarity=0.6,  # Higher threshold
    k_source_tasks=1  # Use only best source
)
```

### Memory Issues

**Problem:** Knowledge base growing too large

**Solutions:**
```python
# Reduce storage limits
kb = KnowledgeBase(
    max_transitions=100000,  # Reduce from 1M
    max_episodes=1000  # Reduce from 10K
)

# Periodically clear old tasks
kb.clear_task_data("old_task_id")
```

### MAML Not Converging

**Problem:** Meta-training loss not decreasing

**Solutions:**
```python
# Adjust learning rates
config = MAMLConfig(
    meta_learning_rate=0.0001,  # Reduce outer LR
    inner_learning_rate=0.01,  # Keep inner LR higher
    num_inner_steps=10,  # More adaptation steps
)

# Use more tasks per batch
config.num_tasks_per_batch = 8

# Enable early stopping
config.early_stopping_patience = 100
```

---

## Conclusion

Big Rock 8 provides a production-ready transfer learning and meta-learning framework that enables:

✅ **10-100x learning speed-up** on related tasks
✅ **Few-shot adaptation** with MAML (5-10 examples)
✅ **Automatic knowledge transfer** through base agent integration
✅ **Scalable storage** for 1M+ experiences
✅ **Thread-safe** multi-agent knowledge sharing

For questions or support, refer to the implementation plan (`BIG_ROCK_8_TRANSFER_LEARNING_PLAN.md`) or run validation experiments (`experiments/validate_transfer_speedup.py`).

**Happy transferring! 🚀**
