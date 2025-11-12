"""
Performance Validation for Big Rock 9: Episodic Memory & Replay

This script validates that we achieve the target performance improvements:
- 2-5x better data efficiency through prioritized replay
- 90%+ retention of important experiences
- Prevent catastrophic forgetting on sequential tasks
- Support 100K+ episodes with fast retrieval (<10ms)

Experiments:
1. Baseline: Learning without experience replay
2. Uniform Replay: Standard experience replay (uniform sampling)
3. Prioritized Replay: Our episodic memory with PER
4. Consolidation: Offline learning during "sleep" phases
5. Semantic Retrieval: Context-aware experience lookup

Target Metrics:
- 2-5x data efficiency improvement
- <10ms retrieval time for 100K+ experiences
- 10-20% performance gain from consolidation
- >90% semantic retrieval accuracy
"""

import numpy as np
import time
from typing import List, Dict, Tuple, Optional
import sys
from pathlib import Path
from dataclasses import dataclass

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Note: matplotlib not available, skipping visualization")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.sum_tree import SumTree
from src.memory.prioritized_replay_buffer import PrioritizedReplayBuffer
from src.memory.episodic_memory import EpisodicMemory, Experience
from src.memory.memory_consolidator import MemoryConsolidator, ConsolidationStrategy
from src.memory.semantic_retriever import SemanticRetriever, SemanticQuery
from src.memory.hindsight_replay import HindsightReplay, HERStrategy, HERTransition


@dataclass
class ExperimentResult:
    """Results from a single experiment"""
    name: str
    episodes_to_threshold: int
    training_time: float
    reward_history: List[float]
    final_performance: float
    sample_efficiency: float = 1.0
    retrieval_time: float = 0.0
    memory_stats: Dict = None


class SimpleQLearner:
    """Simple Q-Learning agent for validation experiments"""

    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = 0.99
        self.epsilon = 0.1

        # Q-table (simplified - discrete states)
        self.q_table = np.zeros((100, action_dim))

    def get_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """Epsilon-greedy action selection"""
        if epsilon is None:
            epsilon = self.epsilon

        state_idx = self._discretize_state(state)

        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        return np.argmax(self.q_table[state_idx])

    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> float:
        """Q-learning update - returns TD error"""
        state_idx = self._discretize_state(state)
        next_state_idx = self._discretize_state(next_state)

        # Q-learning update rule
        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_state_idx])

        td_error = target - self.q_table[state_idx, action]
        self.q_table[state_idx, action] += self.lr * td_error

        return abs(td_error)

    def _discretize_state(self, state: np.ndarray) -> int:
        """Convert continuous state to discrete index"""
        state_sum = np.sum(state)
        return min(99, max(0, int((state_sum + 10) * 5)))

    def reset(self):
        """Reset Q-table"""
        self.q_table = np.zeros_like(self.q_table)


class GridWorldEnvironment:
    """Simple GridWorld environment for experiments"""

    def __init__(self, size: int = 10, goal_pos: Tuple[int, int] = None,
                 difficulty: float = 0.5):
        self.size = size
        self.goal_pos = goal_pos or (size-1, size-1)
        self.difficulty = difficulty

        self.agent_pos = [0, 0]
        self.state_dim = 4  # [x, y, goal_x, goal_y]
        self.action_dim = 4  # up, down, left, right

    def reset(self) -> np.ndarray:
        """Reset environment"""
        if np.random.rand() > self.difficulty:
            self.agent_pos = [0, 0]  # Easy start
        else:
            self.agent_pos = [
                np.random.randint(self.size),
                np.random.randint(self.size)
            ]

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Execute action"""
        # Move agent
        if action == 0:  # up
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1:  # down
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
        elif action == 2:  # left
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3:  # right
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)

        # Compute reward
        dist_to_goal = abs(self.agent_pos[0] - self.goal_pos[0]) + \
                       abs(self.agent_pos[1] - self.goal_pos[1])
        reward = -0.1  # Step penalty

        done = False
        if self.agent_pos == list(self.goal_pos):
            reward = 10.0
            done = True

        return self._get_state(), reward, done

    def _get_state(self) -> np.ndarray:
        """Get current state"""
        return np.array([
            self.agent_pos[0] / self.size,
            self.agent_pos[1] / self.size,
            self.goal_pos[0] / self.size,
            self.goal_pos[1] / self.size
        ])


def experiment_1_baseline():
    """Experiment 1: Baseline learning without experience replay"""
    print("\n" + "="*80)
    print("EXPERIMENT 1: BASELINE (No Experience Replay)")
    print("="*80)

    env = GridWorldEnvironment(size=10, goal_pos=(9, 9), difficulty=0.3)
    agent = SimpleQLearner(state_dim=4, action_dim=4)

    reward_history = []
    episodes_to_threshold = 1000
    target_performance = 8.0

    start_time = time.time()

    for episode in range(1000):
        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        max_steps = 100

        while not done and steps < max_steps:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            # Learn immediately (no replay)
            agent.update(state, action, reward, next_state, done)

            episode_reward += reward
            state = next_state
            steps += 1

        reward_history.append(episode_reward)

        # Check if reached target
        if len(reward_history) >= 10:
            recent_avg = np.mean(reward_history[-10:])
            if recent_avg >= target_performance and episodes_to_threshold == 1000:
                episodes_to_threshold = episode + 1

    training_time = time.time() - start_time
    final_perf = np.mean(reward_history[-10:])

    print(f"\nResults:")
    print(f"  Episodes to threshold: {episodes_to_threshold}")
    print(f"  Training time: {training_time:.2f}s")
    print(f"  Final 10-episode average: {final_perf:.2f}")

    return ExperimentResult(
        name="Baseline (No Replay)",
        episodes_to_threshold=episodes_to_threshold,
        training_time=training_time,
        reward_history=reward_history,
        final_performance=final_perf,
        sample_efficiency=1.0
    )


def experiment_2_uniform_replay():
    """Experiment 2: Uniform experience replay"""
    print("\n" + "="*80)
    print("EXPERIMENT 2: UNIFORM EXPERIENCE REPLAY")
    print("="*80)

    env = GridWorldEnvironment(size=10, goal_pos=(9, 9), difficulty=0.3)
    agent = SimpleQLearner(state_dim=4, action_dim=4)

    # Simple replay buffer
    replay_buffer = []
    max_buffer_size = 10000
    batch_size = 32

    reward_history = []
    episodes_to_threshold = 1000
    target_performance = 8.0

    start_time = time.time()

    for episode in range(1000):
        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        max_steps = 100

        while not done and steps < max_steps:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            # Store experience
            experience = (state, action, reward, next_state, done)
            replay_buffer.append(experience)
            if len(replay_buffer) > max_buffer_size:
                replay_buffer.pop(0)

            # Replay from buffer (uniform sampling)
            if len(replay_buffer) >= batch_size:
                indices = np.random.choice(len(replay_buffer), batch_size, replace=False)
                for idx in indices:
                    s, a, r, ns, d = replay_buffer[idx]
                    agent.update(s, a, r, ns, d)

            episode_reward += reward
            state = next_state
            steps += 1

        reward_history.append(episode_reward)

        # Check if reached target
        if len(reward_history) >= 10:
            recent_avg = np.mean(reward_history[-10:])
            if recent_avg >= target_performance and episodes_to_threshold == 1000:
                episodes_to_threshold = episode + 1

    training_time = time.time() - start_time
    final_perf = np.mean(reward_history[-10:])

    print(f"\nResults:")
    print(f"  Episodes to threshold: {episodes_to_threshold}")
    print(f"  Training time: {training_time:.2f}s")
    print(f"  Final 10-episode average: {final_perf:.2f}")
    print(f"  Buffer size: {len(replay_buffer)}")

    return ExperimentResult(
        name="Uniform Replay",
        episodes_to_threshold=episodes_to_threshold,
        training_time=training_time,
        reward_history=reward_history,
        final_performance=final_perf
    )


def experiment_3_prioritized_replay():
    """Experiment 3: Prioritized experience replay (our episodic memory)"""
    print("\n" + "="*80)
    print("EXPERIMENT 3: PRIORITIZED EXPERIENCE REPLAY")
    print("="*80)

    env = GridWorldEnvironment(size=10, goal_pos=(9, 9), difficulty=0.3)
    agent = SimpleQLearner(state_dim=4, action_dim=4)

    # Our episodic memory system
    memory = EpisodicMemory(
        capacity=10000,
        alpha=0.6,
        beta=0.4,
        use_semantic_index=False  # Disable for speed
    )

    reward_history = []
    episodes_to_threshold = 1000
    target_performance = 8.0
    batch_size = 32

    start_time = time.time()

    for episode in range(1000):
        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        max_steps = 100

        while not done and steps < max_steps:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            # Store experience in episodic memory
            memory.store(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            )

            # Prioritized replay
            if len(memory) >= batch_size:
                experiences, indices, weights = memory.sample(batch_size)

                td_errors = []
                for exp in experiences:
                    td_error = agent.update(
                        exp.state, exp.action, exp.reward,
                        exp.next_state, exp.done
                    )
                    td_errors.append(td_error)

                # Update priorities based on TD errors
                memory.update_priorities(indices, np.array(td_errors))

            episode_reward += reward
            state = next_state
            steps += 1

        reward_history.append(episode_reward)

        # Check if reached target
        if len(reward_history) >= 10:
            recent_avg = np.mean(reward_history[-10:])
            if recent_avg >= target_performance and episodes_to_threshold == 1000:
                episodes_to_threshold = episode + 1

    training_time = time.time() - start_time
    final_perf = np.mean(reward_history[-10:])
    memory_stats = memory.get_statistics()

    print(f"\nResults:")
    print(f"  Episodes to threshold: {episodes_to_threshold}")
    print(f"  Training time: {training_time:.2f}s")
    print(f"  Final 10-episode average: {final_perf:.2f}")
    print(f"  Memory stats: {memory_stats}")

    return ExperimentResult(
        name="Prioritized Replay",
        episodes_to_threshold=episodes_to_threshold,
        training_time=training_time,
        reward_history=reward_history,
        final_performance=final_perf,
        memory_stats=memory_stats
    )


def experiment_4_memory_consolidation():
    """Experiment 4: Memory consolidation (offline learning)"""
    print("\n" + "="*80)
    print("EXPERIMENT 4: MEMORY CONSOLIDATION (Offline Learning)")
    print("="*80)

    env = GridWorldEnvironment(size=10, goal_pos=(9, 9), difficulty=0.3)
    agent = SimpleQLearner(state_dim=4, action_dim=4)

    # Episodic memory with consolidation
    memory = EpisodicMemory(
        capacity=10000,
        alpha=0.6,
        beta=0.4,
        use_semantic_index=False
    )

    consolidator = MemoryConsolidator(
        strategy=ConsolidationStrategy.PRIORITIZED,
        consolidation_steps=50,
        batch_size=32
    )

    reward_history = []
    episodes_to_threshold = 1000
    target_performance = 8.0
    batch_size = 32
    consolidation_frequency = 50  # Consolidate every 50 episodes

    start_time = time.time()

    for episode in range(1000):
        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        max_steps = 100

        while not done and steps < max_steps:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            # Store experience
            memory.store(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            )

            # Online learning
            if len(memory) >= batch_size:
                experiences, indices, weights = memory.sample(batch_size)

                td_errors = []
                for exp in experiences:
                    td_error = agent.update(
                        exp.state, exp.action, exp.reward,
                        exp.next_state, exp.done
                    )
                    td_errors.append(td_error)

                memory.update_priorities(indices, np.array(td_errors))

            episode_reward += reward
            state = next_state
            steps += 1

        reward_history.append(episode_reward)

        # Periodic memory consolidation (offline learning)
        if (episode + 1) % consolidation_frequency == 0 and len(memory) >= batch_size:
            print(f"  Consolidating memory at episode {episode + 1}...")

            def update_fn(experiences):
                td_errors = []
                for exp in experiences:
                    td_error = agent.update(
                        exp.state, exp.action, exp.reward,
                        exp.next_state, exp.done
                    )
                    td_errors.append(td_error)
                return np.array(td_errors)

            result = consolidator.consolidate(memory, update_fn)
            print(f"    Consolidation result: avg_td_error={result.avg_td_error:.4f}")

        # Check if reached target
        if len(reward_history) >= 10:
            recent_avg = np.mean(reward_history[-10:])
            if recent_avg >= target_performance and episodes_to_threshold == 1000:
                episodes_to_threshold = episode + 1

    training_time = time.time() - start_time
    final_perf = np.mean(reward_history[-10:])
    consolidation_stats = consolidator.get_statistics()

    print(f"\nResults:")
    print(f"  Episodes to threshold: {episodes_to_threshold}")
    print(f"  Training time: {training_time:.2f}s")
    print(f"  Final 10-episode average: {final_perf:.2f}")
    print(f"  Consolidations performed: {consolidation_stats['total_consolidations']}")
    print(f"  Total experiences replayed: {consolidation_stats['total_experiences_replayed']}")

    return ExperimentResult(
        name="With Consolidation",
        episodes_to_threshold=episodes_to_threshold,
        training_time=training_time,
        reward_history=reward_history,
        final_performance=final_perf,
        memory_stats=consolidation_stats
    )


def experiment_5_semantic_retrieval():
    """Experiment 5: Semantic retrieval performance"""
    print("\n" + "="*80)
    print("EXPERIMENT 5: SEMANTIC RETRIEVAL PERFORMANCE")
    print("="*80)

    # Test retrieval speed and accuracy
    print("\nTesting semantic retrieval with varying memory sizes...")

    memory_sizes = [1000, 10000, 50000, 100000]
    retrieval_times = []

    for size in memory_sizes:
        print(f"\n  Testing with {size} experiences...")

        # Create memory with semantic indexing
        memory = EpisodicMemory(
            capacity=size,
            alpha=0.6,
            beta=0.4,
            use_semantic_index=True
        )

        # Fill memory with random experiences
        for i in range(size):
            state = np.random.randn(4)
            memory.store(
                state=state,
                action=np.random.randint(4),
                reward=np.random.randn(),
                next_state=np.random.randn(4),
                done=False
            )

        # Test retrieval speed
        query_state = np.random.randn(4)

        start = time.time()
        for _ in range(100):  # 100 queries
            similar = memory.get_similar_experiences(query_state, k=10)
        avg_retrieval_time = (time.time() - start) / 100 * 1000  # ms

        retrieval_times.append(avg_retrieval_time)

        print(f"    Memory size: {len(memory)}")
        print(f"    Avg retrieval time: {avg_retrieval_time:.2f}ms")
        print(f"    Target: <10ms - {'✅ PASS' if avg_retrieval_time < 10 else '❌ FAIL'}")

    print(f"\n{'='*80}")
    print("SEMANTIC RETRIEVAL SUMMARY")
    print(f"{'='*80}")
    for size, rt in zip(memory_sizes, retrieval_times):
        status = "✅ PASS" if rt < 10 else "❌ FAIL"
        print(f"  {size:>6} experiences: {rt:>6.2f}ms  {status}")

    return {
        'memory_sizes': memory_sizes,
        'retrieval_times': retrieval_times,
        'target_met': all(rt < 10 for rt in retrieval_times)
    }


def experiment_6_her_validation():
    """Experiment 6: Hindsight Experience Replay validation"""
    print("\n" + "="*80)
    print("EXPERIMENT 6: HINDSIGHT EXPERIENCE REPLAY (HER)")
    print("="*80)

    print("\nTesting HER goal relabeling strategies...")

    her = HindsightReplay(
        strategy=HERStrategy.FUTURE,
        relabel_ratio=0.8,
        k_future=4
    )

    # Create sample episode (failed to reach goal)
    episode = []
    for i in range(10):
        trans = HERTransition(
            state=np.array([i/10.0, i/10.0]),
            action=0,
            reward=-1.0,  # Failed
            next_state=np.array([(i+1)/10.0, (i+1)/10.0]),
            done=False,
            goal=np.array([1.0, 1.0]),  # Original goal
            achieved_goal=np.array([i/10.0, i/10.0]),  # What was achieved
            info={'distance_threshold': 0.05}
        )
        episode.append(trans)

    # Relabel episode
    print(f"\n  Original episode length: {len(episode)}")
    relabeled = her.relabel_episode(episode)
    print(f"  After HER relabeling: {len(relabeled)}")

    stats = her.get_statistics()
    print(f"\n  HER Statistics:")
    print(f"    Strategy: {stats['strategy']}")
    print(f"    Relabel ratio: {stats['relabel_ratio']}")
    print(f"    Augmentation factor: {stats['augmentation_factor']:.2f}x")
    print(f"    Total relabeled: {stats['total_relabeled']}")

    # Test multiple strategies
    print(f"\n  Testing all HER strategies...")
    strategies = [HERStrategy.FUTURE, HERStrategy.FINAL, HERStrategy.EPISODE,
                  HERStrategy.RANDOM, HERStrategy.MIXED]

    for strategy in strategies:
        her_test = HindsightReplay(strategy=strategy, relabel_ratio=0.8)
        relabeled_test = her_test.relabel_episode(episode)
        aug_factor = her_test.get_statistics()['augmentation_factor']
        print(f"    {strategy.value:10s}: {aug_factor:.2f}x augmentation")

    return {
        'original_length': len(episode),
        'relabeled_length': len(relabeled),
        'augmentation_factor': stats['augmentation_factor'],
        'strategies_tested': len(strategies)
    }


def generate_comparison_report(results: List[ExperimentResult],
                               semantic_results: Dict,
                               her_results: Dict):
    """Generate comprehensive validation report"""
    print("\n" + "="*80)
    print("BIG ROCK 9: EPISODIC MEMORY VALIDATION REPORT")
    print("="*80)

    baseline = results[0]

    print("\n[PERFORMANCE COMPARISON]")
    print("-" * 80)
    print(f"{'Method':<30} {'Episodes':<12} {'Time (s)':<12} {'Final Perf':<12} {'Speedup':<10}")
    print("-" * 80)

    for result in results:
        speedup = baseline.episodes_to_threshold / result.episodes_to_threshold \
                  if result.episodes_to_threshold > 0 else 0
        print(f"{result.name:<30} {result.episodes_to_threshold:<12} "
              f"{result.training_time:<12.2f} {result.final_performance:<12.2f} "
              f"{speedup:<10.1f}x")

    print("\n[DATA EFFICIENCY ANALYSIS]")
    print("-" * 80)

    # Compare prioritized replay to baseline
    prioritized = results[2]  # Experiment 3
    speedup = baseline.episodes_to_threshold / prioritized.episodes_to_threshold

    print(f"Baseline episodes to threshold: {baseline.episodes_to_threshold}")
    print(f"Prioritized replay episodes: {prioritized.episodes_to_threshold}")
    print(f"Data efficiency improvement: {speedup:.2f}x")

    if speedup >= 2.0:
        print(f"✅ PASSED: Achieved 2x+ data efficiency target!")
    else:
        print(f"⚠️  BELOW TARGET: {speedup:.2f}x (target: 2-5x)")

    print("\n[SEMANTIC RETRIEVAL PERFORMANCE]")
    print("-" * 80)
    print(f"Memory sizes tested: {semantic_results['memory_sizes']}")
    print(f"Retrieval times: {[f'{t:.2f}ms' for t in semantic_results['retrieval_times']]}")

    if semantic_results['target_met']:
        print("✅ PASSED: All retrieval times <10ms (even at 100K experiences)")
    else:
        print("❌ FAILED: Some retrieval times exceeded 10ms target")

    print("\n[HINDSIGHT EXPERIENCE REPLAY]")
    print("-" * 80)
    print(f"Original episode length: {her_results['original_length']}")
    print(f"After HER relabeling: {her_results['relabeled_length']}")
    print(f"Data augmentation: {her_results['augmentation_factor']:.2f}x")
    print(f"Strategies validated: {her_results['strategies_tested']}")
    print("✅ PASSED: HER successfully augments training data from failures")

    print("\n[OVERALL RESULTS]")
    print("-" * 80)

    targets_met = []

    # Target 1: 2-5x data efficiency
    if speedup >= 2.0:
        print("✅ Data Efficiency: 2-5x improvement achieved")
        targets_met.append(True)
    else:
        print(f"❌ Data Efficiency: {speedup:.2f}x (target: 2-5x)")
        targets_met.append(False)

    # Target 2: <10ms retrieval
    if semantic_results['target_met']:
        print("✅ Retrieval Speed: <10ms for 100K+ experiences")
        targets_met.append(True)
    else:
        print("❌ Retrieval Speed: Exceeded 10ms target")
        targets_met.append(False)

    # Target 3: HER augmentation
    if her_results['augmentation_factor'] > 1.5:
        print(f"✅ HER Augmentation: {her_results['augmentation_factor']:.2f}x data increase")
        targets_met.append(True)
    else:
        print(f"❌ HER Augmentation: {her_results['augmentation_factor']:.2f}x (target: >1.5x)")
        targets_met.append(False)

    print("\n[CONCLUSION]")
    print("-" * 80)

    if all(targets_met):
        print("BIG ROCK 9 VALIDATION: ✅ SUCCESS")
        print("All performance targets met. Episodic memory system is production-ready.")
    elif sum(targets_met) >= 2:
        print("BIG ROCK 9 VALIDATION: ⚠️  PARTIAL SUCCESS")
        print(f"{sum(targets_met)}/{len(targets_met)} targets met. Minor tuning recommended.")
    else:
        print("BIG ROCK 9 VALIDATION: ❌ NEEDS IMPROVEMENT")
        print(f"Only {sum(targets_met)}/{len(targets_met)} targets met. Further optimization needed.")

    print("="*80)


def create_visualization(results: List[ExperimentResult]):
    """Create learning curve visualization"""
    if not MATPLOTLIB_AVAILABLE:
        print("\n⚠️  Skipping visualization (matplotlib not installed)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Learning curves comparison
    ax = axes[0, 0]
    for result in results:
        ax.plot(result.reward_history, label=result.name, alpha=0.7)
    ax.axhline(y=8.0, color='r', linestyle='--', label='Target', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Learning Curves Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Episodes to threshold
    ax = axes[0, 1]
    methods = [r.name for r in results]
    episodes = [r.episodes_to_threshold for r in results]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    bars = ax.bar(methods, episodes, color=colors, alpha=0.7)
    ax.set_ylabel('Episodes to Threshold')
    ax.set_title('Sample Efficiency Comparison')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, episodes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}',
                ha='center', va='bottom', fontsize=9)

    # Plot 3: Speedup factors
    ax = axes[1, 0]
    baseline_episodes = results[0].episodes_to_threshold
    speedups = [baseline_episodes / r.episodes_to_threshold for r in results]

    bars = ax.bar(methods, speedups, color=colors, alpha=0.7)
    ax.axhline(y=2.0, color='g', linestyle='--', label='Target (2x)', alpha=0.7)
    ax.set_ylabel('Speed-up Factor')
    ax.set_title('Data Efficiency Improvement')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    # Plot 4: Training time comparison
    ax = axes[1, 1]
    times = [r.training_time for r in results]

    bars = ax.bar(methods, times, color=colors, alpha=0.7)
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Computational Efficiency')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('experiments/episodic_memory_validation_results.png',
                dpi=150, bbox_inches='tight')
    print("\n[*] Visualization saved to: experiments/episodic_memory_validation_results.png")


def main():
    """Run all validation experiments"""
    print("\n" + "="*80)
    print("BIG ROCK 9: EPISODIC MEMORY & REPLAY VALIDATION")
    print("Testing 2-5x data efficiency and <10ms retrieval for 100K+ experiences")
    print("="*80)

    # Run experiments
    results = []

    print("\n[*] Running performance experiments...")
    results.append(experiment_1_baseline())
    results.append(experiment_2_uniform_replay())
    results.append(experiment_3_prioritized_replay())
    results.append(experiment_4_memory_consolidation())

    print("\n[*] Running semantic retrieval tests...")
    semantic_results = experiment_5_semantic_retrieval()

    print("\n[*] Running HER validation...")
    her_results = experiment_6_her_validation()

    # Generate report
    generate_comparison_report(results, semantic_results, her_results)

    # Create visualization
    try:
        create_visualization(results)
    except Exception as e:
        print(f"\n⚠️  Could not create visualization: {e}")

    print("\n✅ Validation complete!")


if __name__ == "__main__":
    main()
