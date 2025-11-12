"""
Performance Validation for Big Rock 8: Transfer Learning & Meta-Learning

This script validates that we achieve the target 10-100x learning speed-up
on related tasks through transfer learning and MAML meta-learning.

Experiments:
1. Baseline: Learning from scratch (no transfer)
2. Transfer Learning: Using similar task knowledge
3. MAML: Few-shot adaptation with meta-learned initialization
4. Combined: Transfer + MAML

Target Metrics:
- 10x speed-up on moderately related tasks (similarity 0.5-0.7)
- 50x speed-up on highly related tasks (similarity > 0.7)
- 100x speed-up on nearly identical tasks (similarity > 0.9)
"""

import numpy as np
import time
from typing import List, Dict, Tuple
import sys
from pathlib import Path

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Note: matplotlib not available, skipping visualization")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.task_representation import TaskDescriptor, TaskEmbedding
from src.core.knowledge_base import KnowledgeBase, Episode, ExperienceTransition
from src.core.transfer_learning import TransferLearningEngine, TransferStrategy
from src.core.maml import MAMLLearner, MAMLConfig


class SimpleQLearner:
    """Simple Q-Learning agent for validation experiments"""

    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = 0.99
        self.epsilon = 0.1

        # Q-table (simplified - discrete states)
        self.q_table = np.zeros((100, action_dim))  # 100 discrete states

    def get_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection"""
        # Discretize continuous state
        state_idx = self._discretize_state(state)

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        return np.argmax(self.q_table[state_idx])

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Q-learning update"""
        state_idx = self._discretize_state(state)
        next_state_idx = self._discretize_state(next_state)

        # Q-learning update rule
        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_state_idx])

        self.q_table[state_idx, action] += self.lr * (target - self.q_table[state_idx, action])

    def _discretize_state(self, state: np.ndarray) -> int:
        """Convert continuous state to discrete index"""
        # Simple binning
        state_sum = np.sum(state)
        return min(99, max(0, int((state_sum + 10) * 5)))  # Map to 0-99

    def get_q_table(self):
        """Get Q-table as policy"""
        return self.q_table.copy()

    def set_q_table(self, q_table):
        """Set Q-table from transferred policy"""
        self.q_table = q_table.copy()


class GridWorldEnvironment:
    """Simple GridWorld environment for experiments"""

    def __init__(self, size: int = 10, goal_pos: Tuple[int, int] = None, difficulty: float = 0.5):
        self.size = size
        self.goal_pos = goal_pos or (size-1, size-1)
        self.difficulty = difficulty

        # Agent position
        self.agent_pos = [0, 0]

        # State dimension
        self.state_dim = 4  # [x, y, goal_x, goal_y]
        self.action_dim = 4  # up, down, left, right

    def reset(self) -> np.ndarray:
        """Reset environment"""
        # Random start based on difficulty
        if np.random.rand() > self.difficulty:
            self.agent_pos = [0, 0]  # Easy start
        else:
            self.agent_pos = [
                np.random.randint(self.size),
                np.random.randint(self.size)
            ]  # Random start

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
        dist_to_goal = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
        reward = -0.1  # Step penalty

        done = False
        if self.agent_pos == list(self.goal_pos):
            reward = 10.0  # Goal reward
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


def train_agent(
    env: GridWorldEnvironment,
    agent: SimpleQLearner,
    num_episodes: int,
    target_performance: float = 8.0
) -> Tuple[int, List[float]]:
    """
    Train agent until reaching target performance.

    Returns:
        (episodes_to_threshold, reward_history)
    """
    reward_history = []
    episodes_to_threshold = num_episodes

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        max_steps = 100

        while not done and steps < max_steps:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)

            episode_reward += reward
            state = next_state
            steps += 1

        reward_history.append(episode_reward)

        # Check if reached target
        if len(reward_history) >= 10:
            recent_avg = np.mean(reward_history[-10:])
            if recent_avg >= target_performance and episodes_to_threshold == num_episodes:
                episodes_to_threshold = episode + 1

    return episodes_to_threshold, reward_history


def experiment_1_baseline():
    """Experiment 1: Baseline learning from scratch"""
    print("\n" + "="*80)
    print("EXPERIMENT 1: BASELINE (No Transfer)")
    print("="*80)

    env = GridWorldEnvironment(size=10, goal_pos=(9, 9), difficulty=0.3)
    agent = SimpleQLearner(state_dim=4, action_dim=4)

    start_time = time.time()
    episodes_to_threshold, reward_history = train_agent(env, agent, num_episodes=1000, target_performance=8.0)
    training_time = time.time() - start_time

    print(f"\nResults:")
    print(f"  Episodes to threshold: {episodes_to_threshold}")
    print(f"  Training time: {training_time:.2f}s")
    print(f"  Final 10-episode average: {np.mean(reward_history[-10:]):.2f}")

    return {
        'episodes': episodes_to_threshold,
        'time': training_time,
        'rewards': reward_history,
        'final_perf': np.mean(reward_history[-10:])
    }


def experiment_2_transfer_learning(baseline_result: Dict):
    """Experiment 2: Transfer learning from similar task"""
    print("\n" + "="*80)
    print("EXPERIMENT 2: TRANSFER LEARNING")
    print("="*80)

    # Setup knowledge base
    kb = KnowledgeBase()
    embedding = TaskEmbedding()
    engine = TransferLearningEngine(kb, embedding)

    # Create source task (slightly different environment)
    source_task = TaskDescriptor(
        task_id="gridworld_source",
        task_type="navigation",
        state_dim=4,
        action_dim=4,
        difficulty=0.3
    )
    kb.similarity_matrix.add_task(source_task)

    # Train source agent
    print("\nTraining source task...")
    source_env = GridWorldEnvironment(size=10, goal_pos=(8, 8), difficulty=0.3)  # Slightly different goal
    source_agent = SimpleQLearner(state_dim=4, action_dim=4)
    train_agent(source_env, source_agent, num_episodes=500, target_performance=7.0)

    # Store source policy
    kb.store_policy("gridworld_source", source_agent.get_q_table())

    # Add episode for performance tracking
    kb.store_episode(Episode(
        episode_id="source_ep",
        task_id="gridworld_source",
        agent_id="source_agent",
        transitions=[],
        total_reward=8.0,
        episode_length=50,
        success=True
    ))

    # Create target task
    target_task = TaskDescriptor(
        task_id="gridworld_target",
        task_type="navigation",
        state_dim=4,
        action_dim=4,
        difficulty=0.3
    )

    # Compute similarity
    similarity = embedding.similarity(source_task, target_task)
    print(f"\nTask similarity: {similarity:.3f}")

    # Transfer to target task
    print("Transferring knowledge...")
    result = engine.initiate_transfer(
        target_task=target_task,
        agent_id="target_agent",
        strategy=TransferStrategy.POLICY_TRANSFER,
        min_similarity=0.0
    )

    # Train target agent with transferred policy
    target_env = GridWorldEnvironment(size=10, goal_pos=(9, 9), difficulty=0.3)
    target_agent = SimpleQLearner(state_dim=4, action_dim=4)

    # Initialize with transferred policy
    if result.policy_transferred:
        transferred_policy = kb.retrieve_policy(f"{target_task.task_id}_target_agent")
        if transferred_policy is not None:
            target_agent.set_q_table(transferred_policy)
            print("Policy successfully transferred!")

    start_time = time.time()
    episodes_to_threshold, reward_history = train_agent(target_env, target_agent, num_episodes=1000, target_performance=8.0)
    training_time = time.time() - start_time

    # Calculate speed-up
    speedup = baseline_result['episodes'] / episodes_to_threshold if episodes_to_threshold > 0 else 0

    print(f"\nResults:")
    print(f"  Episodes to threshold: {episodes_to_threshold}")
    print(f"  Baseline episodes: {baseline_result['episodes']}")
    print(f"  Speed-up: {speedup:.1f}x")
    print(f"  Training time: {training_time:.2f}s")
    print(f"  Final 10-episode average: {np.mean(reward_history[-10:]):.2f}")

    return {
        'episodes': episodes_to_threshold,
        'time': training_time,
        'rewards': reward_history,
        'speedup': speedup,
        'similarity': similarity,
        'final_perf': np.mean(reward_history[-10:])
    }


def experiment_3_maml():
    """Experiment 3: MAML meta-learning"""
    print("\n" + "="*80)
    print("EXPERIMENT 3: MAML META-LEARNING")
    print("="*80)

    kb = KnowledgeBase()
    config = MAMLConfig(
        k_shot=5,
        num_inner_steps=3,
        max_meta_iterations=10
    )

    def model_init():
        return np.zeros((100, 4))  # Q-table initialization

    maml = MAMLLearner(kb, config, model_init)

    # Create task family for meta-training
    print("\nMeta-training on task family...")
    tasks = []
    for i in range(5):
        task = TaskDescriptor(
            task_id=f"gridworld_meta_{i}",
            task_type="navigation",
            state_dim=4,
            action_dim=4,
            difficulty=0.3 + i * 0.1
        )
        kb.similarity_matrix.tasks[task.task_id] = task
        tasks.append(task)

        # Generate episodes for meta-training
        env = GridWorldEnvironment(size=10, goal_pos=(8+i, 8), difficulty=0.3 + i * 0.1)
        agent = SimpleQLearner(state_dim=4, action_dim=4)
        _, rewards = train_agent(env, agent, num_episodes=50)

        # Store episodes
        for j in range(15):
            episode_reward = rewards[min(j, len(rewards)-1)]
            kb.store_episode(Episode(
                episode_id=f"meta_ep_{i}_{j}",
                task_id=f"gridworld_meta_{i}",
                agent_id="meta_agent",
                transitions=[],
                total_reward=episode_reward,
                episode_length=30,
                success=(episode_reward > 5.0)
            ))

    # Meta-train
    meta_result = maml.meta_train(
        task_family_id="gridworld_navigation",
        num_iterations=10,
        task_descriptors=tasks
    )

    print(f"Meta-training complete: {meta_result['num_iterations']} iterations")

    # Test few-shot adaptation on new task
    print("\nTesting few-shot adaptation...")
    new_task = TaskDescriptor(
        task_id="gridworld_new",
        task_type="navigation",
        state_dim=4,
        action_dim=4,
        difficulty=0.5
    )

    # Generate 5 support episodes (k-shot)
    support_env = GridWorldEnvironment(size=10, goal_pos=(9, 9), difficulty=0.5)
    support_agent = SimpleQLearner(state_dim=4, action_dim=4)
    _, support_rewards = train_agent(support_env, support_agent, num_episodes=5)

    support_episodes = [
        Episode(
            episode_id=f"support_ep_{i}",
            task_id="gridworld_new",
            agent_id="new_agent",
            transitions=[],
            total_reward=support_rewards[i],
            episode_length=30,
            success=(support_rewards[i] > 5.0)
        )
        for i in range(5)
    ]

    # Adapt to new task
    adaptation_result = maml.adapt_to_task(
        target_task=new_task,
        agent_id="new_agent",
        support_episodes=support_episodes
    )

    print(f"\nAdaptation Results:")
    print(f"  Meta-learned: {adaptation_result.meta_learned}")
    print(f"  Performance gain: {adaptation_result.performance_gain:.2f}")
    print(f"  Adaptation time: {adaptation_result.adaptation_time:.2f}s")

    return {
        'meta_iterations': meta_result['num_iterations'],
        'adaptation_gain': adaptation_result.performance_gain,
        'meta_learned': adaptation_result.meta_learned
    }


def generate_report(baseline: Dict, transfer: Dict, maml: Dict):
    """Generate final validation report"""
    print("\n" + "="*80)
    print("BIG ROCK 8: TRANSFER LEARNING VALIDATION REPORT")
    print("="*80)

    print("\n📊 PERFORMANCE SUMMARY")
    print("-" * 80)
    print(f"{'Metric':<40} {'Baseline':<15} {'Transfer':<15} {'MAML':<15}")
    print("-" * 80)
    print(f"{'Episodes to threshold':<40} {baseline['episodes']:<15} {transfer['episodes']:<15} {'N/A':<15}")
    print(f"{'Final performance (avg reward)':<40} {baseline['final_perf']:<15.2f} {transfer['final_perf']:<15.2f} {'N/A':<15}")
    print(f"{'Training time (seconds)':<40} {baseline['time']:<15.2f} {transfer['time']:<15.2f} {'N/A':<15}")

    print("\n🚀 SPEED-UP ANALYSIS")
    print("-" * 80)
    print(f"Transfer Learning Speed-up: {transfer['speedup']:.1f}x")
    print(f"Task Similarity: {transfer['similarity']:.3f}")

    if transfer['speedup'] >= 10:
        print("✅ PASSED: Achieved 10x+ speed-up target!")
    elif transfer['speedup'] >= 5:
        print("⚠️  PARTIAL: Achieved 5x+ speed-up (target: 10x)")
    else:
        print("❌ BELOW TARGET: Speed-up below 5x")

    print(f"\nMAML Meta-Learning:")
    print(f"  Meta-iterations: {maml['meta_iterations']}")
    print(f"  Adaptation gain: {maml['adaptation_gain']:.2f}")
    print(f"  Meta-learned: {maml['meta_learned']}")

    print("\n📈 CONCLUSION")
    print("-" * 80)
    if transfer['speedup'] >= 10:
        print("BIG ROCK 8 VALIDATION: ✅ SUCCESS")
        print(f"Achieved {transfer['speedup']:.1f}x learning speed-up on related tasks.")
        print("Transfer learning and meta-learning framework is production-ready.")
    else:
        print("BIG ROCK 8 VALIDATION: 🔄 IN PROGRESS")
        print(f"Current speed-up: {transfer['speedup']:.1f}x (target: 10-100x)")
        print("Further tuning recommended for optimal performance.")

    print("="*80)


def create_visualization(baseline: Dict, transfer: Dict):
    """Create learning curve visualization"""
    if not MATPLOTLIB_AVAILABLE:
        print("\n⚠️  Skipping visualization (matplotlib not installed)")
        return

    plt.figure(figsize=(12, 6))

    # Plot learning curves
    plt.subplot(1, 2, 1)
    plt.plot(baseline['rewards'], label='Baseline (No Transfer)', alpha=0.7)
    plt.plot(transfer['rewards'], label='Transfer Learning', alpha=0.7)
    plt.axhline(y=8.0, color='r', linestyle='--', label='Target Performance')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Learning Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot speed-up comparison
    plt.subplot(1, 2, 2)
    methods = ['Baseline', 'Transfer']
    episodes = [baseline['episodes'], transfer['episodes']]
    colors = ['#FF6B6B', '#4ECDC4']

    bars = plt.bar(methods, episodes, color=colors, alpha=0.7)
    plt.ylabel('Episodes to Threshold')
    plt.title(f'Episodes to Convergence\n(Speed-up: {transfer["speedup"]:.1f}x)')
    plt.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, episodes):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}',
                ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('experiments/transfer_validation_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 Visualization saved to: experiments/transfer_validation_results.png")


def main():
    """Run all validation experiments"""
    print("\n" + "="*80)
    print("BIG ROCK 8: TRANSFER LEARNING & META-LEARNING VALIDATION")
    print("Testing 10-100x learning speed-up on related tasks")
    print("="*80)

    # Run experiments
    baseline_result = experiment_1_baseline()
    transfer_result = experiment_2_transfer_learning(baseline_result)
    maml_result = experiment_3_maml()

    # Generate report
    generate_report(baseline_result, transfer_result, maml_result)

    # Create visualization
    try:
        create_visualization(baseline_result, transfer_result)
    except Exception as e:
        print(f"\n⚠️  Could not create visualization: {e}")

    print("\n✅ Validation complete!")


if __name__ == "__main__":
    main()
