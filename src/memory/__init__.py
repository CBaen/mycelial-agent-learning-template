"""
Episodic Memory Module

This module implements episodic memory with prioritized experience replay
for the Mycelial Agent Engine (MAE).

Components:
- SumTree: Binary tree data structure for O(log N) prioritized sampling
- PrioritizedReplayBuffer: Experience replay buffer with priority-based sampling
- EpisodicMemory: High-level episodic memory interface
- MemoryConsolidator: Offline learning and memory consolidation
- SemanticRetriever: Vector DB integration for semantic memory retrieval
- HindsightReplay: Goal relabeling for learning from failures (HER)

Research Foundation:
- Schaul et al. (2016): Prioritized Experience Replay
- Andrychowicz et al. (2017): Hindsight Experience Replay
- Blundell et al. (2016): Model-Free Episodic Control
- Stickgold (2005): Memory consolidation during sleep
"""

from .sum_tree import SumTree
from .prioritized_replay_buffer import PrioritizedReplayBuffer
from .episodic_memory import EpisodicMemory, Experience
from .memory_consolidator import MemoryConsolidator, ConsolidationStrategy, ConsolidationResult
from .semantic_retriever import SemanticRetriever, SemanticQuery
from .hindsight_replay import HindsightReplay, HERStrategy, HERTransition, GoalEnv

__all__ = [
    'SumTree',
    'PrioritizedReplayBuffer',
    'EpisodicMemory',
    'Experience',
    'MemoryConsolidator',
    'ConsolidationStrategy',
    'ConsolidationResult',
    'SemanticRetriever',
    'SemanticQuery',
    'HindsightReplay',
    'HERStrategy',
    'HERTransition',
    'GoalEnv',
]
