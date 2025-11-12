# Big Rock 6: Stigmergic Environment - Implementation Plan

**Project:** Mycelial Agent Engine (MAE) v3.0
**Phase:** 1 (Weeks 2-4)
**Week:** 2
**Status:** 🚧 IN PROGRESS
**Start Date:** 2025-11-12
**Target Completion:** 2025-11-17 (5 days)

---

## Table of Contents

1. [Overview](#overview)
2. [Research Foundation](#research-foundation)
3. [Architecture Design](#architecture-design)
4. [Implementation Specification](#implementation-specification)
5. [Integration Points](#integration-points)
6. [Performance Targets](#performance-targets)
7. [Testing Strategy](#testing-strategy)
8. [Timeline](#timeline)
9. [Success Criteria](#success-criteria)

---

## Overview

### What is Stigmergy?

**Stigmergy** is a mechanism of indirect coordination where agents communicate by modifying their shared environment. The modifications serve as stimuli for subsequent actions by agents, creating self-organizing patterns without central coordination.

**Origins:** First observed in social insects (ants, termites, bees):
- Ants laying pheromone trails to food sources
- Termites building complex structures through local rules
- Bees coordinating hive construction

### Why Stigmergy for MAE?

Traditional agent communication (messages, signals) is **direct** - agents explicitly communicate intent. Stigmergy enables **indirect** communication through environmental markers:

1. **Scalability**: No need to know which agents to communicate with
2. **Emergence**: Complex behaviors arise from simple rules
3. **Persistence**: Information persists in the environment
4. **Adaptability**: Markers naturally decay or strengthen based on utility
5. **Decentralization**: No single point of coordination or failure

### Big Rock 6 Goals

Implement a **Stigmergic Environment** that enables:
- ✅ Pheromone-like marker deposition and sensing
- ✅ Multiple marker types (SUCCESS, DANGER, EXPLORATION, RESOURCE, CONVERGENCE)
- ✅ Natural decay mechanisms (exponential, linear)
- ✅ Marker reinforcement and diffusion
- ✅ Spatial indexing for efficient queries
- ✅ Visualization of stigmergic patterns
- ✅ Integration with existing agent system

---

## Research Foundation

### From MAE Enhancement Proposal Research

**Key Findings:**
1. **Ant Colony Optimization (ACO)** has 45% market share in swarm intelligence applications
2. Pheromone trails enable optimal path finding without central coordination
3. Two types of pheromones in nature:
   - **Attractive**: Guide toward resources
   - **Repulsive**: Mark danger or explored areas
4. Decay rates critical for adaptability (typical: 0.1-0.5 per time step)

### Natural Pheromone Properties

| Property | Natural System | MAE Implementation |
|----------|----------------|-------------------|
| **Deposition** | Agents leave chemical trails | Agents deposit markers with intensity |
| **Evaporation** | Chemicals decay over time | Exponential/linear decay |
| **Diffusion** | Spreads to nearby areas | Spatial diffusion to neighbors |
| **Reinforcement** | Repeated paths strengthen | Multiple deposits increase intensity |
| **Sensing** | Ants sense with antennae | Agents query local markers |
| **Attraction** | Follow strongest trail | Move toward high-intensity markers |

### Stigmergy Types

**1. Sematectonic Stigmergy** (Marker-Based)
- Environmental markers persist
- Examples: Pheromone trails, scent markers
- **Use in MAE**: Success trails, danger zones, exploration markers

**2. Sign-Based Stigmergy** (Structural)
- Environment structure modified
- Examples: Termite mounds, bee combs
- **Use in MAE**: Policy embeddings in Vector DB, learned patterns

For Big Rock 6, we focus on **Sematectonic Stigmergy** (marker-based).

---

## Architecture Design

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    StigmergicEnvironment                        │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   Marker Grid / Space                      │  │
│  │  - Spatial coordinates (continuous or discrete)            │  │
│  │  - Marker storage with intensity values                    │  │
│  │  - Temporal tracking (creation time, last update)          │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │  Marker Types   │  │  Decay System    │  │ Spatial Index │  │
│  │  - SUCCESS      │  │  - Exponential   │  │  - Grid/KDTree│  │
│  │  - DANGER       │  │  - Linear        │  │  - Fast query │  │
│  │  - EXPLORATION  │  │  - Asymptotic    │  │  - Radius     │  │
│  │  - RESOURCE     │  │  - Reinforcement │  │    search     │  │
│  │  - CONVERGENCE  │  │  - Diffusion     │  │               │  │
│  └─────────────────┘  └──────────────────┘  └───────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                  Agent Interface                           │  │
│  │  deposit_marker(type, position, intensity)                │  │
│  │  sense_markers(position, radius, marker_types)            │  │
│  │  get_strongest_marker(position, radius, marker_type)      │  │
│  │  follow_trail(position, marker_type)                      │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MycelialAgent                               │
│  - Current position in stigmergic space                          │
│  - Marker sensing capabilities                                   │
│  - Trail following behaviors                                     │
│  - Integration with electrical signals                           │
└─────────────────────────────────────────────────────────────────┘
```

### Marker Structure

```python
@dataclass
class StigmergicMarker:
    marker_id: str                    # Unique identifier
    marker_type: str                  # SUCCESS, DANGER, etc.
    position: Tuple[float, ...]       # N-dimensional coordinates
    intensity: float                  # Marker strength (0.0-1.0)
    deposited_by: str                 # Agent ID
    timestamp: float                  # Creation time
    last_update: float                # Last reinforcement
    decay_rate: float                 # Decay per time unit
    metadata: Dict[str, Any]          # Additional info
```

### Spatial Representation

**Option 1: Continuous Space**
- Position: (x, y) or (x, y, z) floats
- Pros: Natural movement, precise positioning
- Cons: Need spatial indexing (KD-tree)

**Option 2: Discrete Grid**
- Position: (i, j) integers
- Pros: Simple indexing, fast lookup
- Cons: Discretization artifacts

**Chosen: Continuous Space with Grid-based Spatial Index**
- Best of both worlds
- Agents move in continuous space
- Markers indexed in grid cells for fast queries

---

## Implementation Specification

### File Structure

```
src/core/
├── stigmergy.py              # Main StigmergicEnvironment class
├── marker_types.py           # Marker type definitions
└── spatial_index.py          # Spatial indexing utilities

src/agents/
└── base_agent.py             # Add stigmergy integration

tests/unit/
└── test_stigmergy.py         # Comprehensive tests

docs/
└── BIG_ROCK_6_API_GUIDE.md   # API documentation
```

### Core Classes

#### 1. StigmergicMarker (Dataclass)

```python
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple
import time

@dataclass
class StigmergicMarker:
    """
    Represents a pheromone-like marker in the stigmergic environment.

    Markers are deposited by agents and decay over time, creating
    dynamic environmental information that guides swarm behavior.
    """
    marker_id: str
    marker_type: str
    position: Tuple[float, ...]
    intensity: float
    deposited_by: str
    timestamp: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    decay_rate: float = 0.1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def age(self) -> float:
        """Get marker age in seconds"""
        return time.time() - self.timestamp

    def compute_current_intensity(self) -> float:
        """
        Compute current intensity after decay.

        Uses exponential decay: I(t) = I_0 * e^(-λt)
        where λ is the decay_rate
        """
        import math
        elapsed = time.time() - self.last_update
        decayed = self.intensity * math.exp(-self.decay_rate * elapsed)
        return max(0.0, decayed)

    def reinforce(self, amount: float):
        """Reinforce marker intensity (with saturation)"""
        self.intensity = min(1.0, self.intensity + amount)
        self.last_update = time.time()
```

#### 2. MarkerType (Constants)

```python
class MarkerType:
    """Standard stigmergic marker types"""

    SUCCESS = "SUCCESS"
    """
    Marks successful actions/outcomes (like food trail).
    Attracts agents to replicate successful behaviors.
    """

    DANGER = "DANGER"
    """
    Marks dangerous areas or failed actions (repulsive).
    Warns agents to avoid this region.
    """

    EXPLORATION = "EXPLORATION"
    """
    Marks explored areas to reduce redundant exploration.
    Guides agents toward unexplored regions.
    """

    RESOURCE = "RESOURCE"
    """
    Marks resource locations (data, compute, specialists).
    Attracts agents needing resources.
    """

    CONVERGENCE = "CONVERGENCE"
    """
    Marks areas where agents have converged on policies.
    Helps coordinate team convergence.
    """

    COLLABORATION = "COLLABORATION"
    """
    Marks areas where agents successfully collaborated.
    Encourages team formation.
    """
```

#### 3. SpatialIndex (Grid-based)

```python
from collections import defaultdict
from typing import List, Tuple, Set
import math

class SpatialIndex:
    """
    Grid-based spatial index for fast marker queries.

    Divides continuous space into grid cells for O(1) lookup
    and efficient radius searches.
    """

    def __init__(self, cell_size: float = 1.0, dimensions: int = 2):
        """
        Initialize spatial index.

        Args:
            cell_size: Size of each grid cell
            dimensions: Number of spatial dimensions (2 or 3)
        """
        self.cell_size = cell_size
        self.dimensions = dimensions
        self.grid: Dict[Tuple[int, ...], Set[str]] = defaultdict(set)
        self.marker_positions: Dict[str, Tuple[float, ...]] = {}

    def _get_cell(self, position: Tuple[float, ...]) -> Tuple[int, ...]:
        """Convert continuous position to grid cell"""
        return tuple(int(pos / self.cell_size) for pos in position)

    def add(self, marker_id: str, position: Tuple[float, ...]):
        """Add marker to spatial index"""
        cell = self._get_cell(position)
        self.grid[cell].add(marker_id)
        self.marker_positions[marker_id] = position

    def remove(self, marker_id: str):
        """Remove marker from spatial index"""
        if marker_id in self.marker_positions:
            position = self.marker_positions[marker_id]
            cell = self._get_cell(position)
            self.grid[cell].discard(marker_id)
            del self.marker_positions[marker_id]

    def query_radius(
        self,
        position: Tuple[float, ...],
        radius: float
    ) -> List[str]:
        """
        Find all markers within radius of position.

        Uses grid-based search for efficiency.
        """
        cell = self._get_cell(position)
        cell_radius = int(math.ceil(radius / self.cell_size))

        candidates = set()

        # Check all cells within bounding box
        if self.dimensions == 2:
            for dx in range(-cell_radius, cell_radius + 1):
                for dy in range(-cell_radius, cell_radius + 1):
                    check_cell = (cell[0] + dx, cell[1] + dy)
                    candidates.update(self.grid.get(check_cell, set()))
        else:  # 3D
            for dx in range(-cell_radius, cell_radius + 1):
                for dy in range(-cell_radius, cell_radius + 1):
                    for dz in range(-cell_radius, cell_radius + 1):
                        check_cell = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                        candidates.update(self.grid.get(check_cell, set()))

        # Filter by actual distance
        results = []
        for marker_id in candidates:
            marker_pos = self.marker_positions[marker_id]
            distance = self._distance(position, marker_pos)
            if distance <= radius:
                results.append(marker_id)

        return results

    def _distance(
        self,
        pos1: Tuple[float, ...],
        pos2: Tuple[float, ...]
    ) -> float:
        """Euclidean distance between positions"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))
```

#### 4. StigmergicEnvironment (Main Class)

```python
from typing import List, Optional, Dict, Tuple, Set
import time
import threading
from collections import defaultdict

class StigmergicEnvironment:
    """
    Stigmergic environment for indirect agent coordination.

    Provides pheromone-like marker system inspired by ant colonies:
    - Agents deposit markers at positions
    - Markers decay over time (evaporation)
    - Agents sense nearby markers
    - Markers can be reinforced or diffused

    This enables emergent swarm behaviors without direct communication.
    """

    def __init__(
        self,
        dimensions: int = 2,
        cell_size: float = 1.0,
        default_decay_rate: float = 0.1,
        enable_diffusion: bool = True,
        diffusion_rate: float = 0.05,
        max_markers: int = 100000,
        cleanup_interval: int = 100
    ):
        """
        Initialize stigmergic environment.

        Args:
            dimensions: Spatial dimensions (2 or 3)
            cell_size: Grid cell size for spatial indexing
            default_decay_rate: Default marker decay rate (per second)
            enable_diffusion: Enable marker diffusion to neighbors
            diffusion_rate: Rate of diffusion (0-1)
            max_markers: Maximum markers before cleanup
            cleanup_interval: Steps between cleanup operations
        """
        self.dimensions = dimensions
        self.default_decay_rate = default_decay_rate
        self.enable_diffusion = enable_diffusion
        self.diffusion_rate = diffusion_rate
        self.max_markers = max_markers
        self.cleanup_interval = cleanup_interval

        # Marker storage
        self.markers: Dict[str, StigmergicMarker] = {}
        self.markers_by_type: Dict[str, Set[str]] = defaultdict(set)

        # Spatial indexing
        self.spatial_index = SpatialIndex(cell_size, dimensions)

        # Statistics
        self.total_deposited = 0
        self.total_decayed = 0
        self.total_reinforced = 0
        self.step_count = 0

        # Thread safety
        self.lock = threading.RLock()

        logger.info(f"StigmergicEnvironment initialized: "
                   f"{dimensions}D space, cell_size={cell_size}")

    def deposit_marker(
        self,
        marker_type: str,
        position: Tuple[float, ...],
        agent_id: str,
        intensity: float = 1.0,
        decay_rate: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Deposit a stigmergic marker at a position.

        If a marker of the same type exists at nearby position,
        it will be reinforced instead of creating a new marker.

        Args:
            marker_type: Type of marker (SUCCESS, DANGER, etc.)
            position: N-dimensional position
            agent_id: ID of depositing agent
            intensity: Initial intensity (0.0-1.0)
            decay_rate: Custom decay rate (uses default if None)
            metadata: Additional marker data

        Returns:
            Marker ID
        """
        with self.lock:
            # Check for nearby markers to reinforce
            nearby = self.sense_markers(
                position,
                radius=self.spatial_index.cell_size * 0.5,
                marker_types=[marker_type]
            )

            if nearby:
                # Reinforce existing marker
                marker_id = nearby[0].marker_id
                self.markers[marker_id].reinforce(intensity)
                self.total_reinforced += 1
                logger.debug(f"Reinforced marker {marker_id} at {position}")
                return marker_id

            # Create new marker
            marker_id = f"marker_{time.time_ns()}_{agent_id}"

            marker = StigmergicMarker(
                marker_id=marker_id,
                marker_type=marker_type,
                position=position,
                intensity=min(1.0, intensity),
                deposited_by=agent_id,
                decay_rate=decay_rate or self.default_decay_rate,
                metadata=metadata or {}
            )

            self.markers[marker_id] = marker
            self.markers_by_type[marker_type].add(marker_id)
            self.spatial_index.add(marker_id, position)
            self.total_deposited += 1

            logger.debug(f"Deposited {marker_type} marker at {position} "
                        f"by {agent_id} (intensity={intensity:.2f})")

            # Cleanup if needed
            if len(self.markers) > self.max_markers:
                self._cleanup_weak_markers()

            return marker_id

    def sense_markers(
        self,
        position: Tuple[float, ...],
        radius: float,
        marker_types: Optional[List[str]] = None,
        min_intensity: float = 0.01
    ) -> List[StigmergicMarker]:
        """
        Sense markers within radius of position.

        Args:
            position: Query position
            radius: Sensing radius
            marker_types: Filter by marker types (None = all types)
            min_intensity: Minimum intensity threshold

        Returns:
            List of markers sorted by intensity (strongest first)
        """
        with self.lock:
            # Get nearby marker IDs from spatial index
            candidate_ids = self.spatial_index.query_radius(position, radius)

            # Filter and collect markers
            results = []
            for marker_id in candidate_ids:
                marker = self.markers.get(marker_id)
                if not marker:
                    continue

                # Filter by type
                if marker_types and marker.marker_type not in marker_types:
                    continue

                # Compute current intensity
                current_intensity = marker.compute_current_intensity()

                # Filter by intensity
                if current_intensity < min_intensity:
                    continue

                results.append(marker)

            # Sort by intensity (strongest first)
            results.sort(key=lambda m: m.compute_current_intensity(), reverse=True)

            return results

    def get_gradient(
        self,
        position: Tuple[float, ...],
        marker_type: str,
        radius: float = 5.0
    ) -> Tuple[float, ...]:
        """
        Compute gradient vector pointing toward strongest markers.

        This is used for trail following - agents move in direction
        of increasing marker intensity.

        Args:
            position: Current position
            marker_type: Type of marker to follow
            radius: Sensing radius

        Returns:
            Normalized gradient vector (direction to move)
        """
        markers = self.sense_markers(
            position,
            radius,
            marker_types=[marker_type]
        )

        if not markers:
            return tuple(0.0 for _ in range(self.dimensions))

        # Weighted average of directions to markers
        gradient = [0.0] * self.dimensions
        total_weight = 0.0

        for marker in markers:
            intensity = marker.compute_current_intensity()
            direction = tuple(
                marker.position[i] - position[i]
                for i in range(self.dimensions)
            )

            # Weight by intensity
            for i in range(self.dimensions):
                gradient[i] += direction[i] * intensity
            total_weight += intensity

        # Normalize
        if total_weight > 0:
            gradient = tuple(g / total_weight for g in gradient)

        # Normalize to unit vector
        magnitude = math.sqrt(sum(g**2 for g in gradient))
        if magnitude > 0:
            gradient = tuple(g / magnitude for g in gradient)

        return gradient

    def step(self):
        """
        Advance environment by one time step.

        Handles:
        - Marker decay
        - Diffusion (if enabled)
        - Periodic cleanup
        """
        with self.lock:
            self.step_count += 1

            # Apply decay to all markers
            to_remove = []
            for marker_id, marker in self.markers.items():
                current_intensity = marker.compute_current_intensity()

                if current_intensity < 0.01:  # Marker fully decayed
                    to_remove.append(marker_id)
                else:
                    # Update stored intensity
                    marker.intensity = current_intensity
                    marker.last_update = time.time()

            # Remove decayed markers
            for marker_id in to_remove:
                self._remove_marker(marker_id)
                self.total_decayed += 1

            # Diffusion
            if self.enable_diffusion and self.step_count % 5 == 0:
                self._apply_diffusion()

            # Periodic cleanup
            if self.step_count % self.cleanup_interval == 0:
                self._cleanup_weak_markers()

            logger.debug(f"Stigmergy step {self.step_count}: "
                        f"{len(self.markers)} active markers")

    def _remove_marker(self, marker_id: str):
        """Remove marker from all data structures"""
        if marker_id in self.markers:
            marker = self.markers[marker_id]
            self.markers_by_type[marker.marker_type].discard(marker_id)
            self.spatial_index.remove(marker_id)
            del self.markers[marker_id]

    def _apply_diffusion(self):
        """
        Apply diffusion: spread marker intensity to neighbors.

        Simulates physical diffusion of pheromones.
        """
        # Collect diffusion operations
        diffusions = []

        for marker_id, marker in list(self.markers.items()):
            if marker.intensity < 0.1:  # Only strong markers diffuse
                continue

            # Amount to diffuse
            diffuse_amount = marker.intensity * self.diffusion_rate

            # Find neighboring positions (8 neighbors in 2D, 26 in 3D)
            neighbors = self._get_neighbor_positions(marker.position)

            for neighbor_pos in neighbors:
                diffusions.append((
                    marker.marker_type,
                    neighbor_pos,
                    marker.deposited_by,
                    diffuse_amount / len(neighbors)
                ))

        # Apply diffusions
        for marker_type, position, agent_id, intensity in diffusions:
            self.deposit_marker(marker_type, position, agent_id, intensity)

    def _get_neighbor_positions(
        self,
        position: Tuple[float, ...]
    ) -> List[Tuple[float, ...]]:
        """Get neighboring grid positions"""
        neighbors = []
        cell_size = self.spatial_index.cell_size

        if self.dimensions == 2:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    neighbors.append((
                        position[0] + dx * cell_size,
                        position[1] + dy * cell_size
                    ))
        else:  # 3D
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        neighbors.append((
                            position[0] + dx * cell_size,
                            position[1] + dy * cell_size,
                            position[2] + dz * cell_size
                        ))

        return neighbors

    def _cleanup_weak_markers(self):
        """Remove weakest markers if over limit"""
        if len(self.markers) <= self.max_markers * 0.8:
            return

        # Sort by intensity
        sorted_markers = sorted(
            self.markers.items(),
            key=lambda x: x[1].compute_current_intensity()
        )

        # Remove weakest 20%
        remove_count = int(len(sorted_markers) * 0.2)
        for marker_id, _ in sorted_markers[:remove_count]:
            self._remove_marker(marker_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get environment statistics"""
        with self.lock:
            return {
                'total_markers': len(self.markers),
                'markers_by_type': {
                    mtype: len(ids)
                    for mtype, ids in self.markers_by_type.items()
                },
                'total_deposited': self.total_deposited,
                'total_decayed': self.total_decayed,
                'total_reinforced': self.total_reinforced,
                'step_count': self.step_count
            }
```

---

## Integration Points

### 1. MycelialAgent Extensions

Add to `base_agent.py`:

```python
class MycelialAgent(Agent):
    def __init__(self, ..., stigmergy_env: Optional[StigmergicEnvironment] = None):
        # ... existing init ...

        # BIG ROCK 6: STIGMERGY
        self.stigmergy_env = stigmergy_env
        self.stigmergy_position: Tuple[float, ...] = (0.0, 0.0)  # 2D default
        self.sensing_radius: float = 5.0

    def deposit_success_marker(self, intensity: float = 1.0):
        """Deposit SUCCESS marker at current position"""
        if self.stigmergy_env:
            self.stigmergy_env.deposit_marker(
                MarkerType.SUCCESS,
                self.stigmergy_position,
                self.agent_id,
                intensity
            )

    def deposit_danger_marker(self, risk_level: float):
        """Deposit DANGER marker at current position"""
        if self.stigmergy_env:
            self.stigmergy_env.deposit_marker(
                MarkerType.DANGER,
                self.stigmergy_position,
                self.agent_id,
                intensity=risk_level,
                metadata={'risk_score': self.risk_score}
            )

    def sense_environment(self) -> Dict[str, List[StigmergicMarker]]:
        """Sense all markers in sensing radius"""
        if not self.stigmergy_env:
            return {}

        markers = self.stigmergy_env.sense_markers(
            self.stigmergy_position,
            self.sensing_radius
        )

        # Group by type
        by_type = defaultdict(list)
        for marker in markers:
            by_type[marker.marker_type].append(marker)

        return dict(by_type)

    def follow_success_trail(self) -> Tuple[float, ...]:
        """Get direction to follow SUCCESS markers"""
        if not self.stigmergy_env:
            return tuple(0.0 for _ in range(len(self.stigmergy_position)))

        return self.stigmergy_env.get_gradient(
            self.stigmergy_position,
            MarkerType.SUCCESS,
            radius=self.sensing_radius
        )

    def move_in_stigmergy(self, direction: Tuple[float, ...], step_size: float = 1.0):
        """Move in stigmergic space"""
        new_position = tuple(
            self.stigmergy_position[i] + direction[i] * step_size
            for i in range(len(self.stigmergy_position))
        )
        self.stigmergy_position = new_position
```

### 2. MycelialModel Integration

```python
class MycelialModel(Model):
    def __init__(self, ...):
        # ... existing init ...

        # BIG ROCK 6: STIGMERGY
        self.stigmergy_env = StigmergicEnvironment(
            dimensions=2,
            cell_size=1.0,
            enable_diffusion=True
        )

    def step(self):
        # Advance stigmergic environment
        self.stigmergy_env.step()

        # ... existing step logic ...
```

---

## Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Marker Deposition | < 0.1ms | Must be faster than Redis write |
| Marker Sensing (radius query) | < 1ms | Fast enough for real-time decision |
| Gradient Computation | < 2ms | Complex calculation acceptable |
| Environment Step | < 10ms | Per-step overhead acceptable |
| Max Markers | 100,000+ | Support large swarms |
| Memory per Marker | < 200 bytes | Keep memory manageable |

---

## Testing Strategy

### Unit Tests (30+ test cases)

1. **Marker Tests**
   - Creation, decay, reinforcement
   - Intensity computation over time
   - Metadata handling

2. **Spatial Index Tests**
   - Cell mapping
   - Radius queries (2D and 3D)
   - Edge cases (boundaries, empty cells)

3. **Environment Tests**
   - Marker deposition and sensing
   - Decay mechanics
   - Diffusion behavior
   - Cleanup mechanisms
   - Thread safety

4. **Integration Tests**
   - Agent marker deposition
   - Trail following
   - Emergent behaviors (path optimization)

### Performance Benchmarks

```python
def test_deposition_performance():
    """Verify deposition meets <0.1ms target"""
    env = StigmergicEnvironment()

    start = time.perf_counter()
    for i in range(1000):
        env.deposit_marker(MarkerType.SUCCESS, (i, i), "agent_1")
    elapsed = (time.perf_counter() - start) / 1000

    assert elapsed < 0.0001  # 0.1ms
```

---

## Timeline

### Day 1: Core Implementation (2025-11-12)
- ✅ Implementation plan documented
- ⏳ `StigmergicMarker` dataclass
- ⏳ `MarkerType` constants
- ⏳ `SpatialIndex` class

### Day 2: Environment Implementation (2025-11-13)
- ⏳ `StigmergicEnvironment` class
- ⏳ Deposit and sensing methods
- ⏳ Decay mechanics
- ⏳ Gradient computation

### Day 3: Advanced Features (2025-11-14)
- ⏳ Diffusion system
- ⏳ Cleanup mechanisms
- ⏳ Statistics tracking
- ⏳ Integration with base_agent.py

### Day 4: Testing (2025-11-15)
- ⏳ Unit tests (30+ cases)
- ⏳ Performance benchmarks
- ⏳ Integration tests
- ⏳ Emergent behavior demos

### Day 5: Documentation (2025-11-16)
- ⏳ API documentation
- ⏳ Usage examples
- ⏳ Visualization utilities
- ⏳ Update PROGRESS.md

---

## Success Criteria

✅ **Implementation Complete**
- All classes implemented and integrated
- Agent methods for stigmergy working
- Decay, diffusion, and cleanup functional

✅ **Performance Targets Met**
- Deposition < 0.1ms
- Sensing < 1ms
- Environment step < 10ms

✅ **Tests Passing**
- 30+ unit tests at 90%+ coverage
- Performance benchmarks passing
- Integration tests demonstrating emergent behavior

✅ **Documentation Complete**
- Implementation plan (this document)
- API guide with examples
- Usage patterns documented

✅ **Emergent Behaviors Demonstrated**
- Path optimization (shortest path emerges)
- Danger avoidance (agents avoid marked areas)
- Resource allocation (agents distribute to resources)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| **Performance degradation with many markers** | Spatial indexing + periodic cleanup |
| **Memory exhaustion** | Max marker limit + weak marker removal |
| **Diffusion instability** | Low diffusion rate (5%) + decay balancing |
| **Thread safety issues** | RLock on all environment operations |
| **Integration complexity** | Optional parameter - backward compatible |

---

## References

1. **MAE Enhancement Proposal** - Stigmergy research findings
2. **Ant Colony Optimization** - Dorigo & Stützle (2004)
3. **Big Rock 5: Electrical Signaling** - Integration pattern reference
4. **Nature: Stigmergy in Social Insects** - Theraulaz & Bonabeau (1999)

---

**Next Steps:** Begin implementation of core classes (Day 1)

**Status:** 🚧 Implementation plan complete, ready to code
