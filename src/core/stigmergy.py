"""
Stigmergic Environment for MAE v3.0 (Big Rock 6)

Provides pheromone-like marker system for indirect agent coordination.
Inspired by ant colony optimization and swarm intelligence patterns.

Key Features:
- Marker deposition and sensing
- Natural decay (exponential, linear, asymptotic)
- Spatial diffusion
- Efficient spatial queries
- Thread-safe operations

Author: MAE Development Team
Date: 2025-11-12
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional, Set
from collections import defaultdict
import time
import math
import threading
import logging

from src.core.spatial_index import SpatialIndex
from src.core.marker_types import MarkerType, get_recommended_decay_rate

logger = logging.getLogger(__name__)


@dataclass
class StigmergicMarker:
    """
    Represents a pheromone-like marker in the stigmergic environment.

    Markers are deposited by agents at spatial positions and decay over time,
    creating dynamic environmental information that guides swarm behavior.

    Analogous to ant pheromone trails that evaporate over time.
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
        """
        Get marker age in seconds.

        Returns:
            Age since creation in seconds
        """
        return time.time() - self.timestamp

    def time_since_update(self) -> float:
        """
        Get time since last update/reinforcement.

        Returns:
            Seconds since last update
        """
        return time.time() - self.last_update

    def compute_current_intensity(self, decay_type: str = "exponential") -> float:
        """
        Compute current intensity after decay.

        Args:
            decay_type: Type of decay ("exponential", "linear", "asymptotic")

        Returns:
            Current intensity (0.0-1.0)
        """
        elapsed = self.time_since_update()

        if decay_type == "exponential":
            # I(t) = I_0 * e^(-λt)
            decayed = self.intensity * math.exp(-self.decay_rate * elapsed)

        elif decay_type == "linear":
            # I(t) = I_0 - λt
            decayed = self.intensity - (self.decay_rate * elapsed)

        elif decay_type == "asymptotic":
            # I(t) = I_0 / (1 + λt)
            decayed = self.intensity / (1 + self.decay_rate * elapsed)

        else:
            raise ValueError(f"Unknown decay type: {decay_type}")

        return max(0.0, min(1.0, decayed))

    def reinforce(self, amount: float, max_intensity: float = 1.0):
        """
        Reinforce marker intensity (with saturation).

        Args:
            amount: Amount to add to intensity
            max_intensity: Maximum intensity (saturation limit)
        """
        self.intensity = min(max_intensity, self.intensity + amount)
        self.last_update = time.time()

    def decay_to_current(self, decay_type: str = "exponential"):
        """
        Apply decay and update stored intensity to current value.

        Args:
            decay_type: Type of decay to apply
        """
        self.intensity = self.compute_current_intensity(decay_type)
        self.last_update = time.time()


class StigmergicEnvironment:
    """
    Stigmergic environment for indirect agent coordination.

    Provides pheromone-like marker system inspired by ant colonies:
    - Agents deposit markers at spatial positions
    - Markers decay over time (evaporation)
    - Agents sense nearby markers
    - Markers can be reinforced or diffused

    This enables emergent swarm behaviors without direct communication.

    Performance:
    - Marker deposition: O(1) with spatial indexing
    - Marker sensing: O(k) where k = markers in radius
    - Environment step: O(n) where n = total markers
    """

    def __init__(
        self,
        dimensions: int = 2,
        cell_size: float = 1.0,
        default_decay_rate: float = 0.1,
        decay_type: str = "exponential",
        enable_diffusion: bool = True,
        diffusion_rate: float = 0.05,
        diffusion_interval: int = 5,
        max_markers: int = 100000,
        cleanup_interval: int = 100,
        min_intensity: float = 0.01
    ):
        """
        Initialize stigmergic environment.

        Args:
            dimensions: Spatial dimensions (2 or 3)
            cell_size: Grid cell size for spatial indexing
            default_decay_rate: Default marker decay rate (per second)
            decay_type: Decay function ("exponential", "linear", "asymptotic")
            enable_diffusion: Enable marker diffusion to neighbors
            diffusion_rate: Rate of diffusion (0-1)
            diffusion_interval: Steps between diffusion operations
            max_markers: Maximum markers before cleanup
            cleanup_interval: Steps between cleanup operations
            min_intensity: Minimum intensity threshold for removal
        """
        if dimensions not in (2, 3):
            raise ValueError("Only 2D and 3D spaces supported")

        if decay_type not in ("exponential", "linear", "asymptotic"):
            raise ValueError(f"Unknown decay type: {decay_type}")

        self.dimensions = dimensions
        self.default_decay_rate = default_decay_rate
        self.decay_type = decay_type
        self.enable_diffusion = enable_diffusion
        self.diffusion_rate = diffusion_rate
        self.diffusion_interval = diffusion_interval
        self.max_markers = max_markers
        self.cleanup_interval = cleanup_interval
        self.min_intensity = min_intensity

        # Marker storage
        self.markers: Dict[str, StigmergicMarker] = {}
        self.markers_by_type: Dict[str, Set[str]] = defaultdict(set)

        # Spatial indexing for fast queries
        self.spatial_index = SpatialIndex(cell_size, dimensions)

        # Statistics
        self.total_deposited = 0
        self.total_decayed = 0
        self.total_reinforced = 0
        self.total_diffused = 0
        self.step_count = 0

        # Thread safety
        self.lock = threading.RLock()

        logger.info(f"StigmergicEnvironment initialized: {dimensions}D space, "
                   f"cell_size={cell_size}, decay={decay_type}")

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

        If a marker of the same type exists at nearby position (within half
        a cell), it will be reinforced instead of creating a new marker.

        Args:
            marker_type: Type of marker (SUCCESS, DANGER, etc.)
            position: N-dimensional position
            agent_id: ID of depositing agent
            intensity: Initial intensity (0.0-1.0)
            decay_rate: Custom decay rate (uses default or type-based if None)
            metadata: Additional marker data

        Returns:
            Marker ID

        Raises:
            ValueError: If invalid position or intensity
        """
        if len(position) != self.dimensions:
            raise ValueError(f"Position must have {self.dimensions} dimensions")

        if not 0 <= intensity <= 1:
            raise ValueError("Intensity must be in range [0, 1]")

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
                logger.debug(f"Reinforced {marker_type} marker {marker_id} "
                           f"at {position} (intensity now {self.markers[marker_id].intensity:.2f})")
                return marker_id

            # Determine decay rate
            if decay_rate is None:
                decay_rate = get_recommended_decay_rate(marker_type)
                if decay_rate is None:
                    decay_rate = self.default_decay_rate

            # Create new marker
            marker_id = f"marker_{time.time_ns()}_{agent_id}"

            marker = StigmergicMarker(
                marker_id=marker_id,
                marker_type=marker_type,
                position=position,
                intensity=min(1.0, intensity),
                deposited_by=agent_id,
                decay_rate=decay_rate,
                metadata=metadata or {}
            )

            self.markers[marker_id] = marker
            self.markers_by_type[marker_type].add(marker_id)
            self.spatial_index.add(marker_id, position)
            self.total_deposited += 1

            logger.debug(f"Deposited {marker_type} marker at {position} "
                        f"by {agent_id} (intensity={intensity:.2f}, decay={decay_rate:.3f})")

            # Cleanup if needed
            if len(self.markers) > self.max_markers:
                self._cleanup_weak_markers()

            return marker_id

    def sense_markers(
        self,
        position: Tuple[float, ...],
        radius: float,
        marker_types: Optional[List[str]] = None,
        min_intensity: Optional[float] = None
    ) -> List[StigmergicMarker]:
        """
        Sense markers within radius of position.

        Args:
            position: Query position
            radius: Sensing radius
            marker_types: Filter by marker types (None = all types)
            min_intensity: Minimum intensity threshold (uses default if None)

        Returns:
            List of markers sorted by intensity (strongest first)

        Raises:
            ValueError: If invalid position or radius
        """
        if len(position) != self.dimensions:
            raise ValueError(f"Position must have {self.dimensions} dimensions")

        if radius < 0:
            raise ValueError("radius must be non-negative")

        if min_intensity is None:
            min_intensity = self.min_intensity

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
                current_intensity = marker.compute_current_intensity(self.decay_type)

                # Filter by intensity
                if current_intensity < min_intensity:
                    continue

                results.append(marker)

            # Sort by intensity (strongest first)
            results.sort(
                key=lambda m: m.compute_current_intensity(self.decay_type),
                reverse=True
            )

            return results

    def get_gradient(
        self,
        position: Tuple[float, ...],
        marker_type: str,
        radius: float = 5.0,
        attractive: bool = True
    ) -> Tuple[float, ...]:
        """
        Compute gradient vector pointing toward/away from strongest markers.

        This is used for trail following - agents move in direction
        of increasing (attractive) or decreasing (repulsive) marker intensity.

        Args:
            position: Current position
            marker_type: Type of marker to follow
            radius: Sensing radius
            attractive: True = move toward markers, False = move away

        Returns:
            Normalized gradient vector (direction to move)

        Raises:
            ValueError: If invalid position
        """
        if len(position) != self.dimensions:
            raise ValueError(f"Position must have {self.dimensions} dimensions")

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
            intensity = marker.compute_current_intensity(self.decay_type)

            # Direction vector (toward marker)
            direction = tuple(
                marker.position[i] - position[i]
                for i in range(self.dimensions)
            )

            # If repulsive, invert direction
            if not attractive:
                direction = tuple(-d for d in direction)

            # Weight by intensity (stronger markers have more influence)
            for i in range(self.dimensions):
                gradient[i] += direction[i] * intensity
            total_weight += intensity

        # Normalize by total weight
        if total_weight > 0:
            gradient = tuple(g / total_weight for g in gradient)

        # Normalize to unit vector
        magnitude = math.sqrt(sum(g**2 for g in gradient))
        if magnitude > 0:
            gradient = tuple(g / magnitude for g in gradient)

        return gradient

    def get_strongest_marker(
        self,
        position: Tuple[float, ...],
        radius: float,
        marker_type: Optional[str] = None
    ) -> Optional[StigmergicMarker]:
        """
        Get strongest marker within radius.

        Args:
            position: Query position
            radius: Search radius
            marker_type: Filter by type (None = all types)

        Returns:
            Strongest marker or None if no markers found
        """
        marker_types = [marker_type] if marker_type else None
        markers = self.sense_markers(position, radius, marker_types)

        return markers[0] if markers else None

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
                marker.decay_to_current(self.decay_type)

                if marker.intensity < self.min_intensity:
                    to_remove.append(marker_id)

            # Remove decayed markers
            for marker_id in to_remove:
                self._remove_marker(marker_id)
                self.total_decayed += 1

            # Diffusion
            if (self.enable_diffusion and
                self.step_count % self.diffusion_interval == 0):
                self._apply_diffusion()

            # Periodic cleanup
            if self.step_count % self.cleanup_interval == 0:
                self._cleanup_weak_markers()

            logger.debug(f"Stigmergy step {self.step_count}: "
                        f"{len(self.markers)} active markers, "
                        f"{self.total_decayed} decayed")

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

        Simulates physical diffusion of pheromones to nearby locations.
        Only strong markers (intensity > 0.3) diffuse.
        """
        diffusions = []

        for marker_id, marker in list(self.markers.items()):
            if marker.intensity < 0.3:  # Only strong markers diffuse
                continue

            # Amount to diffuse
            diffuse_amount = marker.intensity * self.diffusion_rate

            # Get neighboring positions
            neighbors = self._get_neighbor_positions(marker.position)

            # Distribute diffusion amount to neighbors
            for neighbor_pos in neighbors:
                diffusions.append((
                    marker.marker_type,
                    neighbor_pos,
                    marker.deposited_by,
                    diffuse_amount / len(neighbors),
                    marker.decay_rate
                ))

        # Apply diffusions
        for marker_type, position, agent_id, intensity, decay_rate in diffusions:
            self.deposit_marker(
                marker_type,
                position,
                agent_id,
                intensity,
                decay_rate
            )
            self.total_diffused += 1

    def _get_neighbor_positions(
        self,
        position: Tuple[float, ...]
    ) -> List[Tuple[float, ...]]:
        """
        Get neighboring grid positions for diffusion.

        Returns positions of adjacent grid cells.
        """
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
            key=lambda x: x[1].compute_current_intensity(self.decay_type)
        )

        # Remove weakest 20%
        remove_count = int(len(sorted_markers) * 0.2)
        for marker_id, _ in sorted_markers[:remove_count]:
            self._remove_marker(marker_id)

        logger.info(f"Cleaned up {remove_count} weak markers (total now: {len(self.markers)})")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get environment statistics.

        Returns:
            Dictionary with marker counts, spatial stats, and counters
        """
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
                'total_diffused': self.total_diffused,
                'step_count': self.step_count,
                'spatial_index': self.spatial_index.get_statistics()
            }

    def get_marker_by_id(self, marker_id: str) -> Optional[StigmergicMarker]:
        """
        Get marker by ID.

        Args:
            marker_id: Marker identifier

        Returns:
            Marker or None if not found
        """
        with self.lock:
            return self.markers.get(marker_id)

    def get_markers_by_type(self, marker_type: str) -> List[StigmergicMarker]:
        """
        Get all markers of a specific type.

        Args:
            marker_type: Type of markers to retrieve

        Returns:
            List of markers of that type
        """
        with self.lock:
            marker_ids = self.markers_by_type.get(marker_type, set())
            return [self.markers[mid] for mid in marker_ids if mid in self.markers]

    def clear(self):
        """Remove all markers from environment"""
        with self.lock:
            self.markers.clear()
            self.markers_by_type.clear()
            self.spatial_index.clear()
            logger.info("Stigmergic environment cleared")

    def __len__(self) -> int:
        """Return number of markers in environment"""
        return len(self.markers)

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"StigmergicEnvironment({self.dimensions}D, "
                f"markers={stats['total_markers']}, "
                f"deposited={stats['total_deposited']}, "
                f"decayed={stats['total_decayed']})")
