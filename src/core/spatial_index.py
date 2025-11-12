"""
Spatial Index for Stigmergic Environment (Big Rock 6)

Provides efficient spatial queries for markers using grid-based indexing.
Enables fast radius searches in O(k) where k is result size, not total markers.

Author: MAE Development Team
Date: 2025-11-12
"""

from collections import defaultdict
from typing import List, Tuple, Set, Dict
import math


class SpatialIndex:
    """
    Grid-based spatial index for fast marker queries.

    Divides continuous space into grid cells for O(1) cell lookup
    and efficient radius searches. Supports 2D and 3D spaces.

    Performance:
    - Cell lookup: O(1)
    - Radius query: O(k) where k = markers in radius
    - Add/Remove: O(1)
    """

    def __init__(self, cell_size: float = 1.0, dimensions: int = 2):
        """
        Initialize spatial index.

        Args:
            cell_size: Size of each grid cell
            dimensions: Number of spatial dimensions (2 or 3)
        """
        if dimensions not in (2, 3):
            raise ValueError("Only 2D and 3D spaces supported")

        if cell_size <= 0:
            raise ValueError("cell_size must be positive")

        self.cell_size = cell_size
        self.dimensions = dimensions

        # Grid: cell coords -> set of marker IDs
        self.grid: Dict[Tuple[int, ...], Set[str]] = defaultdict(set)

        # Marker positions for distance calculations
        self.marker_positions: Dict[str, Tuple[float, ...]] = {}

    def _get_cell(self, position: Tuple[float, ...]) -> Tuple[int, ...]:
        """
        Convert continuous position to grid cell coordinates.

        Args:
            position: Continuous position (x, y) or (x, y, z)

        Returns:
            Grid cell coordinates as integers
        """
        if len(position) != self.dimensions:
            raise ValueError(f"Position must have {self.dimensions} dimensions")

        return tuple(int(math.floor(pos / self.cell_size)) for pos in position)

    def add(self, marker_id: str, position: Tuple[float, ...]):
        """
        Add marker to spatial index.

        Args:
            marker_id: Unique marker identifier
            position: Marker position

        Raises:
            ValueError: If marker already exists or invalid position
        """
        if marker_id in self.marker_positions:
            raise ValueError(f"Marker {marker_id} already in index")

        if len(position) != self.dimensions:
            raise ValueError(f"Position must have {self.dimensions} dimensions")

        cell = self._get_cell(position)
        self.grid[cell].add(marker_id)
        self.marker_positions[marker_id] = position

    def remove(self, marker_id: str) -> bool:
        """
        Remove marker from spatial index.

        Args:
            marker_id: Marker identifier to remove

        Returns:
            True if marker was removed, False if not found
        """
        if marker_id not in self.marker_positions:
            return False

        position = self.marker_positions[marker_id]
        cell = self._get_cell(position)

        self.grid[cell].discard(marker_id)

        # Clean up empty cells
        if not self.grid[cell]:
            del self.grid[cell]

        del self.marker_positions[marker_id]
        return True

    def update(self, marker_id: str, new_position: Tuple[float, ...]):
        """
        Update marker position (remove from old cell, add to new).

        Args:
            marker_id: Marker identifier
            new_position: New position

        Raises:
            ValueError: If marker not found or invalid position
        """
        if marker_id not in self.marker_positions:
            raise ValueError(f"Marker {marker_id} not in index")

        # Remove from old cell
        old_position = self.marker_positions[marker_id]
        old_cell = self._get_cell(old_position)
        self.grid[old_cell].discard(marker_id)

        if not self.grid[old_cell]:
            del self.grid[old_cell]

        # Add to new cell
        new_cell = self._get_cell(new_position)
        self.grid[new_cell].add(marker_id)
        self.marker_positions[marker_id] = new_position

    def query_radius(
        self,
        position: Tuple[float, ...],
        radius: float
    ) -> List[str]:
        """
        Find all markers within radius of position.

        Uses grid-based search:
        1. Determine bounding box of cells to check
        2. Collect candidate markers from those cells
        3. Filter by exact distance

        Args:
            position: Query position
            radius: Search radius

        Returns:
            List of marker IDs within radius

        Complexity: O(k) where k = markers in radius
        """
        if radius < 0:
            raise ValueError("radius must be non-negative")

        if len(position) != self.dimensions:
            raise ValueError(f"Position must have {self.dimensions} dimensions")

        # Get query cell
        center_cell = self._get_cell(position)

        # Determine how many cells to search in each direction
        cell_radius = int(math.ceil(radius / self.cell_size))

        # Collect candidates from nearby cells
        candidates = set()

        if self.dimensions == 2:
            for dx in range(-cell_radius, cell_radius + 1):
                for dy in range(-cell_radius, cell_radius + 1):
                    check_cell = (center_cell[0] + dx, center_cell[1] + dy)
                    candidates.update(self.grid.get(check_cell, set()))

        else:  # 3D
            for dx in range(-cell_radius, cell_radius + 1):
                for dy in range(-cell_radius, cell_radius + 1):
                    for dz in range(-cell_radius, cell_radius + 1):
                        check_cell = (
                            center_cell[0] + dx,
                            center_cell[1] + dy,
                            center_cell[2] + dz
                        )
                        candidates.update(self.grid.get(check_cell, set()))

        # Filter by actual distance
        results = []
        radius_squared = radius * radius  # Avoid sqrt in loop

        for marker_id in candidates:
            marker_pos = self.marker_positions[marker_id]
            dist_squared = self._distance_squared(position, marker_pos)

            if dist_squared <= radius_squared:
                results.append(marker_id)

        return results

    def query_cell(self, position: Tuple[float, ...]) -> List[str]:
        """
        Get all markers in the same cell as position.

        Args:
            position: Query position

        Returns:
            List of marker IDs in same cell
        """
        cell = self._get_cell(position)
        return list(self.grid.get(cell, set()))

    def get_neighbors(
        self,
        position: Tuple[float, ...],
        include_diagonals: bool = True
    ) -> List[str]:
        """
        Get markers in neighboring cells.

        Args:
            position: Query position
            include_diagonals: Include diagonal neighbors

        Returns:
            List of marker IDs in neighboring cells
        """
        cell = self._get_cell(position)
        neighbors = []

        if self.dimensions == 2:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue

                    if not include_diagonals and dx != 0 and dy != 0:
                        continue

                    neighbor_cell = (cell[0] + dx, cell[1] + dy)
                    neighbors.extend(self.grid.get(neighbor_cell, set()))

        else:  # 3D
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue

                        if not include_diagonals:
                            # Only face neighbors (not edge or corner)
                            nonzero_count = sum(1 for d in [dx, dy, dz] if d != 0)
                            if nonzero_count > 1:
                                continue

                        neighbor_cell = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                        neighbors.extend(self.grid.get(neighbor_cell, set()))

        return neighbors

    def get_k_nearest(
        self,
        position: Tuple[float, ...],
        k: int,
        max_radius: float = float('inf')
    ) -> List[Tuple[str, float]]:
        """
        Get k nearest markers to position.

        Args:
            position: Query position
            k: Number of nearest markers to return
            max_radius: Maximum search radius

        Returns:
            List of (marker_id, distance) tuples, sorted by distance
        """
        if k <= 0:
            return []

        # Start with small radius and expand until we have k markers
        search_radius = self.cell_size

        while search_radius <= max_radius:
            candidates = self.query_radius(position, search_radius)

            if len(candidates) >= k:
                # Compute distances and sort
                distances = [
                    (marker_id, self._distance(position, self.marker_positions[marker_id]))
                    for marker_id in candidates
                ]
                distances.sort(key=lambda x: x[1])
                return distances[:k]

            # Expand search radius
            search_radius *= 2

        # Didn't find enough markers, return what we have
        if search_radius > max_radius:
            candidates = self.query_radius(position, max_radius)
            distances = [
                (marker_id, self._distance(position, self.marker_positions[marker_id]))
                for marker_id in candidates
            ]
            distances.sort(key=lambda x: x[1])
            return distances[:k]

        return []

    def _distance(
        self,
        pos1: Tuple[float, ...],
        pos2: Tuple[float, ...]
    ) -> float:
        """Euclidean distance between positions"""
        return math.sqrt(self._distance_squared(pos1, pos2))

    def _distance_squared(
        self,
        pos1: Tuple[float, ...],
        pos2: Tuple[float, ...]
    ) -> float:
        """Squared Euclidean distance (avoids sqrt)"""
        return sum((a - b) ** 2 for a, b in zip(pos1, pos2))

    def get_statistics(self) -> Dict[str, any]:
        """
        Get index statistics.

        Returns:
            Dictionary with stats about the index
        """
        occupied_cells = len(self.grid)
        total_markers = len(self.marker_positions)

        markers_per_cell = (
            total_markers / occupied_cells if occupied_cells > 0 else 0
        )

        max_cell_occupancy = max(len(markers) for markers in self.grid.values()) if self.grid else 0

        return {
            'dimensions': self.dimensions,
            'cell_size': self.cell_size,
            'total_markers': total_markers,
            'occupied_cells': occupied_cells,
            'avg_markers_per_cell': round(markers_per_cell, 2),
            'max_cell_occupancy': max_cell_occupancy
        }

    def clear(self):
        """Remove all markers from index"""
        self.grid.clear()
        self.marker_positions.clear()

    def __len__(self) -> int:
        """Return number of markers in index"""
        return len(self.marker_positions)

    def __contains__(self, marker_id: str) -> bool:
        """Check if marker is in index"""
        return marker_id in self.marker_positions

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"SpatialIndex({self.dimensions}D, cell_size={self.cell_size}, "
                f"markers={stats['total_markers']}, cells={stats['occupied_cells']})")
