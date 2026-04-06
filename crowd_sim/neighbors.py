# neighbors.py
from typing import List, Tuple
from .agents import Agent
from .environment import Walkway


class NeighborGrid:
    """
    Uniform grid (cell list) for neighbor search in 2D.

    - Each cell stores indices of agents.
    - For each agent, we only search its cell and neighbors cells.
    """
    def __init__(self,
                 walkway: Walkway,
                 cell_size: float,
                 periodic_x: bool = True,
                 periodic_y: bool = True):
        self.walkway = walkway
        self.cell_size = cell_size
        self.periodic_x = periodic_x
        self.periodic_y = periodic_y

        self.nx = max(1, int(self.walkway.Lx / self.cell_size))
        self.ny = max(1, int(self.walkway.Ly / self.cell_size))

        # grid[cx][cy] = list of agent indices
        self.grid: List[List[List[int]]] = [
            [[] for _ in range(self.ny)] for _ in range(self.nx)
        ]

    def _cell_indices(self, x: float, y: float) -> Tuple[int, int]:
        cx = int(x / self.cell_size)
        cy = int(y / self.cell_size)
        # clamp to valid range (for non-periodic)
        if cx < 0:
            cx = 0
        elif cx >= self.nx:
            cx = self.nx - 1
        if cy < 0:
            cy = 0
        elif cy >= self.ny:
            cy = self.ny - 1
        return cx, cy

    def build(self, agents: List[Agent]) -> None:
        # clear grid
        for i in range(self.nx):
            for j in range(self.ny):
                self.grid[i][j].clear()

        # insert agents
        for idx, a in enumerate(agents):
            cx, cy = self._cell_indices(a.x, a.y)
            self.grid[cx][cy].append(idx)

    def get_candidate_neighbors(self, agents: List[Agent], idx: int) -> List[int]:
        """
        Return indices of agents in the 3x3 neighborhood of the agent's cell,
        including its own cell.
        """
        a = agents[idx]
        cx, cy = self._cell_indices(a.x, a.y)
        candidates: List[int] = []

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                ncx = cx + dx
                ncy = cy + dy

                # handle periodic in x
                if self.periodic_x:
                    if ncx < 0:
                        ncx += self.nx
                    elif ncx >= self.nx:
                        ncx -= self.nx
                # handle periodic in y
                if self.periodic_y:
                    if ncy < 0:
                        ncy += self.ny
                    elif ncy >= self.ny:
                        ncy -= self.ny

                # for non-periodic, skip out-of-range neighbors
                if not self.periodic_x and (ncx < 0 or ncx >= self.nx):
                    continue
                if not self.periodic_y and (ncy < 0 or ncy >= self.ny):
                    continue

                cell_list = self.grid[ncx][ncy]
                candidates.extend(cell_list)

        # caller will typically exclude idx itself
        return candidates