# simulation.py
from typing import List, Callable, Dict
from .agents import Agent
from .environment import Walkway, BoundaryHandler
from .neighbors import NeighborGrid
from .behaviors import simple_unidirectional_behavior, simple_passing_behavior, circular_orbit_behavior


class Simulation:
    """
    Main simulation class:
    - Holds agents, environment, neighbor grid, behavior.
    - Advances the system in discrete time steps.
    """
    def __init__(self,
                 walkway: Walkway,
                 boundary_handler: BoundaryHandler,
                 agents: List[Agent],
                 dt: float = 0.05,
                 behavior_fn: Callable = simple_unidirectional_behavior,
                 behavior_params: Dict = None,
                 sensing_radius: float = 3.0,
                 periodic_x: bool = True,
                 periodic_y: bool = True):
        self.walkway = walkway
        self.boundary_handler = boundary_handler
        self.agents = agents
        self.dt = dt
        self.behavior_fn = behavior_fn

        # Neighbor grid with cell size ~ sensing radius
        self.neighbor_grid = NeighborGrid(
            walkway=self.walkway,
            cell_size=sensing_radius,
            periodic_x=periodic_x,
            periodic_y=periodic_y,
        )

        # Default behavior parameters
        if behavior_params is None:
            behavior_params = {
                "sensing_radius": sensing_radius,
                "sensing_half_angle": 60.0 * 3.14159265 / 180.0,
                "d_stop": 0.5,
                "d_slow": 1.5,
            }

        # Add domain & periodic info so behavior can do minimum-image distances
        behavior_params.setdefault("Lx", self.walkway.Lx)
        behavior_params.setdefault("Ly", self.walkway.Ly)
        behavior_params.setdefault("periodic_x", periodic_x)
        behavior_params.setdefault("periodic_y", periodic_y)

        self.behavior_params = behavior_params
        self.time = 0.0

    def step(self) -> None:
        """
        Advance the simulation by one time step dt.
        """
        if not self.agents:
            return

        # 1. Build neighbor grid (based on current positions)
        self.neighbor_grid.build(self.agents)

        # 2. Compute new velocities based on neighbors
        # We compute all velocities before updating positions for synchronicity.
        for i, agent in enumerate(self.agents):
            candidate_indices = self.neighbor_grid.get_candidate_neighbors(self.agents, i)
            # Convert indices to Agent objects and exclude self
            neighbors = [self.agents[j] for j in candidate_indices if j != i]
            self.behavior_fn(agent, neighbors, self.behavior_params)

        # 3. Update positions with Euler step
        dt = self.dt
        for agent in self.agents:
            agent.trail.append((agent.x, agent.y))

            agent.x += agent.vx * dt
            agent.y += agent.vy * dt

        # 4. Apply boundary conditions (may remove agents or wrap them)
        self.boundary_handler.apply(self.agents)

        # 5. Update time
        self.time += self.dt