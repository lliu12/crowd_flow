# agents.py
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Tuple



@dataclass
class Agent:
    """
    Represents a single pedestrian agent in the simulation.
    Positions are in meters, velocities in m/s.
    """
    id: int
    x: float
    y: float
    vx: float
    vy: float
    desired_speed: float
    dir_x: float  # direction unit vector x-component
    dir_y: float  # direction unit vector y-component
    trail: Deque[Tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=50)
    )

    def position(self) -> Tuple[float, float]:
        return self.x, self.y

    def velocity(self) -> Tuple[float, float]:
        return self.vx, self.vy

    def speed(self) -> float:
        return (self.vx ** 2 + self.vy ** 2) ** 0.5