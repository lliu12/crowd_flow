# agents.py
from collections import deque
from dataclasses import dataclass, field
import math
from typing import Deque, Tuple, Optional



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
    heading: float  # radians CCW from +x
    current_tangential_speed: float = 0.0
    orbit_cx: float = 0.0
    orbit_cy: float = 0.0
    orbit_radius: float = 0.0
    target_radius: float = 0.0
    angular_speed: float = 0.0
    blocked: bool = False
    left_blocked: bool = False
    right_blocked: bool = False
    dist_to_nearest: Optional[float] = None
    angle_to_nearest: Optional[float] = None
    trail: Deque[Tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=50)
    )

    def position(self) -> Tuple[float, float]:
        return self.x, self.y

    def velocity(self) -> Tuple[float, float]:
        return self.vx, self.vy

    def direction(self) -> Tuple[float, float]:
        return math.cos(self.heading), math.sin(self.heading)

    def speed(self) -> float:
        return (self.vx ** 2 + self.vy ** 2) ** 0.5
