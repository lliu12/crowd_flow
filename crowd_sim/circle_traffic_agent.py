from dataclasses import dataclass
from typing import Optional

from .agents import Agent


@dataclass
class CircleTrafficAgent(Agent):
    base_orbit_radius: float = 0.0
    last_min_dist: Optional[float] = None
    last_blocked_left_time: float = 0.0
    last_blocked_right_time: float = 0.0
    hold_radius: Optional[float] = None
    left_dist_to_nearest: Optional[float] = None
    left_angle_to_nearest: Optional[float] = None
    right_dist_to_nearest: Optional[float] = None
    right_angle_to_nearest: Optional[float] = None
