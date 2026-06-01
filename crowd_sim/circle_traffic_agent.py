from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

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
    current_radius: float = 0.0
    target_tangential_speed: float = 0.0
    current_speed: float = 0.0
    approach_rate: Optional[float] = None
    pass_allowed: bool = True
    lap_count_ccw: float = 0.0
    last_polar_angle: Optional[float] = None
    tangential_speed_command: float = 0.0
    radial_correction_vx: float = 0.0
    radial_correction_vy: float = 0.0
    command_buffer: Deque[dict] = field(default_factory=deque)
    executed_tangential_speed_command: float = 0.0
    executed_target_radius: float = 0.0
    executed_command_age_steps: int = 0
    executed_tangential_speed_before_accel_limit: float = 0.0
    realized_tangential_speed: float = 0.0
    applied_tangential_acceleration: float = 0.0
    tangential_acceleration_limited: bool = False
