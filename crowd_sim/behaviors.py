# behaviors.py

import math
from typing import List, Dict
from .agents import Agent
from typing import Optional, Tuple


def displacement_with_periodic(agent, nb,
                               Lx: Optional[float],
                               Ly: Optional[float],
                               periodic_x: bool,
                               periodic_y: bool) -> Tuple[float, float]:
    """Compute displacement from agent to nb with minimum-image periodic boundaries."""
    dx = nb.x - agent.x
    dy = nb.y - agent.y

    if periodic_x and Lx is not None:
        if dx > 0.5 * Lx:
            dx -= Lx
        elif dx < -0.5 * Lx:
            dx += Lx

    if periodic_y and Ly is not None:
        if dy > 0.5 * Ly:
            dy -= Ly
        elif dy < -0.5 * Ly:
            dy += Ly

    return dx, dy


def is_neighbor_in_vision_cone(agent,
                               nb,
                               params: Dict,
                               dir_x: float,
                               dir_y: float) -> Tuple[bool, Optional[float]]:
    """
    Check if nb is within agent's vision cone.
    Returns (in_cone, distance). Distance is None if not in cone.
    """
    R = params.get("sensing_radius", 3.0)
    theta = params.get("sensing_half_angle", math.radians(60.0))

    Lx = params.get("Lx", None)
    Ly = params.get("Ly", None)
    periodic_x = params.get("periodic_x", False)
    periodic_y = params.get("periodic_y", False)

    # Displacement with periodic boundaries
    dx, dy = displacement_with_periodic(agent, nb, Lx, Ly, periodic_x, periodic_y)

    dist = math.hypot(dx, dy)
    if dist == 0.0 or dist > R:
        return False, None

    # Angle check: cos(angle) = (dir . r) / (|dir||r|)
    dot = dir_x * dx + dir_y * dy
    cos_angle = dot / dist

    # Behind or exactly sideways
    if cos_angle <= 0.0:
        return False, None

    # Outside half-angle
    if cos_angle < math.cos(theta):
        return False, None

    return True, dist

# these agents move forward at their desired speed until a neighbor is detected
# then they slow down at a linear rate depending on distance to closest blocking neighbor
# specifically, there are parameters d_stop (speed 0) and d_slow (last speed where speed is as desired)

def simple_unidirectional_behavior(agent,
                                   neighbors: List,
                                   params: Dict) -> None:
    R = params.get("sensing_radius", 3.0)  # still used? can be removed if only in helper
    d_stop = params.get("d_stop", 0.5)
    d_slow = params.get("d_slow", 1.5)
    v0 = agent.desired_speed

    dir_x = agent.dir_x
    dir_y = agent.dir_y

    # Normalize direction just in case
    norm = math.hypot(dir_x, dir_y)
    if norm == 0.0:
        dir_x, dir_y = 1.0, 0.0
        norm = 1.0
    dir_x /= norm
    dir_y /= norm

    nearest_d = None

    for nb in neighbors:
        in_cone, dist = is_neighbor_in_vision_cone(agent, nb, params, dir_x, dir_y)
        if not in_cone:
            continue

        if nearest_d is None or (dist is not None and dist < nearest_d):
            nearest_d = dist

    # Decide speed
    if nearest_d is None:
        speed = v0
    else:
        if nearest_d <= d_stop:
            speed = 0.0
        elif nearest_d >= d_slow:
            speed = v0
        else:
            t = (nearest_d - d_stop) / (d_slow - d_stop)
            speed = t * v0

    agent.vx = speed * dir_x
    agent.vy = speed * dir_y

# these agents move forward at their desired speed until a neighbor is detected
# then they slow their horizontal speed down at a linear rate but also begin moving to the side
def simple_passing_behavior(agent,
                                   neighbors: List,
                                   params: Dict) -> None:
    R = params.get("sensing_radius", 3.0)  # still used? can be removed if only in helper
    d_stop = params.get("d_stop", 0.5)
    d_slow = params.get("d_slow", 1.5)
    v0 = agent.desired_speed

    dir_x = agent.dir_x
    dir_y = agent.dir_y

    # Normalize direction just in case
    norm = math.hypot(dir_x, dir_y)
    if norm == 0.0:
        dir_x, dir_y = 1.0, 0.0
        norm = 1.0
    dir_x /= norm
    dir_y /= norm

    nearest_d = None

    for nb in neighbors:
        in_cone, dist = is_neighbor_in_vision_cone(agent, nb, params, dir_x, dir_y)
        if not in_cone:
            continue

        if nearest_d is None or (dist is not None and dist < nearest_d):
            nearest_d = dist

    # Decide speed
    if nearest_d is None:
        speed = v0
    else:
        if nearest_d <= d_stop:
            speed = 0.0

            y_speed = v0 / 10
        elif nearest_d >= d_slow:
            speed = v0

            y_speed = 0
        else:
            t = (nearest_d - d_stop) / (d_slow - d_stop)
            speed = t * v0

            y_speed = v0 / 10

    agent.vx = speed * dir_x
    agent.vy = y_speed # speed * dir_y