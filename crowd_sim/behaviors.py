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

def circular_ccw_direction(agent: Agent, params: Dict) -> Tuple[float, float]:
    """Return the unit tangent direction for counterclockwise motion."""
    cx = getattr(agent, "orbit_cx", params.get("circle_center_x", 0.0))
    cy = getattr(agent, "orbit_cy", params.get("circle_center_y", 0.0))

    rx = agent.x - cx
    ry = agent.y - cy
    r = math.hypot(rx, ry)
    if r == 0.0:
        return 1.0, 0.0

    return -ry / r, rx / r



def radial_restoring_velocity(agent: Agent, params: Dict) -> Tuple[float, float]:
    """Return a weak radial correction that keeps agents near the target orbit."""
    cx = getattr(agent, "orbit_cx", params.get("circle_center_x", 0.0))
    cy = getattr(agent, "orbit_cy", params.get("circle_center_y", 0.0))
    target_r = getattr(agent, "pass_target_radius", 0.0)
    if target_r == 0.0:
        target_r = getattr(agent, "orbit_radius", params.get("orbit_radius", 0.0))
    k_r = params.get("radial_gain", 0.0)

    rx = agent.x - cx
    ry = agent.y - cy
    r = math.hypot(rx, ry)
    if r == 0.0:
        return 0.0, 0.0

    err = target_r - r
    ux = rx / r
    uy = ry / r
    return k_r * err * ux, k_r * err * uy



def circular_orbit_behavior(agent: Agent,
                            neighbors: List,
                            params: Dict) -> None:
    d_stop = params.get("d_stop", 0.15)
    d_slow = params.get("d_slow", 0.4)

    dir_x, dir_y = circular_ccw_direction(agent, params)

    nearest_d = None
    for nb in neighbors:
        in_cone, dist = is_neighbor_in_vision_cone(agent, nb, params, dir_x, dir_y)
        if not in_cone:
            continue

        if nearest_d is None or (dist is not None and dist < nearest_d):
            nearest_d = dist

    omega = getattr(agent, "angular_speed", params.get("angular_speed", 0.0))
    radius = getattr(agent, "orbit_radius", params.get("orbit_radius", 0.0))
    v0 = abs(omega) * radius

    if nearest_d is None:
        speed = v0
    elif nearest_d <= d_stop:
        speed = 0.0
    elif nearest_d >= d_slow:
        speed = v0
    else:
        t = (nearest_d - d_stop) / (d_slow - d_stop)
        speed = t * v0

    corr_vx, corr_vy = radial_restoring_velocity(agent, params)

    agent.dir_x = dir_x
    agent.dir_y = dir_y
    agent.vx = speed * dir_x + corr_vx
    agent.vy = speed * dir_y + corr_vy



def circular_passing_behavior(agent: Agent,
                              neighbors: List,
                              params: Dict) -> None:
    d_stop = params.get("d_stop", 0.15)
    d_slow = params.get("d_slow", 0.4)
    max_offset = params.get("passing_radius_offset", 0.15)
    target_gain = params.get("passing_target_gain", 2.0)
    dt = params.get("dt", 0.05)

    dir_x, dir_y = circular_ccw_direction(agent, params)

    base_radius = getattr(agent, "orbit_radius", params.get("orbit_radius", 0.0))
    current_target = getattr(agent, "pass_target_radius", 0.0)
    if current_target == 0.0:
        current_target = base_radius

    nearest_d = None
    nearest_left_d = None

    for nb in neighbors:
        in_cone, dist = is_neighbor_in_vision_cone(agent, nb, params, dir_x, dir_y)
        if not in_cone:
            continue

        dx, dy = displacement_with_periodic(
            agent, nb,
            params.get("Lx", None),
            params.get("Ly", None),
            params.get("periodic_x", False),
            params.get("periodic_y", False),
        )

        side = dir_x * dy - dir_y * dx

        if nearest_d is None or (dist is not None and dist < nearest_d):
            nearest_d = dist

        if side > 0.0 and (nearest_left_d is None or (dist is not None and dist < nearest_left_d)):
            nearest_left_d = dist

    if nearest_left_d is None:
        pass_strength = 0.0
    elif nearest_left_d <= d_stop:
        pass_strength = 1.0
    elif nearest_left_d >= d_slow:
        pass_strength = 0.0
    else:
        pass_strength = 1.0 - (nearest_left_d - d_stop) / (d_slow - d_stop)

    desired_target_radius = base_radius + max_offset * pass_strength
    print(f"Agent {agent.id} has current desired target radius {desired_target_radius}")

    alpha = min(1.0, target_gain * dt)
    current_target = current_target + alpha * (desired_target_radius - current_target)
    agent.pass_target_radius = current_target

    print(f"Agent {agent.id} has current target radius {agent.pass_target_radius}")

    omega = getattr(agent, "angular_speed", params.get("angular_speed", 0.0))
    tangential_speed = abs(omega) * current_target

    if nearest_d is None:
        speed = tangential_speed
    elif nearest_d <= d_stop:
        speed = 0.0
    elif nearest_d >= d_slow:
        speed = tangential_speed
    else:
        t = (nearest_d - d_stop) / (d_slow - d_stop)
        speed = t * tangential_speed

    corr_vx, corr_vy = radial_restoring_velocity(agent, params)

    agent.dir_x = dir_x
    agent.dir_y = dir_y
    agent.vx = speed * dir_x + corr_vx
    agent.vy = speed * dir_y + corr_vy


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