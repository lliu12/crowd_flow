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


def is_neighbor_in_custom_cone(agent,
                               nb,
                               params: Dict,
                               dir_x: float,
                               dir_y: float,
                               range_m: float,
                               half_angle: float) -> Tuple[bool, Optional[float], float, float]:
    """Check if nb is within a cone with explicit direction, range, and half-angle."""
    Lx = params.get("Lx", None)
    Ly = params.get("Ly", None)
    periodic_x = params.get("periodic_x", False)
    periodic_y = params.get("periodic_y", False)

    dx, dy = displacement_with_periodic(agent, nb, Lx, Ly, periodic_x, periodic_y)

    dist = math.hypot(dx, dy)
    if dist == 0.0 or dist > range_m:
        return False, None, dx, dy

    dot = dir_x * dx + dir_y * dy
    cos_angle = dot / dist

    if cos_angle <= 0.0:
        return False, None, dx, dy

    if cos_angle < math.cos(half_angle):
        return False, None, dx, dy

    return True, dist, dx, dy



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
    in_cone, dist, _, _ = is_neighbor_in_custom_cone(agent, nb, params, dir_x, dir_y, R, theta)
    return in_cone, dist

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
        in_cone, dist, dx, dy = is_neighbor_in_custom_cone(
            agent,
            nb,
            params,
            dir_x,
            dir_y,
            params.get("sensing_radius", 3.0),
            params.get("sensing_half_angle", math.radians(60.0)),
        )
        if not in_cone:
            continue

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
    alpha = min(1.0, target_gain * dt)
    current_target = current_target + alpha * (desired_target_radius - current_target)
    agent.pass_target_radius = current_target

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


def circular_robotics_behavior(agent: Agent,
                               neighbors: List,
                               params: Dict) -> None:
    
    
    d_stop = params.get("d_stop", 0.15)
    d_slow = params.get("d_slow", 0.4)
    dt = params.get("dt", 0.05)
    radius_target_gain = params.get("radius_target_gain", 4.0)
    forward_range = params.get("sensing_radius", 3.0)
    forward_half_angle = params.get("sensing_half_angle", math.radians(60.0))
    side_range = params.get("side_sensing_radius", forward_range)
    side_half_angle = params.get("side_sensing_half_angle", math.radians(45.0))


    base_radius = getattr(agent, "orbit_radius", params.get("orbit_radius", 0.0))
    current_target = getattr(agent, "pass_target_radius", 0.0)
    if current_target == 0.0:
        current_target = base_radius

    r_min = params.get("circle_radius_min", base_radius)
    r_max = params.get("circle_radius_max", base_radius)

    dir_x, dir_y = circular_ccw_direction(agent, params)
    left_dir_x, left_dir_y = -dir_y, dir_x
    right_dir_x, right_dir_y = dir_y, -dir_x

    nearest_d = None
    nearest_angle = None
    nearest_left_d = None
    nearest_right_d = None

    for nb in neighbors:
        in_forward, forward_dist, dx, dy = is_neighbor_in_custom_cone(
            agent, nb, params, dir_x, dir_y, forward_range, forward_half_angle
        )
        if in_forward:
            signed_angle = math.atan2(dir_x * dy - dir_y * dx, dir_x * dx + dir_y * dy)
            if nearest_d is None or (forward_dist is not None and forward_dist < nearest_d):
                nearest_d = forward_dist
                nearest_angle = signed_angle

        in_left, left_dist, _, _ = is_neighbor_in_custom_cone(
            agent, nb, params, left_dir_x, left_dir_y, side_range, side_half_angle
        )
        if in_left and (nearest_left_d is None or (left_dist is not None and left_dist < nearest_left_d)):
            nearest_left_d = left_dist

        in_right, right_dist, _, _ = is_neighbor_in_custom_cone(
            agent, nb, params, right_dir_x, right_dir_y, side_range, side_half_angle
        )
        if in_right and (nearest_right_d is None or (right_dist is not None and right_dist < nearest_right_d)):
            nearest_right_d = right_dist

    agent.blocked = nearest_d is not None
    agent.left_blocked = nearest_left_d is not None
    agent.right_blocked = nearest_right_d is not None
    agent.dist_to_nearest = nearest_d
    agent.angle_to_nearest = nearest_angle

    omega = getattr(agent, "angular_speed", params.get("angular_speed", 0.0))
    base_tangential_speed = abs(omega) * current_target

    if nearest_d is None:
        closeness = 0.0
        tangential_speed = base_tangential_speed
    elif nearest_d <= d_stop:
        closeness = 1.0
        tangential_speed = 0.0
    elif nearest_d >= d_slow:
        closeness = 0.0
        tangential_speed = base_tangential_speed
    else:
        t = (nearest_d - d_stop) / (d_slow - d_stop)
        closeness = 1.0 - t
        tangential_speed = t * base_tangential_speed

    if not agent.blocked and not agent.left_blocked and not agent.right_blocked:
        desired_radius = base_radius
    else:
        # if nearest_angle is None or (agent.left_blocked and agent.right_blocked):
        # ^ this line was the bug! front cone empty -> radius reset to base
        if agent.left_blocked and agent.right_blocked:
            side_sign = 0.0
            closeness = 1.0
        elif (not agent.left_blocked) and ((False if nearest_angle is None else nearest_angle < 0.0) or agent.right_blocked):
            side_sign = -1.0
            closeness = 1.0
        elif (not agent.right_blocked) and ((False if nearest_angle is None else nearest_angle > 0.0) or agent.left_blocked):
            side_sign = 1.0
            closeness = 1.0
        else:
            side_sign = 0.0


        max_delta_inward = max(0.0, base_radius - r_min)
        max_delta_outward = max(0.0, r_max - base_radius)
        max_delta = max_delta_inward if side_sign < 0.0 else max_delta_outward
        desired_radius = base_radius + side_sign * closeness * max_delta
        # desired_radius = base_radius + side_sign * max_delta

        desired_radius = max(r_min, min(r_max, desired_radius))
        print(f"nearest angle: {nearest_angle}, side sign: {side_sign}, desired radius {desired_radius}")

        # print(f"Agent {agent.id} desired radius: {desired_radius}, sign {side_sign}, closeness {closeness}, max_delta {max_delta}")


    alpha = min(1.0, radius_target_gain * dt)
    current_target = current_target + alpha * (desired_radius - current_target)
    agent.pass_target_radius = current_target

    tangential_speed = min(tangential_speed, abs(omega) * current_target) if tangential_speed > 0.0 else tangential_speed
    corr_vx, corr_vy = radial_restoring_velocity(agent, params)

    agent.dir_x = dir_x
    agent.dir_y = dir_y
    agent.vx = tangential_speed * dir_x + corr_vx
    agent.vy = tangential_speed * dir_y + corr_vy


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