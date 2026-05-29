# behaviors.py

import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from .agents import Agent
from .circle_traffic_agent import CircleTrafficAgent


def heading_to_direction(heading: float) -> Tuple[float, float]:
    return math.cos(heading), math.sin(heading)

def direction_to_heading(vx: float, vy: float) -> float:
    return math.atan2(vy, vx)


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


def is_neighbor_in_cone(agent,
                        nb,
                        params: Dict,
                        heading: float,
                        range_m: float,
                        half_angle: float) -> Tuple[bool, Optional[float], float, float]:
    """
    Check whether nb lies within a symmetric sensing cone around heading.

    Inputs:
      - agent, nb: objects with x,y positions.
      - params: may contain Lx, Ly, periodic_x, periodic_y for periodic boundaries.
      - heading: agent heading in radians.
      - range_m: maximum sensing distance.
      - half_angle: half of the cone angle (radians) about heading.

    Returns:
      (in_cone, distance, dx, dy) where:
        - in_cone: True if nb is inside the cone and within range, else False.
        - distance: euclidean distance to nb (None if not in cone or at zero distance).
        - dx, dy: displacement from agent to nb after applying periodic wrapping.
    """
    Lx = params.get("Lx", None)
    Ly = params.get("Ly", None)
    periodic_x = params.get("periodic_x", False)
    periodic_y = params.get("periodic_y", False)

    dx, dy = displacement_with_periodic(agent, nb, Lx, Ly, periodic_x, periodic_y)

    dist = math.hypot(dx, dy)
    if dist == 0.0 or dist > range_m:
        return False, None, dx, dy

    dir_x, dir_y = heading_to_direction(heading)
    dot = dir_x * dx + dir_y * dy
    cos_angle = dot / dist

    if cos_angle <= 0.0:
        return False, None, dx, dy

    if cos_angle < math.cos(half_angle):
        return False, None, dx, dy

    return True, dist, dx, dy


def circular_ccw_heading(agent: Agent, params: Dict) -> float:
    """Return the tangent heading for counterclockwise motion."""
    cx = agent.orbit_cx
    cy = agent.orbit_cy
    return math.atan2(agent.y - cy, agent.x - cx) + math.pi / 2.0


def radial_restoring_velocity(agent: Agent, params: Dict) -> Tuple[float, float]:
    """Return a weak radial correction that keeps agents near the target orbit."""
    cx = agent.orbit_cx
    cy = agent.orbit_cy
    target_r = agent.target_radius
    if target_r == 0.0:
        target_r = getattr(agent, "orbit_radius", params.get("orbit_radius", 0.0))
    k_r = params.get("radial_gain")

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
    d_stop = params.get("d_stop")
    d_slow = params.get("d_slow")

    # heading = agent.heading
    heading = circular_ccw_heading(agent, params)
    dir_x, dir_y = heading_to_direction(heading)

    vision_range = params.get("sensing_radius")
    vision_half_angle = params.get("sensing_half_angle")

    nearest_d = None
    for nb in neighbors:
        in_cone, dist, _, _ = is_neighbor_in_cone(
            agent,
            nb,
            params,
            heading,
            vision_range,
            vision_half_angle,
        )
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

    agent.heading = heading
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

    heading = circular_ccw_heading(agent, params)
    # heading = agent.heading
    dir_x, dir_y = heading_to_direction(heading)

    base_radius = getattr(agent, "orbit_radius", params.get("orbit_radius", 0.0))
    current_target = getattr(agent, "target_radius", 0.0)
    if current_target == 0.0:
        current_target = base_radius

    nearest_d = None
    nearest_left_d = None

    sensing_range = params.get("sensing_radius", 3.0)
    sensing_half_angle = params.get("sensing_half_angle", math.radians(60.0))

    for nb in neighbors:
        in_cone, dist, dx, dy = is_neighbor_in_cone(
            agent,
            nb,
            params,
            heading,
            sensing_range,
            sensing_half_angle,
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
    agent.target_radius = current_target

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

    agent.vx = speed * dir_x + corr_vx
    agent.vy = speed * dir_y + corr_vy
    # agent.heading = direction_to_heading(agent.vx, agent.vy)



# Written to match implementation on the robots
def circular_robotics_behavior(agent: Agent,
                               neighbors: List,
                               params: Dict) -> None:
    d_stop = params.get("d_stop", 0.15)
    d_slow = params.get("d_slow", 0.4)
    dt = params.get("dt", 0.05)
    sim_time = params.get("sim_time", 0.0)
    forward_range = params.get("sensing_radius", 3.0)
    forward_half_angle = params.get("sensing_half_angle", math.radians(60.0))
    side_range = params.get("side_sensing_radius", 0)
    side_half_angle = params.get("side_sensing_half_angle", math.radians(45.0))
    side_heading_offset = params.get("side_heading_offset", math.pi / 3.0)
    lane_preference = params.get("lane_preference", "base")
    lane_return_delay = params.get("lane_return_delay", 2.0)
    approach_rate_threshold = params.get("approach_rate_threshold", 0.0)
    max_delta = params.get("max_delta", 0.2)
    reaction_delay = params.get("reaction_delay", 0.0)
    reaction_delay_steps = params.get("reaction_delay_steps")
    if reaction_delay_steps is None:
        reaction_delay_steps = max(0, int(round(reaction_delay / dt)))

    base_radius = getattr(agent, "base_orbit_radius", getattr(agent, "orbit_radius", params.get("orbit_radius", 0.0)))
    current_target = getattr(agent, "target_radius", 0.0)
    if current_target == 0.0:
        current_target = base_radius

    r_min = params.get("circle_radius_min", base_radius)
    r_max = params.get("circle_radius_max", base_radius)

    heading = circular_ccw_heading(agent, params)
    dir_x, dir_y = heading_to_direction(heading)
    left_heading = heading + side_heading_offset
    right_heading = heading - side_heading_offset

    nearest_d = None
    nearest_angle = None
    nearest_left_d = None
    nearest_left_angle = None
    nearest_right_d = None
    nearest_right_angle = None

    for nb in neighbors:
        in_forward, forward_dist, dx, dy = is_neighbor_in_cone(
            agent, nb, params, heading, forward_range, forward_half_angle
        )
        if in_forward:
            signed_angle = math.atan2(dir_x * dy - dir_y * dx, dir_x * dx + dir_y * dy)
            if nearest_d is None or (forward_dist is not None and forward_dist < nearest_d):
                nearest_d = forward_dist
                nearest_angle = signed_angle

        in_left, left_dist, left_dx, left_dy = is_neighbor_in_cone(
            agent, nb, params, left_heading, side_range, side_half_angle
        )
        if in_left:
            left_signed_angle = math.atan2(dir_x * left_dy - dir_y * left_dx, dir_x * left_dx + dir_y * left_dy)
            if nearest_left_d is None or (left_dist is not None and left_dist < nearest_left_d):
                nearest_left_d = left_dist
                nearest_left_angle = left_signed_angle

        in_right, right_dist, right_dx, right_dy = is_neighbor_in_cone(
            agent, nb, params, right_heading, side_range, side_half_angle
        )
        if in_right:
            right_signed_angle = math.atan2(dir_x * right_dy - dir_y * right_dx, dir_x * right_dx + dir_y * right_dy)
            if nearest_right_d is None or (right_dist is not None and right_dist < nearest_right_d):
                nearest_right_d = right_dist
                nearest_right_angle = right_signed_angle

    agent.blocked = nearest_d is not None
    agent.left_blocked = nearest_left_d is not None
    agent.right_blocked = nearest_right_d is not None
    agent.dist_to_nearest = nearest_d
    agent.angle_to_nearest = nearest_angle

    left_side_active = agent.left_blocked or (nearest_angle is not None and nearest_angle > 0.0)
    right_side_active = agent.right_blocked or (nearest_angle is not None and nearest_angle < 0.0)

    if isinstance(agent, CircleTrafficAgent):
        agent.left_dist_to_nearest = nearest_left_d
        agent.left_angle_to_nearest = nearest_left_angle
        agent.right_dist_to_nearest = nearest_right_d
        agent.right_angle_to_nearest = nearest_right_angle
        if left_side_active:
            agent.last_blocked_left_time = sim_time
        if right_side_active:
            agent.last_blocked_right_time = sim_time

    cx = getattr(agent, "orbit_cx", params.get("circle_center_x", 0.0))
    cy = getattr(agent, "orbit_cy", params.get("circle_center_y", 0.0))
    polar_angle = math.atan2(agent.y - cy, agent.x - cx)
    current_radius = math.hypot(agent.x - cx, agent.y - cy)
    current_radius = current_radius if current_radius > 0.0 else base_radius

    if isinstance(agent, CircleTrafficAgent):
        if agent.last_polar_angle is None:
            agent.last_polar_angle = polar_angle
        else:
            delta_angle = polar_angle - agent.last_polar_angle
            while delta_angle <= -math.pi:
                delta_angle += 2.0 * math.pi
            while delta_angle > math.pi:
                delta_angle -= 2.0 * math.pi
            agent.lap_count_ccw += delta_angle / (2.0 * math.pi)
            agent.last_polar_angle = polar_angle

    omega = getattr(agent, "angular_speed", params.get("angular_speed", 0.0))
    target_tangential_speed = abs(omega) * current_radius
    current_speed = agent.speed()

    if nearest_d is None:
        tangential_speed = target_tangential_speed
    elif nearest_d <= d_stop:
        tangential_speed = 0.0
    elif nearest_d >= d_slow:
        tangential_speed = target_tangential_speed
    else:
        t = (nearest_d - d_stop) / (d_slow - d_stop)
        tangential_speed = t * target_tangential_speed

    desired_radius = base_radius

    if isinstance(agent, CircleTrafficAgent):
        if lane_preference == "base":
            relevant_last_blocked = agent.last_blocked_left_time
            drift_back_allowed = (sim_time - relevant_last_blocked) > lane_return_delay
            if agent.hold_radius is None:
                agent.hold_radius = current_radius
            if drift_back_allowed:
                desired_radius = np.clip(base_radius,
                                         current_radius - max_delta,
                                         current_radius + max_delta)
            else:
                desired_radius = agent.hold_radius
        elif lane_preference == "current":
            desired_radius = current_radius
        else:
            print("invalid lane preference argument")
            desired_radius = current_radius
    desired_radius = np.clip(desired_radius, r_min, r_max)

    approach_rate = None
    pass_allowed = True
    if isinstance(agent, CircleTrafficAgent) and nearest_d is not None and agent.last_min_dist is not None and dt > 0.0:
        approach_rate = (agent.last_min_dist - nearest_d) / dt
        if approach_rate < approach_rate_threshold and current_speed > 0.95 * target_tangential_speed:
            pass_allowed = False

    if agent.blocked or agent.left_blocked or agent.right_blocked:
        if agent.left_blocked and agent.right_blocked:
            side_sign = 0.0
        elif (not agent.left_blocked) and pass_allowed and right_side_active:
            side_sign = -1.0
            agent.hold_radius = None
        elif (not agent.right_blocked) and pass_allowed and left_side_active:
            side_sign = 1.0
            agent.hold_radius = None
        else:
            side_sign = 0.0

        if pass_allowed:
            if side_sign < 0.0:
                desired_radius = max(r_min, current_radius - max_delta)
            elif side_sign > 0.0:
                desired_radius = min(r_max, current_radius + max_delta)

    tangential_speed = min(tangential_speed, abs(omega) * current_radius) if tangential_speed > 0.0 else tangential_speed
    corr_vx, corr_vy = radial_restoring_velocity(agent, params)

    command = {
        "heading": heading,
        "vx": tangential_speed * dir_x + corr_vx,
        "vy": tangential_speed * dir_y + corr_vy,
        "target_radius": desired_radius,
        "tangential_speed_command": tangential_speed,
    }

    executed_command = command
    executed_command_age_steps = 0
    if isinstance(agent, CircleTrafficAgent):
        agent.command_buffer.append(command)
        if reaction_delay_steps <= 0:
            executed_command = command
            agent.command_buffer.clear()
        elif len(agent.command_buffer) <= reaction_delay_steps:
            executed_command = agent.command_buffer[0]
            executed_command_age_steps = len(agent.command_buffer) - 1
        else:
            executed_command = agent.command_buffer.popleft()
            executed_command_age_steps = reaction_delay_steps

    agent.target_radius = executed_command["target_radius"]
    agent.heading = executed_command["heading"]
    agent.vx = executed_command["vx"]
    agent.vy = executed_command["vy"]

    if isinstance(agent, CircleTrafficAgent):
        agent.current_radius = current_radius
        agent.target_tangential_speed = target_tangential_speed
        agent.current_speed = current_speed
        agent.approach_rate = approach_rate
        agent.pass_allowed = pass_allowed
        agent.tangential_speed_command = tangential_speed
        agent.radial_correction_vx = corr_vx
        agent.radial_correction_vy = corr_vy
        agent.last_min_dist = nearest_d
        agent.executed_tangential_speed_command = executed_command["tangential_speed_command"]
        agent.executed_target_radius = executed_command["target_radius"]
        agent.executed_command_age_steps = executed_command_age_steps


# these agents move forward at their desired speed until a neighbor is detected
# then they slow down at a linear rate depending on distance to closest blocking neighbor
# specifically, there are parameters d_stop (speed 0) and d_slow (last speed where speed is as desired)

def simple_unidirectional_behavior(agent,
                                   neighbors: List,
                                   params: Dict) -> None:
    d_stop = params.get("d_stop", 0.5)
    d_slow = params.get("d_slow", 1.5)
    v0 = agent.desired_speed

    dir_x, dir_y = heading_to_direction(agent.heading)

    vision_range = params.get("sensing_radius", 3.0)
    vision_half_angle = params.get("sensing_half_angle", math.radians(60.0))

    nearest_d = None

    for nb in neighbors:
        in_cone, dist, _, _ = is_neighbor_in_cone(
            agent,
            nb,
            params,
            agent.heading,
            vision_range,
            vision_half_angle,
        )
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

    agent.blocked = nearest_d is not None
    agent.vx = speed * dir_x
    agent.vy = speed * dir_y


# these agents move forward at their desired speed until a neighbor is detected
# then they slow their horizontal speed down at a linear rate but also begin moving to the side
def simple_passing_behavior(agent,
                            neighbors: List,
                            params: Dict) -> None:
    d_stop = params.get("d_stop", 0.5)
    d_slow = params.get("d_slow", 1.5)
    v0 = agent.desired_speed

    dir_x, dir_y = heading_to_direction(agent.heading)

    vision_range = params.get("sensing_radius", 3.0)
    vision_half_angle = params.get("sensing_half_angle", math.radians(60.0))

    nearest_d = None

    for nb in neighbors:
        in_cone, dist, _, _ = is_neighbor_in_cone(
            agent,
            nb,
            params,
            agent.heading,
            vision_range,
            vision_half_angle,
        )
        if not in_cone:
            continue

        if nearest_d is None or (dist is not None and dist < nearest_d):
            nearest_d = dist

    # Decide speed
    if nearest_d is None:
        speed = v0
        y_speed = 0.0
    else:
        if nearest_d <= d_stop:
            speed = 0.0
            y_speed = v0 / 10
        elif nearest_d >= d_slow:
            speed = v0
            y_speed = 0.0
        else:
            t = (nearest_d - d_stop) / (d_slow - d_stop)
            speed = t * v0
            y_speed = v0 / 10

    agent.blocked = nearest_d is not None
    agent.vx = speed * dir_x
    agent.vy = y_speed  # speed * dir_y