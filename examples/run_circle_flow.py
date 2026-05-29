# examples/run_circle_flow.py
import math
import os
import random
import sys
from typing import Optional

# Allow running this script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame  # noqa: E402

from crowd_sim.behaviors import circular_robotics_behavior
from crowd_sim.circle_traffic_agent import CircleTrafficAgent
from crowd_sim.environment import Walkway, PeriodicBoundary2D, OpenBoundaryWithOverflow2D
from crowd_sim.simulation import Simulation
from crowd_sim.visualization.pygame_view import PygameViewer

WALKWAY_LENGTH = 2.0
CIRCLE_CENTER_X = WALKWAY_LENGTH / 2.0
CIRCLE_CENTER_Y = WALKWAY_LENGTH / 2.0
SENSING_HALF_ANGLE = math.radians(60.0)
AGENT_RADIUS = 0.06


def sample_angular_speed(
    target_speed_dist: str,
    target_speed_min: Optional[float],
    target_speed_max: Optional[float],
    target_speed_mean: Optional[float],
    target_speed_std: Optional[float],
) -> float:
    if target_speed_dist == "uniform":
        if target_speed_min is None or target_speed_max is None:
            raise ValueError("Uniform target speed requires min and max bounds")
        return random.uniform(target_speed_min, target_speed_max)

    if target_speed_dist == "normal":
        if target_speed_mean is None or target_speed_std is None:
            raise ValueError("Normal target speed requires mean and std")
        angular_speed = random.gauss(target_speed_mean, target_speed_std)
        if target_speed_min is not None:
            angular_speed = max(target_speed_min, angular_speed)
        if target_speed_max is not None:
            angular_speed = min(target_speed_max, angular_speed)
        return angular_speed

    raise ValueError(f"Unsupported target speed distribution: {target_speed_dist}")


def create_circle_agents(
    num_agents: int,
    center_x: float,
    center_y: float,
    radius: float,
    radius_min: float,
    radius_max: float,
    target_speed_dist: str,
    target_speed_min: Optional[float],
    target_speed_max: Optional[float],
    target_speed_mean: Optional[float],
    target_speed_std: Optional[float],
) -> list[CircleTrafficAgent]:
    agents = []
    for i in range(num_agents):
        theta = random.uniform(0.0, 2.0 * math.pi)
        u = random.uniform(0.0, 1.0)
        initial_radius = math.sqrt(u * (radius_max ** 2 - radius_min ** 2) + radius_min ** 2)
        x = center_x + initial_radius * math.cos(theta)
        y = center_y + initial_radius * math.sin(theta)

        angular_speed = sample_angular_speed(
            target_speed_dist=target_speed_dist,
            target_speed_min=target_speed_min,
            target_speed_max=target_speed_max,
            target_speed_mean=target_speed_mean,
            target_speed_std=target_speed_std,
        )

        heading = theta + math.pi / 2.0
        initial_tangential_speed = initial_radius * angular_speed

        agents.append(
            CircleTrafficAgent(
                id=i,
                x=x,
                y=y,
                vx=initial_tangential_speed * math.cos(heading),
                vy=initial_tangential_speed * math.sin(heading),
                desired_speed=initial_tangential_speed,
                heading=heading,
                orbit_cx=center_x,
                orbit_cy=center_y,
                orbit_radius=radius,
                base_orbit_radius=radius,
                target_radius=initial_radius,
                angular_speed=angular_speed,
            )
        )

    return agents


def main():
    walkway = Walkway(WALKWAY_LENGTH, WALKWAY_LENGTH)
    boundary = OpenBoundaryWithOverflow2D(walkway)

    dt = 0.025
    sim_time = 300.0
    sensing_radius = 0.4
    num_agents = 30
    circle_radius = 0.2
    circle_radius_min = 0.2
    circle_radius_max = 0.6
    d_stop = 0.15
    d_slow = 0.25
    lane_preference = "base"
    target_speed_dist = "uniform"
    target_speed_min = 0.3
    target_speed_max = 0.6
    target_speed_mean = math.pi / 10
    target_speed_std = 1.0
    gui_speedup = 2
    reaction_delay = 0

    behavior_params = {
        "sensing_radius": sensing_radius,
        "sensing_half_angle": SENSING_HALF_ANGLE,
        "reaction_delay": reaction_delay,
        "d_stop": d_stop,
        "d_slow": d_slow,
        "circle_center_x": CIRCLE_CENTER_X,
        "circle_center_y": CIRCLE_CENTER_Y,
        "orbit_radius": circle_radius,
        "radial_gain": 1.0,
        "side_sensing_radius": 0.18,
        "side_sensing_half_angle": math.pi / 3.0,
        "side_heading_offset": math.pi / 3.0,
        "circle_radius_min": circle_radius_min,
        "circle_radius_max": circle_radius_max,
        "lane_preference": lane_preference,
        "lane_return_delay": 2.0,
        "approach_rate_threshold": 0.0,
        "max_delta": 0.2,
        "sim_time": 0.0,
    }

    agents = create_circle_agents(
        num_agents=num_agents,
        center_x=CIRCLE_CENTER_X,
        center_y=CIRCLE_CENTER_Y,
        radius=circle_radius,
        radius_min=circle_radius_min,
        radius_max=circle_radius_max,
        target_speed_dist=target_speed_dist,
        target_speed_min=target_speed_min,
        target_speed_max=target_speed_max,
        target_speed_mean=target_speed_mean,
        target_speed_std=target_speed_std,
    )

    sim = Simulation(
        walkway=walkway,
        boundary_handler=boundary,
        agents=agents,
        dt=dt,
        behavior_fn=circular_robotics_behavior,
        behavior_params=behavior_params,
        sensing_radius=sensing_radius,
        periodic_x=True,
        periodic_y=True,
    )

    viewer = PygameViewer(
        walkway=walkway,
        sensing_radius=sensing_radius,
        pixels_per_meter=220.0,
        fps_cap=30,
        sensing_half_angle=SENSING_HALF_ANGLE,
        side_sensing_radius=behavior_params["side_sensing_radius"],
        side_sensing_half_angle=behavior_params["side_sensing_half_angle"],
        side_heading_offset=behavior_params["side_heading_offset"],
        orbit_center=(CIRCLE_CENTER_X, CIRCLE_CENTER_Y),
        orbit_radius=circle_radius,
        agent_radius_m=AGENT_RADIUS,
    )

    running = True
    max_steps = int(sim_time / dt)
    step_count = 0

    while running and step_count < max_steps:
        running = viewer.handle_events()

        if viewer.request_reset:
            agents = create_circle_agents(
                num_agents=num_agents,
                center_x=CIRCLE_CENTER_X,
                center_y=CIRCLE_CENTER_Y,
                radius=circle_radius,
                radius_min=circle_radius_min,
                radius_max=circle_radius_max,
                target_speed_dist=target_speed_dist,
                target_speed_min=target_speed_min,
                target_speed_max=target_speed_max,
                target_speed_mean=target_speed_mean,
                target_speed_std=target_speed_std,
            )
            behavior_params["sim_time"] = 0.0
            sim = Simulation(
                walkway=walkway,
                boundary_handler=boundary,
                agents=agents,
                dt=dt,
                behavior_fn=circular_robotics_behavior,
                behavior_params=behavior_params,
                sensing_radius=sensing_radius,
                periodic_x=True,
                periodic_y=True,
            )
            step_count = 0
            viewer.paused = False
            viewer.single_step = False
            viewer.request_reset = False

        if not viewer.paused or viewer.single_step:
            steps_this_frame = 1 if viewer.single_step else gui_speedup
            for _ in range(steps_this_frame):
                if step_count >= max_steps:
                    break
                behavior_params["sim_time"] = sim.time
                sim.step()
                step_count += 1
            viewer.single_step = False

        viewer.draw(sim.agents, sim.time)

    pygame.quit()


if __name__ == "__main__":
    main()
