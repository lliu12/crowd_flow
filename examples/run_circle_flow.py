# examples/run_circle_flow.py
import math
import os
import random
import sys
from typing import Optional
import numpy as np

# Allow running this script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame  # noqa: E402

from crowd_sim.agents import Agent
from crowd_sim.behaviors import circular_orbit_behavior, circular_passing_behavior, circular_robotics_behavior
from crowd_sim.environment import Walkway, PeriodicBoundary2D
from crowd_sim.simulation import Simulation
from crowd_sim.visualization.pygame_view import PygameViewer

WALKWAY_LENGTH = 4.0
CIRCLE_CENTER_X = WALKWAY_LENGTH / 2.0
CIRCLE_CENTER_Y = WALKWAY_LENGTH / 2.0
ORBIT_RADIUS = 0.8
ANGULAR_SPEED = math.pi / 5
ANGULAR_SPEED_STD = 1
SENSING_HALF_ANGLE = math.radians(60.0)
AGENT_RADIUS = 0.06 # for drawing agent body
CIRCLE_RADIUS_MAX = 1.4
CIRCLE_RADIUS_MIN = ORBIT_RADIUS



def create_circle_agents(
    num_agents: int,
    center_x: float,
    center_y: float,
    radius: float,
    # angular_speed: float,
    angular_speed_mean: float,
    angular_speed_std: Optional[float] = None,
) -> list[Agent]:
    agents = []
    for i in range(num_agents):
        theta = 2.0 * math.pi * i / num_agents
        x = center_x + radius * math.cos(theta)
        y = center_y + radius * math.sin(theta)

        # # speed: fixed or normally distributed
        # if angular_speed_std is None or angular_speed_std == 0.0:
        #     angular_speed = angular_speed_mean
        # else:
        #     angular_speed = random.gauss(angular_speed_mean, angular_speed_std)
        #     angular_speed =  np.clip(angular_speed, a_min = 0.3, a_max = None)

        angular_speed = random.uniform(0.3, 0.6)


        dir_x = -math.sin(theta)
        dir_y = math.cos(theta)
        speed = radius * angular_speed

        agents.append(
            Agent(
                id=i,
                x=x,
                y=y,
                vx=speed * dir_x,
                vy=speed * dir_y,
                desired_speed=speed,
                dir_x=dir_x,
                dir_y=dir_y,
                orbit_cx=center_x,
                orbit_cy=center_y,
                orbit_radius=radius,
                pass_target_radius=radius,
                angular_speed=angular_speed,
            )
        )

    return agents


def main():
    walkway = Walkway(4.0, 4.0)
    boundary = PeriodicBoundary2D(walkway)

    dt = 0.03
    sensing_radius = 0.4
    num_agents = 4

    behavior_params = {
        "sensing_radius": sensing_radius,
        "sensing_half_angle": SENSING_HALF_ANGLE,
        "d_stop": 0.15,
        "d_slow": 0.4,
        "circle_center_x": CIRCLE_CENTER_X,
        "circle_center_y": CIRCLE_CENTER_Y,
        "orbit_radius": ORBIT_RADIUS,
        "angular_speed": ANGULAR_SPEED,
        "radial_gain": 2.0,
        "passing_radius_offset": 0.3,
        "passing_target_gain": 4.0,
        "side_sensing_radius": 0.15,
        "side_sensing_half_angle": math.radians(30.0),
        "circle_radius_min": CIRCLE_RADIUS_MIN,
        "circle_radius_max": CIRCLE_RADIUS_MAX,
        "radius_target_gain": 4.0,
    }

    agents = create_circle_agents(
        num_agents=num_agents,
        center_x=CIRCLE_CENTER_X,
        center_y=CIRCLE_CENTER_Y,
        radius=ORBIT_RADIUS,
        angular_speed_mean=ANGULAR_SPEED,
        angular_speed_std=ANGULAR_SPEED_STD
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
        orbit_center=(CIRCLE_CENTER_X, CIRCLE_CENTER_Y),
        orbit_radius=ORBIT_RADIUS,
        agent_radius_m=AGENT_RADIUS,
    )

    running = True
    max_steps = 10_000
    step_count = 0

    while running and step_count < max_steps:
        running = viewer.handle_events()

        if viewer.request_reset:
            agents = create_circle_agents(
                num_agents=num_agents,
                center_x=CIRCLE_CENTER_X,
                center_y=CIRCLE_CENTER_Y,
                radius=ORBIT_RADIUS,
                angular_speed_mean=ANGULAR_SPEED,
                angular_speed_std = ANGULAR_SPEED_STD
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
            step_count = 0
            viewer.paused = False
            viewer.single_step = False
            viewer.request_reset = False

        if not viewer.paused or viewer.single_step:
            sim.step()
            viewer.single_step = False
            step_count += 1

        viewer.draw(sim.agents, sim.time)

    pygame.quit()


if __name__ == "__main__":
    main()
