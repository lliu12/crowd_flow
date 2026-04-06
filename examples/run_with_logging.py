# examples/run_with_logging.py
import os
import sys
import random
from typing import Optional

# Allow running this script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame  # noqa: E402

from crowd_sim.agents import Agent
from crowd_sim.environment import Walkway, PeriodicBoundary2D
from crowd_sim.simulation import Simulation
from crowd_sim.visualization.pygame_view import PygameViewer
from crowd_sim.data_io import CSVLogger


def create_initial_agents(
    num_agents: int,
    walkway: Walkway,
    desired_speed_mean: float,
    desired_speed_std: Optional[float] = None,
) -> list[Agent]:
    """
    Create agents with initial positions uniformly distributed in the walkway.

    - If speed_std is None: all agents have speed = desired_speed.
    - If speed_std is not None: each agent's speed is drawn from
      N(desired_speed, speed_std^2).
    """

    agents = []
    for i in range(num_agents):
        # random initial position in the domain
        x = random.uniform(0.0, walkway.Lx)
        y = random.uniform(0.0, walkway.Ly)

        # speed: fixed or normally distributed
        if desired_speed_std is None or desired_speed_std == 0.0:
            spd = desired_speed_mean
        else:
            spd = random.gauss(desired_speed_mean, desired_speed_std)

        # initial direction: +x
        dir_x, dir_y = 1.0, 0.0
        vx = spd * dir_x
        vy = spd * dir_y

        a = Agent(
            id=i,
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            desired_speed=spd,
            dir_x=dir_x,
            dir_y=dir_y,
        )
        agents.append(a)

    return agents

def main():
    # Domain and params
    Lx = 10.0  # m
    Ly = 3.0   # m
    walkway = Walkway(Lx, Ly)

    dt = 0.05  # s
    sensing_radius = 3.0  # m
    num_agents = 30
    
    # all agents have the same desired speed
    desired_speed_mean = 1.4  # m/s
    desired_speed_std = 0.5

    boundary = PeriodicBoundary2D(walkway)
    agents = create_initial_agents(num_agents, walkway, desired_speed_mean, desired_speed_std=desired_speed_std)

    sim = Simulation(
        walkway=walkway,
        boundary_handler=boundary,
        agents=agents,
        dt=dt,
        sensing_radius=sensing_radius,
        periodic_x=True,
        periodic_y=True,
    )

    viewer = PygameViewer(
        walkway=walkway,
        sensing_radius=sensing_radius,
        pixels_per_meter=80.0,
        fps_cap=20,
    )

    # Logging: every 5 steps (~0.25 s)
    log_interval_steps = 5
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    output_file = os.path.join(output_dir, "sim_data.csv")

    running = True
    max_steps = 10_000
    step_count = 0

    with CSVLogger(output_file, log_interval_steps=log_interval_steps) as logger:
        while running and step_count < max_steps:
            running = viewer.handle_events()

            if viewer.request_reset:
                # 1) Reset environment and agents
                agents = create_initial_agents(num_agents, walkway, desired_speed_mean, desired_speed_std=desired_speed_std)
                sim = Simulation(
                    walkway=walkway,
                    boundary_handler=boundary,
                    agents=agents,
                    dt=dt,
                    sensing_radius=sensing_radius,
                    periodic_x=True,
                    periodic_y=True,
                )

                # 2) Reset simulation time and viewer state as needed
                step_count = 0
                viewer.paused = False
                viewer.single_step = False
                viewer.request_reset = False

            # Advance simulation
            if not viewer.paused or viewer.single_step:
                sim.step()
                viewer.single_step = False
                step_count += 1

                # Draw
                viewer.draw(sim.agents, sim.time)

                # Log state
                logger.log(step_count, sim.time, sim.agents)


    pygame.quit()
    print(f"Simulation finished. Data saved to: {output_file}")


if __name__ == "__main__":
    main()