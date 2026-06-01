import math
import os
import sys
import time

# Allow running this script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crowd_sim.behaviors import circular_robotics_behavior
from crowd_sim.data_io import CSVLogger
from crowd_sim.environment import Walkway, OpenBoundaryWithOverflow2D
from crowd_sim.simulation import Simulation
from examples.run_circle_flow import (
    WALKWAY_LENGTH,
    CIRCLE_CENTER_X,
    CIRCLE_CENTER_Y,
    SENSING_HALF_ANGLE,
    create_circle_agents,
)


NUM_AGENTS_OPTIONS = [1, 3, 5, 7, 10, 15, 20, 25, 30, 40]
MAX_SPEEDUP_ACCELERATION_OPTIONS = [None, 0.05]
REACTION_DELAY_OPTIONS = [0.0, 0.05]
NUM_TRIALS = 20
EXPERIMENT_NAME = "20260529_circle_flow_acceleration"
LOG_INTERVAL_TIME = 10.0
ROUND_DIGITS = 3


def round_if_number(value, digits: int = ROUND_DIGITS):
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(value, digits)
    return value


def format_acceleration_label(value) -> str:
    return f"{value:.3f}" if value is not None else "unlimited"


def format_delay_label(value) -> str:
    return f"{value:.3f}"


def make_behavior_params(
    sensing_radius: float,
    circle_radius: float,
    circle_radius_min: float,
    circle_radius_max: float,
    d_stop: float,
    d_slow: float,
    lane_preference: str,
    max_speedup_acceleration: float,
    reaction_delay: float,
    dt: float,
) -> dict:
    return {
        "sensing_radius": sensing_radius,
        "sensing_half_angle": SENSING_HALF_ANGLE,
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
        "max_speedup_acceleration": max_speedup_acceleration,
        "sim_time": 0.0,
        "reaction_delay": reaction_delay,
        "reaction_delay_steps": int(round(reaction_delay / dt)),
    }


def build_circle_experiment_row(step, time, agent, context):
    return [
        round_if_number(context["max_speedup_acceleration"]),
        context["max_speedup_acceleration_label"],
        round_if_number(context["reaction_delay"]),
        context["reaction_delay_label"],
        context["reaction_delay_steps"],
        context["trial"],
        context["num_robots"],
        step,
        round_if_number(time),
        agent.id,
        round_if_number(agent.x),
        round_if_number(agent.y),
        round_if_number(agent.heading),
        round_if_number(getattr(agent, "target_tangential_speed", None)),
        round_if_number(getattr(agent, "executed_tangential_speed_command", None)),
        round_if_number(getattr(agent, "realized_tangential_speed", None)),
        round_if_number(getattr(agent, "current_speed", agent.speed())),
        round_if_number(getattr(agent, "current_radius", None)),
        round_if_number(getattr(agent, "target_radius", None)),
        round_if_number(getattr(agent, "executed_target_radius", None)),
        getattr(agent, "executed_command_age_steps", None),
        round_if_number(getattr(agent, "angular_speed", None)),
        round_if_number(getattr(agent, "lap_count_ccw", None)),
        round_if_number(agent.vx),
        round_if_number(agent.vy),
        round_if_number(agent.desired_speed),
        agent.blocked,
        agent.left_blocked,
        agent.right_blocked,
        round_if_number(agent.dist_to_nearest),
        round_if_number(agent.angle_to_nearest),
        round_if_number(getattr(agent, "left_dist_to_nearest", None)),
        round_if_number(getattr(agent, "left_angle_to_nearest", None)),
        round_if_number(getattr(agent, "right_dist_to_nearest", None)),
        round_if_number(getattr(agent, "right_angle_to_nearest", None)),
        round_if_number(getattr(agent, "approach_rate", None)),
        getattr(agent, "pass_allowed", None),
        round_if_number(getattr(agent, "base_orbit_radius", None)),
        round_if_number(context["dt"]),
    ]


def build_circle_experiment_header() -> list[str]:
    return [
        "max_speedup_acceleration",
        "max_speedup_acceleration_label",
        "reaction_delay",
        "reaction_delay_label",
        "reaction_delay_steps",
        "trial",
        "num_robots",
        "step",
        "time",
        "robot_id",
        "x",
        "y",
        "heading",
        "target_tangential_speed",
        "executed_tangential_speed_command",
        "realized_tangential_speed",
        "current_speed",
        "current_radius",
        "target_radius",
        "executed_target_radius",
        "executed_command_age_steps",
        "angular_speed",
        "lap_count_ccw",
        "vx",
        "vy",
        "desired_speed",
        "blocked",
        "left_blocked",
        "right_blocked",
        "dist_to_nearest",
        "angle_to_nearest",
        "left_dist_to_nearest",
        "left_angle_to_nearest",
        "right_dist_to_nearest",
        "right_angle_to_nearest",
        "approach_rate",
        "pass_allowed",
        "base_orbit_radius",
        "sim_dt",
    ]


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def main():
    walkway = Walkway(WALKWAY_LENGTH, WALKWAY_LENGTH)
    boundary = OpenBoundaryWithOverflow2D(walkway)

    dt = 0.05
    sim_time = 300.0
    sensing_radius = 0.4
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

    output_dir = os.path.join(os.path.dirname(__file__), "..", "output", EXPERIMENT_NAME)
    output_file = os.path.join(output_dir, "circle_flow_acceleration_experiments.csv")

    total_trials = 0
    total_trial_count = (
        len(MAX_SPEEDUP_ACCELERATION_OPTIONS)
        * len(REACTION_DELAY_OPTIONS)
        * len(NUM_AGENTS_OPTIONS)
        * NUM_TRIALS
    )
    experiment_start_time = time.perf_counter()

    print(f"Starting experiment: {EXPERIMENT_NAME}")
    print(f"Output file: {output_file}")
    print(f"Acceleration limits: {[format_acceleration_label(v) for v in MAX_SPEEDUP_ACCELERATION_OPTIONS]}")
    print(f"Reaction delays: {[format_delay_label(v) for v in REACTION_DELAY_OPTIONS]}")
    print(f"Agent settings: {NUM_AGENTS_OPTIONS}")
    print(f"Total trials: {total_trial_count}")

    with CSVLogger(
        output_file,
        log_interval_time=LOG_INTERVAL_TIME,
        row_builder=build_circle_experiment_row,
        header=build_circle_experiment_header(),
    ) as logger:
        for reaction_delay in REACTION_DELAY_OPTIONS:
            delay_label = format_delay_label(reaction_delay)
            delay_steps = int(round(reaction_delay / dt))
            print(
                f"\nStarting reaction_delay={delay_label}s (delay_steps={delay_steps})",
                flush=True,
            )

            for max_speedup_acceleration in MAX_SPEEDUP_ACCELERATION_OPTIONS:
                accel_label = format_acceleration_label(max_speedup_acceleration)
                print(
                    f"  Starting max_speedup_acceleration={accel_label}",
                    flush=True,
                )

                for num_agents in NUM_AGENTS_OPTIONS:
                    setting_start_time = time.perf_counter()
                    completed_before_setting = total_trials
                    print(
                        f"    Starting num_agents={num_agents} "
                        f"({completed_before_setting + 1}-{completed_before_setting + NUM_TRIALS} "
                        f"of {total_trial_count} trials)",
                        flush=True,
                    )

                    for trial in range(NUM_TRIALS):
                        trial_number = trial + 1
                        overall_trial_number = total_trials + 1
                        trial_start_time = time.perf_counter()
                        print(
                            f"      Trial {trial_number}/{NUM_TRIALS} for reaction_delay={delay_label}s, "
                            f"max_speedup_acceleration={accel_label}, num_agents={num_agents} "
                            f"(overall {overall_trial_number}/{total_trial_count})...",
                            flush=True,
                        )

                        behavior_params = make_behavior_params(
                            sensing_radius=sensing_radius,
                            circle_radius=circle_radius,
                            circle_radius_min=circle_radius_min,
                            circle_radius_max=circle_radius_max,
                            d_stop=d_stop,
                            d_slow=d_slow,
                            lane_preference=lane_preference,
                            max_speedup_acceleration=max_speedup_acceleration,
                            reaction_delay=reaction_delay,
                            dt=dt,
                        )

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

                        logger.reset_schedule(0.0)
                        step_count = 0
                        context = {
                            "experiment_name": EXPERIMENT_NAME,
                            "max_speedup_acceleration": max_speedup_acceleration,
                            "max_speedup_acceleration_label": accel_label,
                            "reaction_delay": reaction_delay,
                            "reaction_delay_label": delay_label,
                            "reaction_delay_steps": delay_steps,
                            "trial": trial,
                            "num_robots": num_agents,
                            "dt": dt,
                        }

                        logger.log(step_count, sim.time, sim.agents, context)
                        while sim.time < sim_time:
                            behavior_params["sim_time"] = sim.time
                            sim.step()
                            step_count += 1
                            logger.log(step_count, sim.time, sim.agents, context)

                        total_trials += 1
                        trial_elapsed = time.perf_counter() - trial_start_time
                        total_elapsed = time.perf_counter() - experiment_start_time
                        avg_trial_time = total_elapsed / total_trials
                        trials_remaining = total_trial_count - total_trials
                        eta_seconds = avg_trial_time * trials_remaining
                        print(
                            f"        Completed in {format_duration(trial_elapsed)} | "
                            f"elapsed {format_duration(total_elapsed)} | "
                            f"ETA {format_duration(eta_seconds)}",
                            flush=True,
                        )

                    setting_elapsed = time.perf_counter() - setting_start_time
                    print(
                        f"    Finished reaction_delay={delay_label}s, "
                        f"max_speedup_acceleration={accel_label}, num_agents={num_agents} "
                        f"in {format_duration(setting_elapsed)} "
                        f"({total_trials}/{total_trial_count} trials complete)",
                        flush=True,
                    )

    total_elapsed = time.perf_counter() - experiment_start_time
    print(f"\nExperiment finished. Data saved to: {output_file}")
    print(f"Trials completed: {total_trials}")
    print(f"Rows written: {logger.rows_written}")
    print(f"Total wall time: {format_duration(total_elapsed)}")


if __name__ == "__main__":
    main()
