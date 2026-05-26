# Crowd Flow Simulation

Small Python simulations for crowd and pedestrian flow with a PyGame viewer.

## Quick start
Run commands from the repository root. Make sure the required Python packages are installed first (at least `pygame` and `numpy`).

```bash
pip install pygame numpy
python examples/run_simple.py
```

## Main commands
```bash
python examples/run_simple.py          # basic walkway simulation
python examples/run_with_logging.py    # simulation + CSV logging
python examples/try_simple_passing.py  # walkway with passing behavior
python examples/run_circle_flow.py     # circular flow scenario
```

## Key files
- `examples/run_simple.py` - easiest place to start
- `examples/run_with_logging.py` - example that saves CSV output
- `examples/try_simple_passing.py` - simple passing behavior example
- `examples/run_circle_flow.py` - circular/orbit scenario
- `crowd_sim/simulation.py` - main simulation loop
- `crowd_sim/behaviors.py` - agent behavior rules
- `crowd_sim/visualization/pygame_view.py` - viewer and keyboard controls
- `crowd_sim/data_io.py` - CSV logging utilities

## Viewer controls
- `p` - pause/resume
- `n` - step one frame while paused
- `r` - reset simulation
- `c` - toggle sensing cones
- `v` - toggle velocity vectors
- `i` - toggle agent IDs
- `t` - toggle trails
- close the window to exit

## Output
`python examples/run_with_logging.py` writes simulation data to `output/sim_data.csv`.