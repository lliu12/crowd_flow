# data_io.py
import csv
import os
from typing import List, Callable, Optional

from .agents import Agent


class CSVLogger:
    """
    Simple CSV logger for simulation data.

    Logs, at a specified interval:
        time, id, x, y, vx, vy, speed

    You can extend this later by:
    - Adding more fields (e.g., neighbor_count, direction, etc.).
    - Passing a custom row_builder.
    """
    def __init__(self,
                 filename: str,
                 log_interval_steps: int = 1,
                 row_builder: Optional[Callable[[float, Agent], list]] = None):
        """
        :param filename: CSV file to write to (will overwrite if exists).
        :param log_interval_steps: log every N simulation steps.
        :param row_builder: optional function (time, agent) -> list of row values.
                            If None, use default: [time, id, x, y, vx, vy, speed].
        """
        self.filename = filename
        self.log_interval_steps = max(1, log_interval_steps)
        self.row_builder = row_builder or self._default_row_builder
        self._file = None
        self._writer = None
        self._is_header_written = False

    def _default_row_builder(self, t: float, a: Agent) -> list:
        speed = a.speed()
        return [t, a.id, a.x, a.y, a.vx, a.vy, speed, a.desired_speed]

    def _default_header(self) -> list:
        return ["time", "id", "x", "y", "vx", "vy", "speed", "desired_speed"]

    def open(self) -> None:
        """
        Open the CSV file and prepare for writing.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

        self._file = open(self.filename, mode="w", newline="")
        self._writer = csv.writer(self._file)
        # Write header
        header = self._default_header()
        self._writer.writerow(header)
        self._is_header_written = True

    def close(self) -> None:
        """
        Close the CSV file.
        """
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None
            self._is_header_written = False

    def log(self, step: int, time: float, agents: List[Agent]) -> None:
        """
        Log all agents at this step if step % log_interval_steps == 0.
        """
        if step % self.log_interval_steps != 0:
            return

        if self._writer is None:
            raise RuntimeError("CSVLogger: file not opened. Call open() first.")

        for a in agents:
            row = self.row_builder(time, a)
            self._writer.writerow(row)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()