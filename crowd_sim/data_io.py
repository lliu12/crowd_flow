# data_io.py
import csv
import os
from typing import List, Callable, Optional

from .agents import Agent


class CSVLogger:
    """
    CSV logger for simulation data.

    Supports either step-based logging or time-based logging,
    and allows callers to provide custom headers and row builders.
    """
    def __init__(self,
                 filename: str,
                 log_interval_steps: Optional[int] = 1,
                 log_interval_time: Optional[float] = None,
                 row_builder: Optional[Callable[[int, float, Agent, dict], list]] = None,
                 header: Optional[list[str]] = None):
        """
        :param filename: CSV file to write to (will overwrite if exists).
        :param log_interval_steps: log every N simulation steps. Ignored if
                                   log_interval_time is provided.
        :param log_interval_time: log every N seconds of simulation time.
        :param row_builder: optional function (step, time, agent, context) -> list.
        :param header: optional CSV header row.
        """
        self.filename = filename
        self.log_interval_steps = None if log_interval_time is not None else max(1, log_interval_steps or 1)
        self.log_interval_time = log_interval_time
        self.row_builder = row_builder or self._default_row_builder
        self.header = header or self._default_header()
        self._file = None
        self._writer = None
        self._is_header_written = False
        self._next_log_time = 0.0
        self._time_epsilon = 1e-9
        self.rows_written = 0

    def _default_row_builder(self, step: int, t: float, a: Agent, context: dict) -> list:
        speed = a.speed()
        return [step, t, a.id, a.x, a.y, a.vx, a.vy, speed, a.desired_speed]

    def _default_header(self) -> list:
        return ["step", "time", "id", "x", "y", "vx", "vy", "speed", "desired_speed"]

    def reset_schedule(self, start_time: float = 0.0) -> None:
        self._next_log_time = start_time

    def open(self) -> None:
        """
        Open the CSV file and prepare for writing.
        """
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

        self._file = open(self.filename, mode="w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.header)
        self._is_header_written = True
        self.rows_written = 0
        self.reset_schedule(0.0)

    def close(self) -> None:
        """
        Close the CSV file.
        """
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None
            self._is_header_written = False

    def _should_log(self, step: int, time: float) -> bool:
        if self.log_interval_time is not None:
            if time + self._time_epsilon < self._next_log_time:
                return False
            while time + self._time_epsilon >= self._next_log_time:
                self._next_log_time += self.log_interval_time
            return True

        if self.log_interval_steps is None:
            return False

        return step % self.log_interval_steps == 0

    def log(self,
            step: int,
            time: float,
            agents: List[Agent],
            context: Optional[dict] = None) -> None:
        """
        Log all agents at this step if the configured interval is reached.
        """
        if not self._should_log(step, time):
            return

        if self._writer is None:
            raise RuntimeError("CSVLogger: file not opened. Call open() first.")

        row_context = context or {}
        for a in agents:
            row = self.row_builder(step, time, a, row_context)
            self._writer.writerow(row)
            self.rows_written += 1

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
