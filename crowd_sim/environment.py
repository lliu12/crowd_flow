# environment.py
from abc import ABC, abstractmethod
from typing import List
from .agents import Agent


class Walkway:
    """
    2D rectangular domain: [0, Lx) x [0, Ly)
    """
    def __init__(self, Lx: float, Ly: float):
        self.Lx = Lx
        self.Ly = Ly


class BoundaryHandler(ABC):
    """
    Base class for boundary conditions.
    """
    def __init__(self, walkway: Walkway):
        self.walkway = walkway

    @abstractmethod
    def apply(self, agents: List[Agent]) -> None:
        """
        Apply boundary conditions in-place to agent positions.
        """
        ...


class PeriodicBoundary2D(BoundaryHandler):
    """
    Periodic in both x and y.
    """
    def apply(self, agents: List[Agent]) -> None:
        Lx = self.walkway.Lx
        Ly = self.walkway.Ly
        for a in agents:
            # wrap x
            if a.x < 0.0:
                a.x += Lx
            elif a.x >= Lx:
                a.x -= Lx
            # wrap y
            if a.y < 0.0:
                a.y += Ly
            elif a.y >= Ly:
                a.y -= Ly


class OpenBoundaryWithOverflow2D(BoundaryHandler):
    """
    Non-periodic: agents leaving the domain are moved to an overflow list.
    For now, overflow agents are not re-inserted or used in dynamics.
    """
    def __init__(self, walkway: Walkway):
        super().__init__(walkway)
        self.overflow: List[Agent] = []

    def apply(self, agents: List[Agent]) -> None:
        Lx = self.walkway.Lx
        Ly = self.walkway.Ly
        remaining = []
        for a in agents:
            if 0.0 <= a.x < Lx and 0.0 <= a.y < Ly:
                remaining.append(a)
            else:
                self.overflow.append(a)
        # mutate original list in-place
        agents.clear()
        agents.extend(remaining)