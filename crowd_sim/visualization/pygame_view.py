# visualization/pygame_view.py
import math
from typing import List, Optional, Tuple
import pygame

from ..agents import Agent
from ..environment import Walkway


class PygameViewer:
    """
    Simple PyGame-based viewer for the 2D walkway.
    - Draws agents as circles.
    - Draws all sensing cones.
    - Colors agents by speed (red=0, green/blue for moving).
    """
    def __init__(self,
                 walkway: Walkway,
                 sensing_radius: float,
                 pixels_per_meter: float = 80.0,
                 fps_cap: int = 20,
                 sensing_half_angle: float = math.radians(60.0),
                 side_sensing_radius: Optional[float] = None,
                 side_sensing_half_angle: float = math.radians(45.0),
                 side_heading_offset: float = math.pi / 2.0,
                 orbit_center: Optional[Tuple[float, float]] = None,
                 orbit_radius: Optional[float] = None,
                 agent_radius_m = 0.2):
        pygame.init()
        self.walkway = walkway
        self.sensing_radius = sensing_radius
        self.ppm = pixels_per_meter
        self.width_px = int(self.walkway.Lx * self.ppm)
        self.height_px = int(self.walkway.Ly * self.ppm)
        self.screen = pygame.display.set_mode((self.width_px, self.height_px))
        pygame.display.set_caption("Crowd Simulation")
        self.clock = pygame.time.Clock()
        self.fps_cap = fps_cap
        self.sensing_half_angle = sensing_half_angle
        self.side_sensing_radius = side_sensing_radius if side_sensing_radius is not None else sensing_radius
        self.side_sensing_half_angle = side_sensing_half_angle
        self.side_heading_offset = side_heading_offset
        self.orbit_center = orbit_center
        self.orbit_radius = orbit_radius

        self.background_color = (255, 255, 255) # (0, 0, 0)
        self.agent_radius_m = agent_radius_m  # meters
        self.agent_radius_px = int(self.agent_radius_m * self.ppm)

        self.show_cones = True
        self.paused = False
        self.single_step = False
        self.show_velocities = False
        self.show_ids = False
        self.id_font = pygame.font.SysFont(None, 14)
        self.request_reset = False
        self.show_trails = False


    def handle_events(self) -> bool:
        """Return False if the main loop should exit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    self.toggle_cones()
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                elif event.key == pygame.K_n:
                    self.single_step = True  # advance one frame when paused
                elif event.key == pygame.K_v:
                    self.show_velocities = not self.show_velocities
                elif event.key == pygame.K_i:
                    self.show_ids = not self.show_ids
                elif event.key == pygame.K_r:
                    self.request_reset = True
                elif event.key == pygame.K_t:
                    self.show_trails = not self.show_trails
                    
        return True

    def toggle_cones(self) -> None:
        self.show_cones = not self.show_cones

    def world_to_screen(self, x: float, y: float) -> (int, int):
        """
        Map world coords (meters) to screen coords (pixels).
        - x to right, y up in world;
        - Pygame y is down, so we flip y.
        """
        sx = int(x * self.ppm)
        sy = int((self.walkway.Ly - y) * self.ppm)
        return sx, sy

    def speed_to_color(self, speed: float, v0: float) -> (int, int, int):
        """
        Map speed in [0, v0] to RGB.
        - 0   -> red
        - v0  -> blue/green
        We'll do a simple two-phase blend:
        - 0 -> v0/2: red -> green
        - v0/2 -> v0: green -> cyan/blue
        """
        if v0 <= 0:
            return (255, 0, 0)

        t = max(0.0, min(1.0, speed / v0))

        if t < 0.5:
            # red (1,0,0) to green (0,1,0)
            alpha = t / 0.5
            r = int((1.0 - alpha) * 255)
            g = int(alpha * 255)
            b = 0
        else:
            # green (0,1,0) to cyan/blue-ish (0,1,1)
            alpha = (t - 0.5) / 0.5
            r = 0
            g = 255
            b = int(alpha * 255)

        return (r, g, b)

    def draw_cone_outline(self,
                          x: float,
                          y: float,
                          direction_angle: float,
                          half_angle: float,
                          radius: float,
                          color: Tuple[int, int, int],
                          num_segments: int = 16) -> None:
        left_ang = direction_angle - half_angle
        right_ang = direction_angle + half_angle

        sx, sy = self.world_to_screen(x, y)
        ex_left = x + radius * math.cos(left_ang)
        ey_left = y + radius * math.sin(left_ang)
        ex_right = x + radius * math.cos(right_ang)
        ey_right = y + radius * math.sin(right_ang)
        esx_left, esy_left = self.world_to_screen(ex_left, ey_left)
        esx_right, esy_right = self.world_to_screen(ex_right, ey_right)

        pygame.draw.line(self.screen, color, (sx, sy), (esx_left, esy_left), width=1)
        pygame.draw.line(self.screen, color, (sx, sy), (esx_right, esy_right), width=1)

        arc_points = []
        for i in range(num_segments + 1):
            t = i / num_segments
            ang = left_ang + t * (right_ang - left_ang)
            ex = x + radius * math.cos(ang)
            ey = y + radius * math.sin(ang)
            arc_points.append(self.world_to_screen(ex, ey))

        if len(arc_points) >= 2:
            pygame.draw.lines(self.screen, color, False, arc_points, width=1)


    def draw_agents_and_cones(self, agents: List[Agent]) -> None:
        for a in agents:
            sx, sy = self.world_to_screen(a.x, a.y)
            speed = a.speed()
            color = self.speed_to_color(speed, 2)

            # Draw agent as circle
            pygame.draw.circle(self.screen, color, (sx, sy), self.agent_radius_px)

            if self.show_cones:
                dir_x = a.dir_x
                dir_y = a.dir_y
                norm = math.hypot(dir_x, dir_y)
                if norm == 0.0:
                    dir_x, dir_y = 1.0, 0.0
                    norm = 1.0
                dir_x /= norm
                dir_y /= norm

                angle = math.atan2(dir_y, dir_x)
                left_angle = angle + self.side_heading_offset
                right_angle = angle - self.side_heading_offset

                forward_color = (120, 120, 120) if not getattr(a, "blocked", False) else (80, 80, 180)
                side_clear_color =  (200, 230, 250) # (250, 230, 200)
                side_blocked_color = (255, 150, 160) #  (160, 150, 255)
                left_color = side_clear_color if not getattr(a, "left_blocked", False) else side_blocked_color
                right_color = side_clear_color if not getattr(a, "right_blocked", False) else side_blocked_color

                self.draw_cone_outline(
                    a.x, a.y, angle, self.sensing_half_angle, self.sensing_radius, forward_color, num_segments=24
                )
                self.draw_cone_outline(
                    a.x, a.y, left_angle, self.side_sensing_half_angle, self.side_sensing_radius, left_color, num_segments=12
                )
                self.draw_cone_outline(
                    a.x, a.y, right_angle, self.side_sensing_half_angle, self.side_sensing_radius, right_color, num_segments=12
                )

            if self.show_velocities:
                vx, vy = a.vx, a.vy  # or however your Agent exposes velocity
                # length scale (tune this)
                scale = 1
                ex = a.x + vx * scale
                ey = a.y + vy * scale
                esx, esy = self.world_to_screen(ex, ey)
                pygame.draw.line(self.screen, (0, 0, 0), (sx, sy), (esx, esy), width=1)

            if self.show_ids:
                # Fallback if no numeric id: str(a) or any attribute
                label = str(getattr(a, "id", "?"))
                text_surface = self.id_font.render(label, True, (0, 0, 0))
                # small offset so text isn't centered on the circle
                self.screen.blit(text_surface, (sx + 4, sy - 4))

            if self.show_trails and len(a.trail) > 1:
                n = len(a.trail)
                for i, (tx, ty) in enumerate(a.trail):
                    # i=0 oldest, i=n-1 newest
                    t = (i + 1) / n  # 0..1
                    # newer points darker
                    v = int(240 - 40 * t)
                    color = (v, v, v)

                    tsx, tsy = self.world_to_screen(tx, ty)
                    pygame.draw.circle(self.screen, color, (tsx, tsy), 2)

    def draw(self, agents: List[Agent], sim_time: float) -> None:
        self.screen.fill(self.background_color)

        if self.orbit_center is not None and self.orbit_radius is not None:
            cx, cy = self.world_to_screen(*self.orbit_center)
            pygame.draw.circle(
                self.screen,
                (180, 180, 180),
                (cx, cy),
                int(self.orbit_radius * self.ppm),
                width=1,
            )

        # Draw agents + cones
        self.draw_agents_and_cones(agents)

        # Optionally draw time text
        font = pygame.font.SysFont(None, 24)
        text_surface = font.render(f"t = {sim_time:.2f} s", True, (20, 20, 20))
        self.screen.blit(text_surface, (10, 10))

        pygame.display.flip()
        self.clock.tick(self.fps_cap)