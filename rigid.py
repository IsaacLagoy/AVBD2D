import glm
from glm import vec2, vec3
import pygame
from maths import transform
from mesh import Mesh


class Rigid():
    
    def __init__(self, mesh: Mesh, pos: vec3, scale: vec2, vel: vec3=None, friction: float=0.4, density: float=1, color: vec3=None) -> None:
        self.mesh = mesh
        self._pos = pos
        self.scale = scale
        self.vel = vel if vel else vec3(0)
        self.prev_vel = self.vel
        self.friction = friction
        self.color = color if color else vec3(0.5)
        
        self.mass = scale.x * scale.y * density
        self.radius = glm.length(scale)
        self.moment = self.mass * glm.dot(scale, scale) / 12
        
        self.inertial = vec3()
        self.initial = vec3()
        
        self.forces = []
        
    def is_constrained_to(self, body) -> bool:
        for force in self.forces:
            if body is force.body_a or body is force.body_b:
                return True
        return False
        
    def draw(self, screen, scale_factor=20, offset=(400, 300)):
        """Draws the rectangle in Pygame coordinates, accounting for position, rotation, and scale."""
        # Transform each corner from local space to world space, then to screen space
        screen_points = []
        for world_point in self.vertices:
            # Transform world point to screen coordinates
            sx = int(world_point.x * scale_factor + offset[0])
            sy = int(-world_point.y * scale_factor + offset[1]) # Pygame's y-axis is inverted
            screen_points.append((sx, sy))
        
        # Draw the polygon using the transformed corners
        pygame.draw.polygon(screen, self.color, screen_points, 2)
        
    @property
    def pos(self) -> vec3:
        return self._pos
    
    @pos.setter
    def pos(self, value) -> None:
        assert isinstance(value, vec3), f'Rigid: pos is not vec3, {type(value)}'
        self._pos = value
        
    @property
    def vertices(self) -> list[vec2]:
        vs = [transform(self.pos, self.scale, v) for v in self.mesh.vertices]
        return vs
    
    @property
    def edges(self) -> list[tuple[vec2, vec2]]:
        vertices = self.vertices
        return [(vertices[i], vertices[(i + 1) % 4]) for i in range(len(self.vertices))]