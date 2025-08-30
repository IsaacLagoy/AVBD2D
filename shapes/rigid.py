import glm
from glm import vec2, vec3
import pygame
from helper.maths import transform
from shapes.mesh import Mesh


class Rigid():
    
    def __init__(self, solver, mesh: Mesh, pos: vec3, scale: vec2, vel: vec3=None, friction: float=0.8, density: float=1, color: vec3=None) -> None:
        # add to solver linked list
        self.solver = solver
        self.next = solver.bodies
        solver.bodies = self
        
        # initialize variables
        self.mesh = mesh
        self.pos = pos
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
        
        # start linked list - head of force list for this body
        self.forces = None
        
        # graph coloring
        self.reset_coloring()
        
        # lazy updating variables
        self.update_vertices = True
        
    # remove self from solver linked list
    def remove_self(self) -> None:
        if self.solver.bodies is self:
            self.solver.bodies = self.next
            return

        prev = None
        current = self.solver.bodies
        while current is not None and current is not self:
            prev = current
            current = current.next

        if prev is not None and current is self:
            prev.next = self.next
          
    # remove a force from the linked list  
    def remove_force(self, force) -> None:
        # Head of the list?
        if self.forces is force:
            # Pick correct next pointer depending on whether this rigid is bodyA or bodyB
            self.forces = force.next_a if force.body_a is self else force.next_b
            return

        # Traverse the list to find the force
        prev = None
        current = self.forces
        
        while current is not None and current is not force:
            prev = current
            # Move to next force in this body's list
            current = current.next_a if current.body_a is self else current.next_b

        # If we found the force, remove it
        if current is force and prev is not None:
            # Update the previous node's next pointer
            if prev.body_a is self:
                prev.next_a = force.next_a if force.body_a is self else force.next_b
            else:
                prev.next_b = force.next_a if force.body_a is self else force.next_b
        
    def is_constrained_to(self, body) -> bool:
        current = self.forces
        while current is not None:
            if body is current.body_a or body is current.body_b:
                return True
            # Move to next force in this body's list
            current = current.next_a if current.body_a is self else current.next_b
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
        
    # --------------------
    # Iterators
    # --------------------
    
    def get_adjacent_bodies(self):
        """
        Iterator for rigid body adjacency list
        """
        for force in self.get_forces_iterator():
            other_body = force.body_b if force.body_a is self else force.body_a
            if other_body:
                yield other_body
        
    def get_forces_iterator(self):
        """
        Iterator for forces linked list
        """
        current = self.forces
        while current is not None:
            yield current
            current = current.next_a if self is current.body_a else current.next_b
        
    # --------------------
    # Graph Coloring Methods
    # --------------------
        
    def reset_coloring(self) -> None:
        self.color_next = None
        self.graph_color = -1
        self.degree = 0
        self.saturation_degree = 0
        self.used_colors = set()
        
    def get_next_unused_color(self) -> int:
        """Find the smallest color not used by any adjacent body"""
        color_candidate = 0
        while color_candidate in self.used_colors:
            color_candidate += 1
        return color_candidate
            
    def is_colored(self) -> bool:
        return self.graph_color != -1
        
    def assign_color(self, color: int) -> None:
        # check errors
        assert not self.is_colored(), 'Rigid: Colored Rigid was attemping to recolor' # TODO remove this once we get persistent graph coloring
        assert color not in self.used_colors, 'Rigid: Color is in self.used_colors'
        
        self.graph_color = color
        self.used_colors.add(color)
        
        # update the adjacency list (saturation and used colors)
        for adjacent_body in self.get_adjacent_bodies():
            if not adjacent_body.is_colored():  # Only update uncolored neighbors
                adjacent_body.used_colors.add(color)
                adjacent_body.saturation_degree = len(adjacent_body.used_colors)
                
    # --------------------
    # Properties
    # --------------------
    
    @property
    def pos(self) -> vec3:
        return self._pos
    
    @pos.setter
    def pos(self, value) -> None:
        assert isinstance(value, vec3), f'Rigid: pos is not vec3, {type(value)}'
        self._pos = value
        self.update_vertices = True
        
    @property
    def vertices(self) -> list[vec2]:
        # lazy check
        if not self.update_vertices:
            return self._vertices
        
        self._vertices = [transform(self.pos, self.scale, v) for v in self.mesh.vertices]
        self.update_vertices = False
        return self.vertices
    
    @property
    def edges(self) -> list[tuple[vec2, vec2]]:
        vertices = self.vertices
        return [(vertices[i], vertices[(i + 1) % len(self.mesh.vertices)]) for i in range(len(self.vertices))]