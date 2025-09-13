from shapes.mesh import Mesh
from shapes.body_system import BodySystem
import pygame
import numpy as np

# TODO make mesh and scale property

class Rigid():
    
    def __init__(self, system: BodySystem, mesh: Mesh, pos: tuple, scale: tuple, vel: tuple=None, friction: float=0.8, density: float=1, color: tuple=None) -> None:
        self.system = system
        
        # default parameter values
        vel = vel if vel else np.zeros(3)
        
        # add self to solver linked list
        self.next = self.solver.bodies
        self.solver.bodies = self
        
        self.color = color if color is not None else [0.5, 0.5, 0.5]
        
        # compute mass and moment TODO this is only correct for the basic cube mesh
        mass = scale[0] * scale[1] * density
        moment = mass * np.dot(scale, scale) / 12
        
        # add self to body system
        self.index = self.system.insert(pos, vel, friction, mass, moment, scale, mesh.index)
        self.system.bodies[self.index] = self
        
        # start force linked list
        self.forces = None
        
        # graph coloring
        self.reset_coloring()
        
    # --------------------
    # Self Linked List Operations
    # --------------------
        
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
    
    # --------------------
    # Pygame Draw
    # --------------------

    def draw(self, screen, scale_factor=20, offset=(400, 300), color=None, draw_normals=False, normal_length=1.0, normal_color=(255, 0, 0)):
        """Draws the rectangle with normals emanating from the body center."""
        
        # Use consistent transformations
        world_vertices = self.trans_vertices
        color = self.color if color is None else color
        
        screen_points = []
        for world_point in world_vertices:
            sx = int(world_point[0] * scale_factor + offset[0])
            sy = int(-world_point[1] * scale_factor + offset[1])
            screen_points.append((sx, sy))
        
        pygame.draw.polygon(screen, color, screen_points, 2)
        
        if draw_normals:
            # Use your precomputed matrix for consistency
            world_normals = self.mesh.normals @ self.inv_rot_sca_mat
            
            # Calculate body center in world space
            body_center = np.mean(world_vertices, axis=0)
            
            # Convert center to screen coordinates
            center_sx = int(body_center[0] * scale_factor + offset[0])
            center_sy = int(-body_center[1] * scale_factor + offset[1])
            
            # Draw normals from body center
            for i, normal in enumerate(world_normals):
                normal_end = body_center + normal * normal_length
                
                end_sx = int(normal_end[0] * scale_factor + offset[0])
                end_sy = int(-normal_end[1] * scale_factor + offset[1])
                
                pygame.draw.line(screen, normal_color, (center_sx, center_sy), (end_sx, end_sy), 2)
                pygame.draw.circle(screen, normal_color, (end_sx, end_sy), 3)
        
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
    
    def num_adjacent(self) -> int:
        return sum(1 for _ in self.get_adjacent_bodies())
        
    def reset_coloring(self) -> None:
        self.color_next = None
        self.graph_color = -1
        self.degree = self.num_adjacent()
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
        
    def assign_color(self, color: int) -> list:
        # check errors
        assert not self.is_colored(), 'Rigid: Colored Rigid was attemping to recolor' # TODO remove this once we get persistent graph coloring
        assert color not in self.used_colors, 'Rigid: Color is in self.used_colors'
        
        self.graph_color = color
        self.used_colors.add(color)
        
        # this list will store the rigid bodies that are adjacent to this one and have their satuartion degree changed
        to_update = []
        
        # update the adjacency list (saturation and used colors)
        for adjacent_body in self.get_adjacent_bodies():
            if adjacent_body.is_colored() or color in adjacent_body.used_colors:  # Only update uncolored neighbors
                continue
                
            adjacent_body.used_colors.add(color)
            adjacent_body.saturation_degree += 1
            to_update.append(adjacent_body)
            
        return to_update
        
    # --------------------
    # Properties
    # --------------------
        
    # piped access properties
    @property
    def solver(self):
        return self.system.solver
    
    @property
    def vertices(self):
        return self.mesh.vertices
    
    # TODO do manual transformation
    @property
    def trans_vertices(self):
        verts = self.vertices @ self.irs.T
        return verts + self.pos[:2]
    
    # --------------------
    # body system properties
    # --------------------
    @property
    def mesh(self) -> Mesh:
        return self.solver.mesh_system.meshes[self.system.mesh[self.index]]
    
    @property
    def radius(self) -> float:
        return self.system.radius[self.index]
    
    @property
    def irs(self):
        return self.system.irs[self.index]
    
    @property
    def s_ir(self):
        return self.system.s_ir[self.index]
    
    @property
    def scale(self):
        return self.system.scale[self.index]
        
    @property
    def updated(self):
        return self.system.updated[self.index]
    
    @property
    def pos(self):
        return self.system.pos[self.index]
        
    @property
    def initial(self):
        return self.system.initial[self.index]
        
    @property
    def inertial(self):
        return self.system.inertial[self.index]
        
    @property
    def vel(self):
        return self.system.vel[self.index]
        
    @property
    def prev_vel(self):
        return self.system.prev_vel[self.index]
        
    @property
    def friction(self):
        return self.system.friction[self.index]
        
    @property
    def mass(self):
        return self.system.mass[self.index]
        
    @property
    def moment(self):
        return self.system.moment[self.index]