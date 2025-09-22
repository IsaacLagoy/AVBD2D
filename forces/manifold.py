from forces.force import Force
from shapes.rigid import Rigid
import numpy as np
from math import sqrt
from helper.maths import rotate_scale, cross, transform
import pygame


COLLISION_MARGIN = 0.01
STICK_THRESH = 0.001
SHOW_CONTACTS = True

class Manifold(Force):
    
    def __init__(self, system, body_a: Rigid, body_b: Rigid, contact_index: int) -> None:
        super().__init__(system, body_a, body_b, 0)
        
        # insert into contacts list
        self.contact_index = contact_index
        self.system.contacts.contacts[self.contact_index] = self
        self.system.contacts.size += 1
        
        # # set starting values
        # self.fmax[0] = self.fmax[2] = 0
        # self.fmin[0] = self.fmin[2] = -np.inf
        
    def initialize(self) -> bool:
        return False
        
        self.friction = sqrt(self.body_a.friction * self.body_b.friction)
        self.num_contact = collide(self)
        
        # set penalty and lambda
        # TODO add persistent contact
        for i in range(self.num_contact):
            self.penalty[i * 2 + 0] = self.penalty[i * 2 + 1] = 0
            self.lamb[i * 2 + 0] = self.lamb[i * 2 + 1] = 0
            
        # precompute Jacobians
        for i in range(self.num_contact):
            n = self.normal[i]
            t = np.array([n[1], -n[0]], dtype='float32')
            basis = np.array([n, t], dtype='float32')
            
            rAW = rotate_scale(self.body_a.pos[2], self.body_a.scale, self.rA[i])
            rBW = rotate_scale(self.body_b.pos[2], self.body_b.scale, self.rB[i])
            
            self.JAn[i] = np.array([n[0], n[1], cross(rAW, n)], dtype='float32')
            self.JBn[i] = np.array([-n[0], -n[1], -cross(rBW, n)], dtype='float32')
            self.JAt[i] = np.array([t[0], t[1], cross(rAW, t)], dtype='float32')
            self.JBt[i] = np.array([-t[0], -t[1], -cross(rBW, t)], dtype='float32')
            
            self.C0[i] = basis @ (self.body_a.pos[:2] + rAW - self.body_b.pos[:2] - rBW + np.array([COLLISION_MARGIN, 0], dtype='float32'))

        return self.num_contact > 0
    
    def remove_self(self) -> None:
        super().remove_self()
        
        self.system.contacts.delete(self.contact_index)
        
    def draw(self, screen, scale_factor=20, offset=(400, 300)) -> None:
        if not SHOW_CONTACTS:
            return

        normal_length = 30  # pixels for drawing the normal

        for i in range(2):
            # Contact points in world space
            world_a = transform(self.body_a.pos, self.body_a.scale, self.rA[i])
            world_b = transform(self.body_b.pos, self.body_b.scale, self.rB[i])

            # Convert to screen space
            sx_a = int(world_a[0] * scale_factor + offset[0])
            sy_a = int(-world_a[1] * scale_factor + offset[1])  # invert y for pygame

            sx_b = int(world_b[0] * scale_factor + offset[0])
            sy_b = int(-world_b[1] * scale_factor + offset[1])

            # Draw as small circles
            pygame.draw.circle(screen, (255, 0, 0), (sx_a, sy_a), 4)
            pygame.draw.circle(screen, (0, 0, 255), (sx_b, sy_b), 4)

            # Draw contact normal (from A's contact point)
            n = self.normal[i]
            nx, ny = n[0], n[1]

            end_x = int(sx_a + nx * normal_length)
            end_y = int(sy_a - ny * normal_length)  # minus because y is inverted

            pygame.draw.line(screen, (200, 200, 200), (sx_a, sy_a), (end_x, end_y), 1)
            
            end_x = int(sx_b + nx * normal_length)
            end_y = int(sy_b - ny * normal_length)  # minus because y is inverted

            pygame.draw.line(screen, (200, 200, 200), (sx_b, sy_b), (end_x, end_y), 1)
        
    # -----------------
    # Properties
    # -----------------
    
    # Contact properties
    @property
    def normal(self):
        return self.system.contacts.normal[self.contact_index]
    
    @property
    def rA(self):
        return self.system.contacts.rA[self.contact_index]
    
    @property
    def rB(self):
        return self.system.contacts.rB[self.contact_index]