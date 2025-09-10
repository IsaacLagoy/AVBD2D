from forces.force import Force
from forces.force_system import ForceSystem
from shapes.rigid import Rigid
import numpy as np
from collision.collide import collide
from math import sqrt
from helper.maths import rotate_scale, cross, transform
import pygame


COLLISION_MARGIN = 0.01
STICK_THRESH = 0.001
SHOW_CONTACTS = True

class Manifold(Force):
    
    def __init__(self, system: ForceSystem, body_a: Rigid, body_b: Rigid) -> None:
        super().__init__(system, body_a, body_b, 0)
        
        # insert into contacts list
        self.contact_index = self.system.contacts.insert()
        self.system.contacts.contacts[self.contact_index] = self
        
        # set starting values
        self.fmax[0] = self.fmax[2] = 0
        self.fmin[0] = self.fmin[2] = -np.inf
        
    def initialize(self) -> bool:
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

        for i in range(self.num_contact):
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

            # # Draw contact normal (from A's contact point)
            # n = self.normal[i]
            # nx, ny = n.x, n.y

            # end_x = int(sx_a + nx * normal_length)
            # end_y = int(sy_a - ny * normal_length)  # minus because y is inverted

            # pygame.draw.line(screen, (200, 200, 200), (sx_a, sy_a), (end_x, end_y), 1)
        
    # -----------------
    # Properties
    # -----------------
    
    # Contact properties
    @property
    def normal(self):
        return self.system.contacts.normal[self.contact_index]
    
    @normal.setter
    def normal(self, value):
        self.system.contacts.normal[self.contact_index][:] = value
    
    @property
    def rA(self):
        return self.system.contacts.rA[self.contact_index]
    
    @rA.setter
    def rA(self, value):
        self.system.contacts.rA[self.contact_index][:] = value
    
    @property
    def rB(self):
        return self.system.contacts.rB[self.contact_index]
    
    @rB.setter
    def rB(self, value):
        self.system.contacts.rB[self.contact_index][:] = value
    
    @property
    def stick(self):
        return self.system.contacts.stick[self.contact_index]
    
    @stick.setter
    def stick(self, value):
        self.system.contacts.stick[self.contact_index][:] = value
    
    # Jacobians
    @property
    def JAn(self):
        return self.system.contacts.JAn[self.contact_index]
    
    @JAn.setter
    def JAn(self, value):
        self.system.contacts.JAn[self.contact_index][:] = value
    
    @property
    def JBn(self):
        return self.system.contacts.JBn[self.contact_index]
    
    @JBn.setter
    def JBn(self, value):
        self.system.contacts.JBn[self.contact_index][:] = value
    
    @property
    def JAt(self):
        return self.system.contacts.JAt[self.contact_index]
    
    @JAt.setter
    def JAt(self, value):
        self.system.contacts.JAt[self.contact_index][:] = value
    
    @property
    def JBt(self):
        return self.system.contacts.JBt[self.contact_index]
    
    @JBt.setter
    def JBt(self, value):
        self.system.contacts.JBt[self.contact_index][:] = value
    
    @property
    def C0(self):
        return self.system.contacts.C0[self.contact_index]
    
    @C0.setter
    def C0(self, value):
        self.system.contacts.C0[self.contact_index][:] = value
    
    # Manifold variables (scalars)
    @property
    def num_contact(self):
        return self.system.contacts.num_contact[self.contact_index]
    
    @num_contact.setter
    def num_contact(self, value):
        self.system.contacts.num_contact[self.contact_index] = value
    
    @property
    def friction(self):
        return self.system.contacts.friction[self.contact_index]
    
    @friction.setter
    def friction(self, value):
        self.system.contacts.friction[self.contact_index] = value