import glm
from glm import vec2, vec3, mat2x2
from math import sqrt
from force import Force
from rigid import Rigid
from maths import cross, transform, rotate_scale
from collide import collide
import pygame

# Constants (you’ll want to put these somewhere central)
COLLISION_MARGIN = 0.01
STICK_THRESH = 0.001
SHOW_CONTACTS = True

class Contact:
    def __init__(self):
        self.normal = vec2()
        self.rA = vec2()
        self.rB = vec2()
        self.stick = False

        # Jacobians
        self.JAn = vec3()
        self.JBn = vec3()
        self.JAt = vec3()
        self.JBt = vec3()

        self.C0 = vec2()


class Manifold(Force):
    def __init__(self, body_a: Rigid, body_b: Rigid) -> None:
        super().__init__(body_a, body_b)
        self.fmax[0] = self.fmax[2] = 0
        self.fmin[0] = self.fmin[2] = float('-inf')

        self.num_contacts = 0

        self.contacts = [Contact(), Contact()]
        self.friction = 0.0

    def initialize(self) -> bool:
        # compute friction
        self.friction = sqrt(self.body_a.friction * self.body_b.friction)

        # store previous contacts
        old_contacts = [self.contacts[0], self.contacts[1]]
        old_penalty = self.penalty[:]
        old_lambda = self.lamb[:]
        old_stick = [c.stick for c in old_contacts]
        old_num = self.num_contacts

        # recompute contacts
        self.num_contacts = collide(self.body_a, self.body_b, self.contacts)

        # merge old state into new contacts
        for i in range(self.num_contacts):
            
            self.penalty[i * 2 + 0] = self.penalty[i * 2 + 1] = 0
            self.lamb[i * 2 + 0] = self.lamb[i * 2 + 1] = 0

            for j in range(old_num):
                if False:
                    self.penalty[i * 2 + 0] = old_penalty[j * 2 + 0]
                    self.penalty[i * 2 + 1] = old_penalty[j * 2 + 1]
                    self.lamb[i * 2 + 0] = old_lambda[j * 2 + 0]
                    self.lamb[i * 2 + 1] = old_lambda[j * 2 + 1]
                    self.contacts[i].stick = old_stick[j]

                    if old_stick[j]:
                        self.contacts[i].rA = old_contacts[j].rA
                        self.contacts[i].rB = old_contacts[j].rB

        # precompute constraint Jacobians
        for i in range(self.num_contacts):
            n = self.contacts[i].normal
            t = vec2(n.y, -n.x)
            basis = mat2x2(
                n.x, n.y, 
                t.x, t.y
            )

            rAW = rotate_scale(self.body_a.pos.z, self.body_a.scale, self.contacts[i].rA)
            rBW = rotate_scale(self.body_b.pos.z, self.body_b.scale, self.contacts[i].rB)

            self.contacts[i].JAn = vec3(n.x, n.y, cross(rAW, n))
            self.contacts[i].JBn = vec3(-n.x, -n.y, -cross(rBW, n))
            self.contacts[i].JAt = vec3(t.x, t.y, cross(rAW, t))
            self.contacts[i].JBt = vec3(-t.x, -t.y, -cross(rBW, t))

            self.contacts[i].C0 = basis * (
                (self.body_a.pos.xy + rAW)
                - (self.body_b.pos.xy + rBW)
            ) + vec2(COLLISION_MARGIN, 0)

        return self.num_contacts > 0

    def computeConstraint(self, alpha: float) -> None:
        for i in range(self.num_contacts):
            dpA = self.body_a.pos - self.body_a.initial
            dpB = self.body_b.pos - self.body_b.initial

            self.C[i * 2 + 0] = (
                self.contacts[i].C0.x * (1 - alpha)
                + glm.dot(self.contacts[i].JAn, dpA)
                + glm.dot(self.contacts[i].JBn, dpB)
            )
            self.C[i * 2 + 1] = (
                self.contacts[i].C0.y * (1 - alpha)
                + glm.dot(self.contacts[i].JAt, dpA)
                + glm.dot(self.contacts[i].JBt, dpB)
            )

            friction_bound = abs(self.lamb[i * 2 + 0]) * self.friction
            self.fmax[i * 2 + 1] = friction_bound
            self.fmin[i * 2 + 1] = -friction_bound

            self.contacts[i].stick = (
                abs(self.lamb[i * 2 + 1]) < friction_bound
                and abs(self.contacts[i].C0.y) < STICK_THRESH
            )

    def computeDerivatives(self, body: Rigid) -> None:
        for i in range(self.num_contacts):
            if body == self.body_a:
                self.J[i * 2 + 0] = self.contacts[i].JAn
                self.J[i * 2 + 1] = self.contacts[i].JAt
            else:
                self.J[i * 2 + 0] = self.contacts[i].JBn
                self.J[i * 2 + 1] = self.contacts[i].JBt
                
    def rows(self) -> int:
        return self.num_contacts * 2

    def draw(self, screen, scale_factor=20, offset=(400, 300)) -> None:
        if not SHOW_CONTACTS:
            return

        normal_length = 30  # pixels for drawing the normal

        for i in range(self.num_contacts):
            # Contact points in world space
            world_a = transform(self.body_a.pos, self.body_a.scale, self.contacts[i].rA)
            world_b = transform(self.body_b.pos, self.body_b.scale, self.contacts[i].rB)

            # Convert to screen space
            sx_a = int(world_a.x * scale_factor + offset[0])
            sy_a = int(-world_a.y * scale_factor + offset[1])  # invert y for pygame

            sx_b = int(world_b.x * scale_factor + offset[0])
            sy_b = int(-world_b.y * scale_factor + offset[1])

            # Draw as small circles
            pygame.draw.circle(screen, (255, 0, 0), (sx_a, sy_a), 4)
            pygame.draw.circle(screen, (0, 0, 255), (sx_b, sy_b), 4)

            # Draw contact normal (from A's contact point)
            n = self.contacts[i].normal
            nx, ny = n.x, n.y

            end_x = int(sx_a + nx * normal_length)
            end_y = int(sy_a - ny * normal_length)  # minus because y is inverted

            pygame.draw.line(screen, (0, 200, 0), (sx_a, sy_a), (end_x, end_y), 2)

