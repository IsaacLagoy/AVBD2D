# solver.py
import glm
from glm import vec2, mat3x3, mat3
from math import isinf
from shapes.rigid import Rigid
from forces.force import Force
from forces.manifold import Manifold
import time
from linalg.ldlt import solve
from helper.constants import DEBUG_TIMING, PENALTY_MAX, PENALTY_MIN
from helper.decorators import timer
from shapes.body_system import BodySystem
from forces.force_system import ForceSystem
import numpy as np
import numba as nb
from graph.dsatur import dsatur_coloring


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def diag3(a: float, b: float, c: float) -> mat3x3:
    return mat3(a, 0.0, 0.0,
                0.0, b, 0.0,
                0.0, 0.0, c)

class Solver:
    def __init__(self) -> None:
        # physics params
        self.iterations = 10
        self.beta = 1e5
        self.alpha = 0.99
        self.gamma = 0.99

        self.gravity: vec2 = vec2(0.0, -10.0)

        # scene - linked list heads
        self.bodies: Rigid = None
        self.forces: Force = None
        
        # systems
        self.body_system = BodySystem(self, 128)
        self.force_system = ForceSystem(self, 1024)
        
        # parallelization variables
        self.colors = []

    def get_bodies_iterator(self):
        """
        Iterator for bodies linked list
        """
        current = self.bodies
        while current is not None:
            yield current # poggers yield moment
            current = current.next
            
    def get_forces_iterator(self):
        """
        Iterator for forces linked list
        """
        current = self.forces
        while current is not None:
            yield current
            current = current.next

    def step(self, dt: float) -> None:
        start_time = time.perf_counter()
        
        self.spherical_broad_collision()
        self.warmstart_forces()
        self.warmstart_bodies(dt)
        
        # color rigid bodies
        self.colors = dsatur_coloring(self)

        for color in self.colors:
            color.reserve_space()
            
        inv_dt2 = 1 / (dt * dt)
        
        # main solver loop
        for iteration in range(self.iterations):
            
            # primal update
            for color in self.colors:
                # build lhs
                color.lhs[:, 0, 0] = self.body_system.mass[color.indices]
                color.lhs[:, 1, 1] = self.body_system.mass[color.indices]  
                color.lhs[:, 2, 2] = self.body_system.moment[color.indices]
                
                color.lhs *= inv_dt2
                
                # build rhs
                color.rhs = np.einsum('bij,bj->bi', color.lhs, self.body_system.pos[color.indices] - self.body_system.inertial[color.indices])
                
                # find forces
                for i, body in enumerate(color.get_bodies_iterator()):
                    body_idx_in_color = i
                    
                    for force in body.get_forces_iterator():

                        # compute constraint & derivatives
                        force.computeConstraint(self.alpha)
                        force.computeDerivatives(body)
                        
                        # Vectorized accumulation for this force's contribution
                        for r in range(force.rows()):
                            lam = force.lamb[r] if isinf(force.stiffness[r]) else 0
                            f = clamp(force.penalty[r] * force.C[r] + lam + force.motor[r],
                                    force.fmin[r], force.fmax[r])
                            
                            # Geometric stiffness
                            Hc0, Hc1, Hc2 = force.H[r][0], force.H[r][1], force.H[r][2]
                            G_diag = [glm.length(Hc0), glm.length(Hc1), glm.length(Hc2)]
                            G_scaled = [g * abs(f) for g in G_diag]
                            
                            # Accumulate into vectorized arrays
                            J = force.J[r]  # vec3
                            color.rhs[body_idx_in_color] += J * f
                            
                            # Add outer product and geometric stiffness
                            for j in range(3):
                                for k in range(3):
                                    color.lhs[body_idx_in_color, j, k] += J[j] * J[k] * force.penalty[r]
                                color.lhs[body_idx_in_color, j, j] += G_scaled[j]
                        
                self.body_system.pos[color.indices] -= solve(color.lhs, color.rhs)
                
                # dual update
                for force in self.get_forces_iterator():
                    force.computeConstraint(self.alpha)
                
                # Eq. 11 (do not include motors in dual update)
                self.force_system.lamb = np.clip(self.force_system.penalty * self.force_system.C + np.where(np.isinf(self.force_system.stiffness), self.force_system.lamb, 0.0), self.force_system.fmin, self.force_system.fmax)
                
                # fracture
                mask = np.any(np.abs(self.force_system.lamb) >= self.force_system.fracture, axis=1)
                self.force_system.stiffness[mask, :] = 0
                self.force_system.penalty[mask, :]   = 0
                self.force_system.lamb[mask, :]      = 0

                # Eq. 16 — increment penalty within bounds if within force limits
                mask = (self.force_system.lamb > self.force_system.fmin) & (self.force_system.lamb < self.force_system.fmax)
                self.force_system.penalty = np.where(
                    mask,
                    np.minimum(self.force_system.penalty + self.beta * np.abs(self.force_system.C),
                            np.minimum(PENALTY_MAX, self.force_system.stiffness)),
                    self.force_system.penalty
                )


        self.update_velocities(dt)
            
        total_time = time.perf_counter() - start_time
        if DEBUG_TIMING:
            print(f"TOTAL STEP TIME: {total_time*1000:.3f}ms")
            print("-" * 50)
            
    # --------------------
    # BroadPhase
    # --------------------
    
    @timer('Broadphase', on=DEBUG_TIMING)
    def spherical_broad_collision(self) -> None:
        # Convert linked list to list for easier iteration
        bodies_list = list(self.get_bodies_iterator())
        
        for i, A in enumerate(bodies_list):
            for B in bodies_list[i + 1:]:
                dp = (A.pos.xy - B.pos.xy)  # vec2
                r = A.radius + B.radius
                if glm.dot(dp, dp) <= r * r and not A.is_constrained_to(B):
                    # Create new manifold force - it will add itself to the linked lists
                    Manifold(self.force_system, A, B)
                    
    # ------------------------------------
    # Force Warmstart and Narrow Collision
    # ------------------------------------
            
    @timer('Warmstart Forces', on=DEBUG_TIMING)        
    def warmstart_forces(self) -> None:
        current_force = self.forces
        while current_force is not None:
            next_force = current_force.next  # Save next before potential removal

            if not current_force.initialize():
                # force inactive this step — remove it
                self.remove(current_force)
            
            current_force = next_force
            
        self.force_system.compact()

        # warmstart forces
        active_slice = slice(0, self.force_system.size)
        self.force_system.lamb[active_slice] *= self.alpha * self.gamma
        self.force_system.penalty[active_slice] = np.clip(self.force_system.penalty[active_slice] * self.gamma, PENALTY_MIN, PENALTY_MAX)
        self.force_system.penalty[active_slice] = np.minimum(self.force_system.penalty[active_slice], self.force_system.stiffness[active_slice])
            
    # ------------------------------------
    # Initialize & warmstart all body state
    # ------------------------------------

    @timer('Warmstart Bodies', on=DEBUG_TIMING)  
    def warmstart_bodies(self, dt: float) -> None:
        if self.body_system.size == 0:
            return
            
        gravity = self.gravity.y
        
        # Get active slice of arrays
        active_slice = slice(0, self.body_system.size)
        
        # Limit angular velocity
        self.body_system.vel[active_slice, 2] = np.clip(
            self.body_system.vel[active_slice, 2], -10.0, 10.0
        )
        
        # Inertial (Eq. 2)
        self.body_system.inertial[active_slice] = (
            self.body_system.pos[active_slice] + 
            self.body_system.vel[active_slice] * dt
        )
        
        # Add gravity for bodies with mass > 0
        mass_mask = self.body_system.mass[active_slice] > 0
        gravity_vec = np.array([0, gravity, 0], dtype='float32')
        self.body_system.inertial[active_slice][mass_mask] += gravity_vec * (dt * dt)
        
        # Adaptive warmstart (original VBD)
        accel = (self.body_system.vel[active_slice] - self.body_system.prev_vel[active_slice]) / dt
        accel_ext = accel[:, 1] * np.sign(gravity)  # y component * sign of gravity
        accel_weight = np.clip(accel_ext / abs(gravity), 0, 1)
        
        # Handle infinite values
        accel_weight = np.where(np.isinf(accel_weight), 0, accel_weight)
        
        # Save initial positions
        self.body_system.initial[active_slice] = self.body_system.pos[active_slice].copy()
        
        # Warmstart position
        gravity_term = gravity_vec * (accel_weight[:, np.newaxis] * dt * dt)
        self.body_system.pos[active_slice] = (
            self.body_system.pos[active_slice] + 
            self.body_system.vel[active_slice] * dt + 
            gravity_term
        )
            
    # ----------------
    # Velocity update (BDF1)
    # ----------------
    
    def update_velocities(self, dt: float) -> None:
        current_body = self.bodies
        while current_body is not None:
            current_body.prev_vel = current_body.vel
            if current_body.mass > 0:
                current_body.vel = (current_body.pos - current_body.initial) / dt
            current_body = current_body.next
                
    # --------------------
    # Subobject Management
    # --------------------
                
    def remove(self, value):
        if isinstance(value, Force):
            # Remove from solver's force linked list
            if self.forces is value:
                self.forces = value.next
            else:
                prev = None
                current = self.forces
                while current is not None and current is not value:
                    prev = current
                    current = current.next
                if current is value and prev is not None:
                    prev.next = value.next
            
            # Remove from both bodies' force lists
            body_a = value.body_a
            body_b = value.body_b
            
            if body_a:
                body_a.remove_force(value)
            if body_b and body_b is not body_a:
                body_b.remove_force(value)
                
            self.force_system.delete(value.index)
                
        elif isinstance(value, Rigid):
            # Remove from solver's body linked list
            if self.bodies is value:
                self.bodies = value.next
            else:
                prev = None
                current = self.bodies
                while current is not None and current is not value:
                    prev = current
                    current = current.next
                if current is value and prev is not None:
                    prev.next = value.next
                    
            self.body_system.delete(value.index)