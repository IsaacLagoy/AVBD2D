# solver.py
from math import isinf
from shapes.rigid import Rigid
from forces.force import Force
from forces.manifold import Manifold
import time
from linalg.ldlt import solve, solve_glm
from helper.constants import DEBUG_TIMING, PENALTY_MAX, PENALTY_MIN
from helper.decorators import timer
from shapes.body_system import BodySystem
from forces.force_system import ForceSystem
import numpy as np
import numba as nb
from graph.dsatur import dsatur_coloring
from forces.manifold import Manifold
from shapes.mesh_system import MeshSystem
import math


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def diag3(a: float, b: float, c: float) -> np.ndarray:
    return np.diag([a, b, c]).astype(np.float32)

class Solver:
    def __init__(self, screen) -> None:
        self.screen = screen
        
        # physics params
        self.iterations = 10
        self.beta = 1e5
        self.alpha = 0.99
        self.gamma = 0.99

        self.gravity = np.array([0, -10], dtype='float32')

        # scene - linked list heads
        self.bodies: Rigid = None
        self.forces: Force = None
        
        # systems
        self.body_system = BodySystem(self, 128)
        self.force_system = ForceSystem(self, 1024)
        self.mesh_system = MeshSystem(self, 1024)
        
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
        self.collide()
        self.warmstart_forces()
        self.warmstart_bodies(dt)
        
        start_color = time.perf_counter()
        # color rigid bodies
        self.colors = dsatur_coloring(self)

        for color in self.colors:
            color.reserve_space()
            
        print(f'Graph Coloring: {(time.perf_counter() - start_color) * 1000:.3f}ms')
            
        inv_dt2 = 1 / (dt * dt)
        
        # main solver loop
        for iteration in range(self.iterations):
            
            start_primal = time.perf_counter()
            
            # primal update
            for color in self.colors:
                if color.count == 0:
                    continue
                
                # LHS/RHS (Eqs. 5,6)
                color.lhs[:, 0, 0] = self.body_system.mass[color.indices]
                color.lhs[:, 1, 1] = self.body_system.mass[color.indices]  
                color.lhs[:, 2, 2] = self.body_system.moment[color.indices]
                color.lhs *= inv_dt2

                color.rhs = np.einsum('bij,bj->bi', color.lhs, self.body_system.pos[color.indices] - self.body_system.inertial[color.indices])
        
                # TODO primal update each body in the color
                
            print(f'Primal Update: {(time.perf_counter() - start_primal) * 1000:.3f}ms')
                
            # --------------------
            # Dual update
            # --------------------
                
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
            
            start_dual = time.perf_counter()
            
            print(f'Dual Update: {(time.perf_counter() - start_dual) * 1000:.3f}ms')

        self.update_velocities(dt)
            
        total_time = time.perf_counter() - start_time
        if DEBUG_TIMING:
            print(f"TOTAL STEP TIME: {total_time*1000:.3f}ms")
            print("-" * 50)
            
    # --------------------
    # BroadPhase
    # --------------------
    
    # TODO replace this with Dynamic BVH TODO iterate since bodies are compact
    @timer('Broadphase', on=DEBUG_TIMING)
    def spherical_broad_collision(self) -> None:
        # clear collision pairs
        self.force_system.pairs = []
        
        # compact so that indexing is more cache friendly
        self.body_system.compact()
        
        # compute all radii sums
        N = self.body_system.size
        
        # NOTE this looped part may be better when njit
        
        # table = np.empty((N, N), dtype=np.float32)

        # for i in range(N):
        #     temp = self.body_system.pos[i, :2] - self.body_system.pos[:N, :2]  # shape (N, 2)
        #     dist2 = np.sum(temp**2, axis=1)  # squared distances
        #     rsum2 = (self.body_system.radius[i] + self.body_system.radius[:N])**2
        #     table[i, :] = rsum2 - dist2
        
        pos = self.body_system.pos[:N, :2]  # shape (N,2)
        radii = self.body_system.radius[:N]  # shape (N,)

        # pairwise squared distances using broadcasting
        diff = pos[:, None, :] - pos[None, :, :]  # shape (N, N, 2)
        dist2 = np.sum(diff**2, axis=2)           # shape (N, N)
        rsum2 = (radii[:, None] + radii[None, :])**2

        table = rsum2 - dist2

        overlaps = np.argwhere(table > 0)
        for i, j in overlaps:
            if i < j:  # optional: only handle each pair once
                # count += 1
                self.force_system.pairs.append((i, j))
                
        # convert pairs to numpy array
        self.force_system.pairs = np.array(self.force_system.pairs)
                
        keeps = np.ones(len(self.force_system.pairs), dtype=np.bool_)
        for index, (i, j) in enumerate(self.force_system.pairs):
            A: Rigid = self.body_system.bodies[i]
            B: Rigid = self.body_system.bodies[j]
            
            if not A.is_constrained_to(B):
                continue
            
            # if it is constrained, remove from mask
            keeps[index] = 0
            
        self.force_system.pairs = self.force_system.pairs[keeps]
        
    # --------------------
    # Narrow Collision
    # --------------------
    
    @timer('Narrow Collision', on=DEBUG_TIMING)
    def collide(self) -> None:
        # compute transformation matrices of moved objects
        self.compute_body_transforms(
            self.body_system.pos,
            self.body_system.scale, 
            self.body_system.updated,
            self.body_system.s_ir,
            self.body_system.irs,
            self.body_system.size
        )
        
        # compute collisions between objects
        self.force_system.collide()
        
    # NOTE threading overhead is too expensive for parallelization
    @staticmethod
    @nb.njit(fastmath=True)
    def compute_body_transforms(pos, scale, updated, s_ir, irs, size):
        for i in range(size):
            if updated[i]:  # Only process non-updated bodies
                continue
            
            angle = pos[i, 2]
            sx = scale[i, 0]
            sy = scale[i, 1]
            
            # Compute trigonometric values
            c = math.cos(angle)
            s = math.sin(angle)
            
            # s_ir = scale @ inv(rotation)
            s_ir[i, 0, 0] = sx * c
            s_ir[i, 0, 1] = sx * s
            s_ir[i, 1, 0] = -sy * s
            s_ir[i, 1, 1] = sy * c
            
            # irs = inv(rotation @ scale)
            irs[i, 0, 0] = c * sx
            irs[i, 0, 1] = -s * sy
            irs[i, 1, 0] = s * sx
            irs[i, 1, 1] = c * sy
                    
    # ------------------------------------
    # Force Warmstart
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
        
        print(len(self.force_system.forces))

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
            
        gravity = self.gravity[1]
        
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
        inv_dt = 1 / dt
        # Get active slice of arrays
        active_slice = slice(0, self.body_system.size)
        self.body_system.prev_vel[active_slice] = self.body_system.vel[active_slice]
        
        mass_slice = self.body_system.mass[active_slice] > 0
        
        self.body_system.vel[active_slice][mass_slice] = inv_dt * (self.body_system.pos[active_slice][mass_slice] - self.body_system.initial[active_slice][mass_slice])
                
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
            
            if isinstance(value, Manifold):
                self.force_system.contacts.delete(value.contact_index)
                
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