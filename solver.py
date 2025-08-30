# solver.py
import glm
from glm import vec2, vec3, mat3x3, mat3
from math import isinf
from helper.maths import sign
from shapes.rigid import Rigid
from forces.force import Force
from forces.manifold import Manifold
import time
from linalg.ldlt import solve
from helper.constants import DEBUG_TIMING, PENALTY_MAX, PENALTY_MIN
from helper.decorators import timer


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

        # --------------------
        # Main solver iterations
        # --------------------
        solver_start = time.perf_counter()
        for iteration in range(self.iterations):
            iteration_start = time.perf_counter()
            
            # ---- Primal update ----
            primal_start = time.perf_counter()
            current_body = self.bodies
            while current_body is not None:
                if current_body.mass <= 0:
                    current_body = current_body.next
                    continue

                # LHS/RHS (Eqs. 5,6)
                M = diag3(current_body.mass, current_body.mass, current_body.moment)
                inv_dt2 = 1.0 / (dt * dt)
                lhs = M * inv_dt2
                rhs = (M * inv_dt2) * (current_body.pos - current_body.inertial)

                # iterate forces acting on this body
                for force in current_body.get_forces_iterator():

                    # compute constraint & derivatives
                    force.computeConstraint(self.alpha)
                    force.computeDerivatives(current_body)

                    for r in range(force.rows()):
                        # lambda=0 if not a hard constraint
                        lam = force.lamb[r] if isinf(force.stiffness[r]) else 0

                        # clamped force magnitude (Sec 3.2)
                        f = clamp(force.penalty[r] * force.C[r] + lam + force.motor[r],
                                     force.fmin[r], force.fmax[r])

                        # diagonally lumped geometric stiffness G (Sec 3.5)
                        # PyGLM mat3 is column-major; mat[0], mat[1], mat[2] are vec3 columns
                        Hc0 = force.H[r][0] # vec3
                        Hc1 = force.H[r][1]
                        Hc2 = force.H[r][2]
                        G = diag3(glm.length(Hc0), glm.length(Hc1), glm.length(Hc2)) * abs(f)

                        # accumulate (Eq. 13,17)
                        J = force.J[r]
                        rhs += J * f
                        lhs += glm.outerProduct(J, J * force.penalty[r]) + G

                # Solve SPD system and apply update (Eq. 4)
                delta = solve(lhs, rhs)
                current_body.pos -= delta
                
                current_body = current_body.next
            
            primal_time = time.perf_counter() - primal_start

            # ---- Dual update ----
            dual_start = time.perf_counter()
            current_force = self.forces
            while current_force is not None:

                current_force.computeConstraint(self.alpha)

                for r in range(current_force.rows()):
                    lam = current_force.lamb[r] if isinf(current_force.stiffness[r]) else 0

                    # Eq. 11 (do not include motors in dual update)
                    current_force.lamb[r] = clamp(current_force.penalty[r] * current_force.C[r] + lam,
                                          current_force.fmin[r], current_force.fmax[r])

                    # fracture
                    if abs(current_force.lamb[r]) >= current_force.fracture[r]:
                        current_force.disable()

                    # Eq. 16 — increment penalty within bounds if within force limits
                    if current_force.lamb[r] > current_force.fmin[r] and current_force.lamb[r] < current_force.fmax[r]:
                        current_force.penalty[r] = min(
                            current_force.penalty[r] + self.beta * abs(current_force.C[r]),
                            min(PENALTY_MAX, current_force.stiffness[r])
                        )
                
                current_force = current_force.next
            
            dual_time = time.perf_counter() - dual_start
            iteration_time = time.perf_counter() - iteration_start
            
            if DEBUG_TIMING:
                print(f"  Iteration {iteration+1}: Primal {primal_time*1000:.3f}ms, Dual {dual_time*1000:.3f}ms, Total {iteration_time*1000:.3f}ms")
        
        solver_time = time.perf_counter() - solver_start
        if DEBUG_TIMING:
            print(f"Main solver iterations ({self.iterations} iterations): {solver_time*1000:.3f}ms")


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
                    Manifold(self, A, B)
                    
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
                continue

            # warmstart duals and penalties (Eq. 19)
            for k in range(current_force.rows()):
                current_force.lamb[k] *= self.alpha * self.gamma
                current_force.penalty[k] = clamp(current_force.penalty[k] * self.gamma, PENALTY_MIN, PENALTY_MAX)
                # clamp by material stiffness for non-hard constraints
                current_force.penalty[k] = min(current_force.penalty[k], current_force.stiffness[k])
            
            current_force = next_force
            
    # ------------------------------------
    # Initialize & warmstart all body state
    # ------------------------------------

    @timer('Warmstart Bodies', on=DEBUG_TIMING)     
    def warmstart_bodies(self, dt: float) -> None:
        gravity = self.gravity.y

        current_body = self.bodies
        while current_body is not None:
            # limit angular velocity
            current_body.vel.z = clamp(current_body.vel.z, -10.0, 10.0)

            # inertial (Eq. 2)
            current_body.inertial = current_body.pos + current_body.vel * dt
            if current_body.mass > 0:
                current_body.inertial += vec3(0, gravity, 0) * (dt * dt)

            # adaptive warmstart (original VBD)
            accel = (current_body.vel - current_body.prev_vel) / dt  # vec3
            accel_ext = accel.y * sign(gravity)
            accel_weight = glm.clamp(accel_ext / abs(gravity), 0, 1)
            if isinf(accel_weight): accel_weight = 0

            # save x- and warmstart position
            current_body.initial = current_body.pos
            current_body.pos = current_body.pos + current_body.vel * dt + vec3(0, gravity, 0) * (accel_weight * dt * dt)
            
            current_body = current_body.next
            
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