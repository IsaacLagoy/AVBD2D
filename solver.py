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
        self.beta = 1e5      # closer to C++ defaultParams() comment; tune as needed
        self.alpha = 0.99
        self.gamma = 0.99

        self.gravity: vec2 = vec2(0.0, -10.0)

        # scene
        self.bodies: list[Rigid] = []
        self.forces: list[Force] = []

    def step(self, dt: float) -> None:
        start_time = time.perf_counter()
        
        # -------------------------
        # Broadphase (naive O(n^2))
        # -------------------------
        broadphase_start = time.perf_counter()
        for i, A in enumerate(self.bodies):
            A = self.bodies[i]
            for B in self.bodies[i + 1:]:
                dp = (A.pos.xy - B.pos.xy)  # vec2
                r = A.radius + B.radius
                if glm.dot(dp, dp) <= r * r and not A.is_constrained_to(B):
                    self.forces.append(Manifold(A, B))
        
        broadphase_time = time.perf_counter() - broadphase_start
        if DEBUG_TIMING:
            print(f"Broadphase: {broadphase_time*1000:.3f}ms")

        # --------------------------------
        # Initialize & warmstart all forces
        # --------------------------------
        force_init_start = time.perf_counter()
        i = 0
        while i < len(self.forces):
            force = self.forces[i]

            if not force.initialize():
                # force inactive this step — remove it
                self.remove(self.forces[i])
                continue

            # warmstart duals and penalties (Eq. 19)
            for k in range(force.rows()):
                force.lamb[k] *= self.alpha * self.gamma
                force.penalty[k] = clamp(force.penalty[k] * self.gamma, PENALTY_MIN, PENALTY_MAX)
                # clamp by material stiffness for non-hard constraints
                force.penalty[k] = min(force.penalty[k], force.stiffness[k])
            i += 1
        
        force_init_time = time.perf_counter() - force_init_start
        if DEBUG_TIMING:
            print(f"Force initialization & warmstart: {force_init_time*1000:.3f}ms")

        # ------------------------------------
        # Initialize & warmstart all body state
        # ------------------------------------
        body_init_start = time.perf_counter()
        gravity = self.gravity.y

        for body in self.bodies:
            # limit angular velocity
            body.vel.z = clamp(body.vel.z, -10.0, 10.0)

            # inertial (Eq. 2)
            body.inertial = body.pos + body.vel * dt
            if body.mass > 0:
                body.inertial += vec3(0, gravity, 0) * (dt * dt)

            # adaptive warmstart (original VBD)
            accel = (body.vel - body.prev_vel) / dt  # vec3
            accel_ext = accel.y * sign(gravity)
            accel_weight = glm.clamp(accel_ext / abs(gravity), 0, 1)
            if isinf(accel_weight): accel_weight = 0

            # save x- and warmstart position
            body.initial = body.pos
            body.pos = body.pos + body.vel * dt + vec3(0, gravity, 0) * (accel_weight * dt * dt)
        
        body_init_time = time.perf_counter() - body_init_start
        if DEBUG_TIMING:
            print(f"Body initialization & warmstart: {body_init_time*1000:.3f}ms")

        # --------------------
        # Main solver iterations
        # --------------------
        solver_start = time.perf_counter()
        for iteration in range(self.iterations):
            iteration_start = time.perf_counter()
            
            # ---- Primal update ----
            primal_start = time.perf_counter()
            for body in self.bodies:
                if body.mass <= 0:
                    continue

                # LHS/RHS (Eqs. 5,6)
                M = diag3(body.mass, body.mass, body.moment)
                inv_dt2 = 1.0 / (dt * dt)
                lhs = M * inv_dt2
                rhs = (M * inv_dt2) * (body.pos - body.inertial)

                # iterate forces acting on this body
                for force in body.forces:

                    # compute constraint & derivatives
                    force.computeConstraint(self.alpha)
                    force.computeDerivatives(body)

                    for r in range(force.rows()):
                        # lambda=0 if not a hard constraint
                        lam = force.lamb[r] if isinf(force.stiffness[r]) else 0

                        # clamped force magnitude (Sec 3.2)
                        f = clamp(force.penalty[r] * force.C[r] + lam + force.motor[r],
                                     force.fmin[r], force.fmax[r])

                        # diagonally lumped geometric stiffness G (Sec 3.5)
                        # PyGLM mat3 is column-major; mat[0], mat[1], mat[2] are vec3 columns
                        Hc0 = force.H[r][0]  # vec3
                        Hc1 = force.H[r][1]
                        Hc2 = force.H[r][2]
                        G = diag3(glm.length(Hc0), glm.length(Hc1), glm.length(Hc2)) * abs(f)

                        # accumulate (Eq. 13,17)
                        J = force.J[r]  # vec3
                        rhs += J * f
                        lhs += glm.outerProduct(J, J * force.penalty[r]) # + G

                # Solve SPD system and apply update (Eq. 4)
                delta = solve(lhs, rhs)
                body.pos -= delta
            
            primal_time = time.perf_counter() - primal_start

            # ---- Dual update ----
            dual_start = time.perf_counter()
            for force in self.forces:

                force.computeConstraint(self.alpha)

                for r in range(force.rows()):
                    lam = force.lamb[r] if isinf(force.stiffness[r]) else 0

                    # Eq. 11 (do not include motors in dual update)
                    force.lamb[r] = clamp(force.penalty[r] * force.C[r] + lam,
                                          force.fmin[r], force.fmax[r])

                    # fracture
                    if abs(force.lamb[r]) >= force.fracture[r]:
                        force.disable()

                    # Eq. 16 — increment penalty within bounds if within force limits
                    if force.lamb[r] > force.fmin[r] and force.lamb[r] < force.fmax[r]:
                        force.penalty[r] = min(
                            force.penalty[r] + self.beta * abs(force.C[r]),
                            min(PENALTY_MAX, force.stiffness[r])
                        )
            
            dual_time = time.perf_counter() - dual_start
            iteration_time = time.perf_counter() - iteration_start
            
            if DEBUG_TIMING:
                print(f"  Iteration {iteration+1}: Primal {primal_time*1000:.3f}ms, Dual {dual_time*1000:.3f}ms, Total {iteration_time*1000:.3f}ms")
        
        solver_time = time.perf_counter() - solver_start
        if DEBUG_TIMING:
            print(f"Main solver iterations ({self.iterations} iterations): {solver_time*1000:.3f}ms")

        # ----------------
        # Velocity update (BDF1)
        # ----------------
        velocity_start = time.perf_counter()
        for body in self.bodies:
            body.prev_vel = body.vel
            if body.mass > 0:
                body.vel = (body.pos - body.initial) / dt
        
        velocity_time = time.perf_counter() - velocity_start
        if DEBUG_TIMING:
            print(f"Velocity update: {velocity_time*1000:.3f}ms")
            
        total_time = time.perf_counter() - start_time
        if DEBUG_TIMING:
            print(f"TOTAL STEP TIME: {total_time*1000:.3f}ms")
            print("-" * 50)
                
                
    def remove(self, value):
        if isinstance(value, Force):
            if value not in self.forces:
                return
            
            self.forces.remove(value)
    
            body_a = value.body_a
            body_b = value.body_b
            
            if body_a and value in body_a.forces:
                body_a.forces.remove(value)
            if body_b and value in body_b.forces:
                body_b.forces.remove(value)