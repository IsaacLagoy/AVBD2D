# solver.py
import glm
from glm import vec2, vec3, mat3x3, mat3
from math import isinf, fabs
from maths import sign
from rigid import Rigid
from force import Force
from manifold import Manifold  # assumes Manifold(body_a, body_b) and .rows() implemented

PENALTY_MIN = 1e4
PENALTY_MAX = 1e9

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def diag3(a: float, b: float, c: float) -> mat3x3:
    # column-major constructor: (c0x, c0y, c0z, c1x, c1y, c1z, c2x, c2y, c2z)
    return mat3(a, 0.0, 0.0,
                0.0, b, 0.0,
                0.0, 0.0, c)

def solve_spd(lhs: mat3x3, rhs: vec3) -> vec3:
    # small 3x3; inversion is fine here
    return glm.inverse(lhs) * rhs

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
        # -------------------------
        # Broadphase (naive O(n^2))
        # -------------------------
        for i, A in enumerate(self.bodies):
            A = self.bodies[i]
            for B in self.bodies[i + 1:]:
                dp = (A.pos.xy - B.pos.xy)  # vec2
                r = A.radius + B.radius
                if glm.dot(dp, dp) <= r * r and not A.is_constrained_to(B):
                    self.forces.append(Manifold(A, B))

        # --------------------------------
        # Initialize & warmstart all forces
        # --------------------------------
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

        # ------------------------------------
        # Initialize & warmstart all body state
        # ------------------------------------
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

        # --------------------
        # Main solver iterations
        # --------------------
        for _ in range(self.iterations):
            # ---- Primal update ----
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
                delta = solve_spd(lhs, rhs)
                body.pos -= delta

            # ---- Dual update ----
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

        # ----------------
        # Velocity update (BDF1)
        # ----------------
        for body in self.bodies:
            body.prev_vel = body.vel
            if body.mass > 0:
                body.vel = (body.pos - body.initial) / dt
                
                
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