# force.py
import glm
from glm import vec2, vec3, mat3x3
from rigid import Rigid

ROWS = 4

class Force:
    def __init__(self, body_a: Rigid, body_b: Rigid) -> None:
        self.J = [vec3() for _ in range(ROWS)]
        self.H = [mat3x3() for _ in range(ROWS)]
        self.C = [0 for _ in range(ROWS)]
        self.motor = [0 for _ in range(ROWS)]
        self.stiffness = [float('inf') for _ in range(ROWS)]
        self.fmax = [float('inf') for _ in range(ROWS)]
        self.fmin = [float('-inf') for _ in range(ROWS)]
        self.fracture = [float('inf') for _ in range(ROWS)]

        self.penalty = [0 for _ in range(ROWS)]
        self.lamb = [0 for _ in range(ROWS)]

        self.body_a = body_a
        self.body_b = body_b

        # link to bodies (mirror the C++ intrusive list semantics)
        self.body_a.forces.append(self)
        self.body_b.forces.append(self)

    def rows(self) -> int: return 0
    def initialize(self) -> bool: return False
    def computeConstraint(self, alpha: float) -> None: ...
    def computeDerivatives(self, rigid: Rigid) -> None: ...

    def disable(self) -> None:
        self.stiffness = [0 for _ in range(ROWS)]
        self.penalty = [0 for _ in range(ROWS)]
        self.lamb = [0 for _ in range(ROWS)]
            
    # prints debug information about the force
    def __repr__(self) -> str:
        return f"""
Force: {id(self)}
Jacobians:
{self.J[0]}
{self.J[1]}
{self.J[2]}
{self.J[3]}
{self.penalty}
{self.lamb}
"""
