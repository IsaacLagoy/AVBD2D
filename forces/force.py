from glm import vec3, mat3x3
from shapes.rigid import Rigid
from helper.constants import ROWS


class Force:
    def __init__(self, solver, body_a: Rigid, body_b: Rigid) -> None:
        # add self to solver linked list
        self.solver = solver
        self.next = solver.forces
        solver.forces = self  # Fixed: was setting self.solver = self
        
        self.body_a = body_a
        self.body_b = body_b
        
        self.next_a = None
        self.next_b = None
        
        # Link to body_a's force list
        if body_a:
            self.next_a = body_a.forces
            body_a.forces = self
            
        # Link to body_b's force list  
        if body_b:
            self.next_b = body_b.forces
            body_b.forces = self
            
        # increment graph degree tracker only if force is an edge
        if body_a and body_b:
            body_a.degree += 1
            body_b.degree += 1
        
        # initialize variables
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
        
    # remove self from solver linked list
    def remove_self(self) -> None:
        # --- Remove from solver list ---
        if self.solver.forces is self:
            self.solver.forces = self.next
        else:
            prev = None
            current = self.solver.forces
            while current is not None and current is not self:
                prev = current
                current = current.next
            if current is self and prev is not None:
                prev.next = self.next

        # --- Remove from both bodies ---
        if self.body_a:
            self.body_a.remove_force(self)
        if self.body_b and self.body_b is not self.body_a:
            self.body_b.remove_force(self)
            
        # reduce graph degree tracker
        if self.body_a and self.body_b:
            self.body_a.degree -= 1
            self.body_b.degree -= 1

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