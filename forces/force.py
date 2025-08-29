from glm import vec3, mat3x3
from shapes.rigid import Rigid
from helper.constants import ROWS


class Force:
    def __init__(self, solver, body_a: Rigid, body_b: Rigid) -> None:
        # add self to solver linked list and body linked lists
        self.solver = solver
        self.next = solver.forces
        self.solver = self
        
        self.body_a = body_a
        self.body_b = body_b
        
        self.next_a = None
        self.next_b = None
        
        if body_a:
            self.next_a = body_a.forces
            body_a.forces = self
            
        if body_b:
            self.next_b = body_b.forces
            body_b.forces = self
        
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

        # link to bodies (mirror the C++ intrusive list semantics)
        self.body_a.forces.append(self)
        self.body_b.forces.append(self)
        
    # remove self from solver linked list
    def remove_self(self) -> None:
        # --- Remove from solver list ---
        node = self.solver.forces
        if node is self:
            self.solver.forces = self.next
        else:
            prev = node
            node = node.next
            while node is not None and node is not self:
                prev = node
                node = node.next
            if node is self:
                prev.next = self.next

        # --- Remove from both bodies ---
        if self.bodyA:
            self.bodyA.remove_force(self)
        if self.bodyB and self.bodyB is not self.bodyA:
            self.bodyB.remove_force(self)

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
