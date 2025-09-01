from forces.force_system import ForceSystem
from shapes.rigid import Rigid
from helper.constants import ROWS


class Force():
    
    def __init__(self, system: ForceSystem, body_a: Rigid, body_b: Rigid) -> None:
        self.system = system
        
        # initiate linked lists
        self.next = self.solver.forces
        self.solver.forces = self
        
        self.body_a = body_a
        self.body_b = body_b
        
        self.next_a = None
        self.next_b = None
        
        # link to the body linked lists
        if body_a:
            self.next_a = body_a.forces
            body_a.forces = self
            
        if body_b:
            self.next_b = body_b.forces
            body_b.forces = self
            
        # increment graph degree tracker if force is an edge
        if body_a and body_b:
            body_a.degree += 1
            body_b.degree += 1
            
        # initialize in the system
        self.index = self.system.insert()
        self.system.forces[self.index] = self
        
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
        
    # piped shortcuts
    @property
    def solver(self):
        return self.system.solver
        
    # Derivatives
    @property
    def J(self):
        return self.system.J[self.index]
    
    @J.setter
    def J(self, value):
        self.system.J[self.index] = value
    
    @property
    def H(self):
        return self.system.H[self.index]
    
    @H.setter
    def H(self, value):
        self.system.H[self.index] = value
    
    # Basic force properties
    @property
    def C(self):
        return self.system.C[self.index]
    
    @C.setter
    def C(self, value):
        self.system.C[self.index] = value
    
    @property
    def motor(self):
        return self.system.motor[self.index]
    
    @motor.setter
    def motor(self, value):
        self.system.motor[self.index] = value
    
    # Constraint parameters
    @property
    def stiffness(self):
        return self.system.stiffness[self.index]
    
    @stiffness.setter
    def stiffness(self, value):
        self.system.stiffness[self.index] = value
    
    @property
    def fmax(self):
        return self.system.fmax[self.index]
    
    @fmax.setter
    def fmax(self, value):
        self.system.fmax[self.index] = value
    
    @property
    def fmin(self):
        return self.system.fmin[self.index]
    
    @fmin.setter
    def fmin(self, value):
        self.system.fmin[self.index] = value
    
    @property
    def fracture(self):
        return self.system.fracture[self.index]
    
    @fracture.setter
    def fracture(self, value):
        self.system.fracture[self.index] = value
    
    # Solver parameters
    @property
    def penalty(self):
        return self.system.penalty[self.index]
    
    @penalty.setter
    def penalty(self, value):
        self.system.penalty[self.index] = value
    
    @property
    def lamb(self):
        return self.system.lamb[self.index]
    
    @lamb.setter
    def lamb(self, value):
        self.system.lamb[self.index] = value