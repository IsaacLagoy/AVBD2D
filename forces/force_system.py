import numpy as np


class ForceSystem():
    
    def __init__(self, solver, max_forces: int) -> None:
        self.solver = solver
        
        self.max_forces = max_forces
        self.size = 0
        
        # derivatives
        self.J = np.zeros((self.max_forces, 2, 3), dtype='float32')
        self.H = np.zeros((self.max_forces, 2, 3, 3), dtype='float32')
        
        self.C     = np.zeros(max_forces, dtype='float32')
        self.motor = np.zeros(max_forces, dtype='float32')
        
        self.stiffness = np.full(max_forces, np.inf,  dtype='float32')
        self.fmax      = np.full(max_forces, np.inf,  dtype='float32')
        self.fmin      = np.full(max_forces, -np.inf, dtype='float32')
        self.fracture  = np.full(max_forces, np.inf,  dtype='float32')
        
        self.penalty = np.zeros(max_forces, dtype='float32')
        self.lamb    = np.zeros(max_forces, dtype='float32')
        
        # track empty indices
        self.free_indices = set(range(max_forces))
        
        # map index -> force object
        self.forces = {}
        
    def insert(self) -> int: