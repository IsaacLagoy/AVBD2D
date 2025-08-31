import numpy as np
from helper.constants import ROWS


class ForceSystem():
    
    def __init__(self, solver, max_forces: int) -> None:
        self.solver = solver
        
        self.max_forces = max_forces
        self.size = 0
        
        # derivatives
        self.J = np.zeros((self.max_forces, ROWS, 3), dtype='float32')
        self.H = np.zeros((self.max_forces, ROWS, 3, 3), dtype='float32')
        
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
        """
        Insert Force data into the arrays. This should only be called from the Force constructor. Returns index inserted
        """
        # allocate new space if old space is exceeded
        if not self.free_indices:
            new_max = self.max_forces * 2
            
            # allocate new arrays
            self.J = np.vstack([self.J, np.zeros((self.max_forces, 2, 3), dtype='float32')])
            self.H = np.vstack([self.H, np.zeros((self.max_forces, 2, 3, 3), dtype='float32')])
            
            self.C     = np.hstack([self.C,     np.zeros(self.max_forces, dtype='float32')])
            self.motor = np.hstack([self.motor, np.zeros(self.max_forces, dtype='float32')])
            
            self.stiffness = np.hstack([self.stiffness, np.full(self.max_forces, np.inf,  dtype='float32')])
            self.fmax      = np.hstack([self.fmax,      np.full(self.max_forces, np.inf,  dtype='float32')])
            self.fmin      = np.hstack([self.fmin,      np.full(self.max_forces, -np.inf, dtype='float32')])
            self.fracture  = np.hstack([self.fracture,  np.full(self.max_forces, np.inf,  dtype='float32')])
            
            self.penalty = np.hstack([self.penalty, np.zeros(self.max_forces, dtype='float32')])
            self.lamb    = np.hstack([self.lamb,    np.zeros(self.max_forces, dtype='float32')])
            
            # add new free indices
            self.free_indices.update(range(self.max_forces, new_max))
            self.max_forces = new_max
        
        # add new force
        index = self.free_indices.pop()
        
        # Reset to default values (important for reused indices)
        self.J[index] = 0.0
        self.H[index] = 0.0
        
        self.C[index] = 0.0
        self.motor[index] = 0.0
        
        self.stiffness[index] = np.inf
        self.fmax[index] = np.inf
        self.fmin[index] = -np.inf
        self.fracture[index] = np.inf
        
        self.penalty[index] = 0.0
        self.lamb[index] = 0.0
        
        self.size += 1
        return index
        # TODO Force needs to add itself to the forces list
        
    def delete(self, index) -> None:
        if index in self.forces:
            del self.forces[index]
            self.free_indices.add(index)
            self.size -= 1
            
    def compact(self) -> None:
        """
        Move active forces to the contiguous front of the arrays using an O(n) front-back swap.
        """
        if not self.free_indices:
            return

        front = 0
        back = self.max_forces - 1

        while front < back:
            # Move front forward to the first free slot
            while front < back and front not in self.free_indices:
                front += 1
            # Move back backward to the last active slot
            while front < back and back in self.free_indices:
                back -= 1

            if front >= back:
                break

            # Swap all attributes
            self.J[front],        self.J[back]        = self.J[back],        self.J[front]
            self.H[front],        self.H[back]        = self.H[back],        self.H[front]
            
            self.C[front],        self.C[back]        = self.C[back],        self.C[front]
            self.motor[front],    self.motor[back]    = self.motor[back],    self.motor[front]
            
            self.stiffness[front], self.stiffness[back] = self.stiffness[back], self.stiffness[front]
            self.fmax[front],     self.fmax[back]     = self.fmax[back],     self.fmax[front]
            self.fmin[front],     self.fmin[back]     = self.fmin[back],     self.fmin[front]
            self.fracture[front], self.fracture[back] = self.fracture[back], self.fracture[front]
            
            self.penalty[front],  self.penalty[back]  = self.penalty[back],  self.penalty[front]
            self.lamb[front],     self.lamb[back]     = self.lamb[back],     self.lamb[front]

            # Update Force objects
            force = self.forces.pop(back)
            force.index = front
            self.forces[front] = force

            # Update free indices
            self.free_indices.remove(front)
            self.free_indices.add(back)

            # Move pointers
            front += 1
            back -= 1