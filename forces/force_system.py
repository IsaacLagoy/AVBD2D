import numpy as np
from helper.constants import ROWS, DEBUG_TIMING
from helper.decorators import timer


class ForceSystem():
    
    def __init__(self, solver, max_forces: int) -> None:
        self.solver = solver
        
        self.max_forces = max_forces
        self.size = 0
        
        # derivatives
        self.J = np.zeros((self.max_forces, ROWS, 3), dtype='float32')
        self.H = np.zeros((self.max_forces, ROWS, 3, 3), dtype='float32')
        
        self.C     = np.zeros((max_forces, ROWS), dtype='float32')
        self.motor = np.zeros((max_forces, ROWS), dtype='float32')
        
        self.stiffness = np.full((max_forces, ROWS), np.inf,  dtype='float32')
        self.fmax      = np.full((max_forces, ROWS), np.inf,  dtype='float32')
        self.fmin      = np.full((max_forces, ROWS), -np.inf, dtype='float32')
        self.fracture  = np.full((max_forces, ROWS), np.inf,  dtype='float32')
        
        self.penalty = np.zeros((max_forces, ROWS), dtype='float32')
        self.lamb    = np.zeros((max_forces, ROWS), dtype='float32')
        
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
            self.J = np.vstack([self.J, np.zeros((self.max_forces, ROWS, 3), dtype='float32')])
            self.H = np.vstack([self.H, np.zeros((self.max_forces, ROWS, 3, 3), dtype='float32')])
            
            self.C     = np.vstack([self.C,     np.zeros((self.max_forces, ROWS), dtype='float32')])
            self.motor = np.vstack([self.motor, np.zeros((self.max_forces, ROWS), dtype='float32')])
            
            self.stiffness = np.vstack([self.stiffness, np.full((self.max_forces, ROWS), np.inf,  dtype='float32')])
            self.fmax      = np.vstack([self.fmax,      np.full((self.max_forces, ROWS), np.inf,  dtype='float32')])
            self.fmin      = np.vstack([self.fmin,      np.full((self.max_forces, ROWS), -np.inf, dtype='float32')])
            self.fracture  = np.vstack([self.fracture,  np.full((self.max_forces, ROWS), np.inf,  dtype='float32')])
            
            self.penalty = np.vstack([self.penalty, np.zeros((self.max_forces, ROWS), dtype='float32')])
            self.lamb    = np.vstack([self.lamb,    np.zeros((self.max_forces, ROWS), dtype='float32')])
            
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
            
    @timer('Compacting Forces', on=True)
    def compact(self) -> None:
        """
        Move active forces to the contiguous front of the arrays using vectorized operations.
        Much faster than element-by-element swaps.
        """
        if not self.free_indices or self.size == 0:
            return
        
        print(len(self.free_indices))

        # Create mapping from old indices to new indices
        active_indices = [i for i in range(self.max_forces) if i not in self.free_indices]
        
        if len(active_indices) == 0:
            return
            
        # Only proceed if we actually need to move things
        if active_indices == list(range(len(active_indices))):
            return  # Already compact
            
        # Use numpy advanced indexing to reorder arrays (vectorized)
        self.J[:self.size] = self.J[active_indices]
        self.H[:self.size] = self.H[active_indices]
        
        self.C[:self.size] = self.C[active_indices]
        self.motor[:self.size] = self.motor[active_indices]
        
        self.stiffness[:self.size] = self.stiffness[active_indices]
        self.fmax[:self.size] = self.fmax[active_indices]
        self.fmin[:self.size] = self.fmin[active_indices]
        self.fracture[:self.size] = self.fracture[active_indices]
        
        self.penalty[:self.size] = self.penalty[active_indices]
        self.lamb[:self.size] = self.lamb[active_indices]

        # Update force objects and rebuild dictionary
        new_forces = {}
        for new_idx, old_idx in enumerate(active_indices):
            if old_idx in self.forces:
                force = self.forces[old_idx]
                force.index = new_idx
                new_forces[new_idx] = force
        
        self.forces = new_forces

        # Update free indices to be the tail end
        self.free_indices = set(range(self.size, self.max_forces))

            
    def is_compact(self) -> bool:
        """
        Verify if the force system is compact (all active forces are at the front).
        Returns True if compact, False otherwise.
        """
        # If no forces or all slots are used, it's automatically compact
        if self.size == 0 or len(self.free_indices) == 0:
            return True
            
        # Check that all indices from 0 to size-1 are active (not in free_indices)
        for i in range(self.size):
            if i in self.free_indices:
                return False
                
        # Check that all indices from size to max_forces-1 are free
        for i in range(self.size, self.max_forces):
            if i not in self.free_indices:
                return False
                
        # Verify that forces dictionary keys match the active range
        active_indices = set(range(self.size))
        if set(self.forces.keys()) != active_indices:
            return False
            
        return True