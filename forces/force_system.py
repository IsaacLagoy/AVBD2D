import numpy as np
from helper.constants import ROWS, DEBUG_TIMING
from helper.decorators import timer
from forces.contact_system import ContactSystem
from collision.collide import collide
from shapes.rigid import Rigid


STICK_THRESH = 0.001

class ForceSystem():
    
    def __init__(self, solver, max_forces: int) -> None:
        self.solver = solver
        self.contacts = ContactSystem(self, 2048)
        
        self.max_forces = max_forces
        self.size = 0
        
        # types of forces
        # --------------------------
        # 0 = Contact Manifold
        # 1 = Joint
        # 2 = Spring
        # --------------------------
        self.type = np.zeros(max_forces, dtype=int)
        
        # derivatives
        self.J = np.zeros((max_forces, ROWS, 3), dtype='float32')
        self.H = np.zeros((max_forces, ROWS, 3, 3), dtype='float32')
        
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
        
        self.pairs = []
        
    def collide(self) -> None:
        mesh_system = self.solver.mesh_system
        
        num_pairs = len(self.pairs)
        
        # reserve enough space for the collisions and get slice
        self.contacts.reserve_space(num_pairs)
        normal_slice = self.contacts.normal[self.contacts.size : self.contacts.size + num_pairs]
        rA_slice = self.contacts.rA[self.contacts.size : self.contacts.size + num_pairs]
        rB_slice = self.contacts.rB[self.contacts.size : self.contacts.size + num_pairs]
        contact_slice = self.contacts.num_contact[self.contacts.size : self.contacts.size + num_pairs]
        
        collide(
            self.pairs, 
            mesh_system.vertices, 
            mesh_system.starts, 
            mesh_system.lengths, 
            self.body_system.pos, 
            self.body_system.irs, 
            self.body_system.s_ir,
            self.body_system.mesh,
            normal_slice,
            rA_slice,
            rB_slice,
            contact_slice
        )
        
        for i in range(num_pairs):
            if contact_slice[i] == 0:
                continue
            
            a = self.body_system.bodies[self.pairs[i, 0]]
            b = self.body_system.bodies[self.pairs[i, 1]]
            
            a.color = (255, 0, 0)
            b.color = (255, 0, 0)
        
    def insert(self, type: int) -> int:
        """
        Insert Force data into the arrays. This should only be called from the Force constructor. Returns index inserted
        """
        # allocate new space if old space is exceeded
        if not self.free_indices:
            new_max = self.max_forces * 2
            
            # allocate new arrays
            self.type = np.hstack(self.type, np.zeros(self.max_forces, dtype=int))
            
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
        self.type[index] = type
        
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
        # TODO Force needs to add itself to the forces manifolds
        
    def delete(self, index) -> None:
        if index in self.forces:
            del self.forces[index]
            self.free_indices.add(index)
            self.size -= 1
            
    @timer('Compacting Forces', on=DEBUG_TIMING)
    def compact(self) -> None:
        """
        Move active forces to the contiguous front of the arrays using vectorized operations.
        Much faster than element-by-element swaps.
        """
        if not self.free_indices or self.size == 0:
            return

        # Create mapping from old indices to new indices
        active_indices = [i for i in range(self.max_forces) if i not in self.free_indices]
        
        if len(active_indices) == 0:
            return
            
        # Only proceed if we actually need to move things
        if active_indices == list(range(len(active_indices))):
            return  # Already compact
            
        # Use numpy advanced indexing to reorder arrays (vectorized)
        self.type[:self.size] = self.type[active_indices]
        
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
        
    @property
    def body_system(self):
        return self.solver.body_system