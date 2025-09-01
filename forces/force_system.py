import numpy as np
from helper.constants import ROWS, DEBUG_TIMING
from helper.decorators import timer
from forces.contact_system import ContactSystem

STICK_THRESH = 0.001

class ForceSystem():
    
    def __init__(self, solver, max_forces: int) -> None:
        self.solver = solver
        self.contacts = ContactSystem(self, 2048)
        
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
        # TODO Force needs to add itself to the forces manifolds
    def compute_constraints_manifold(self, alpha: float, manifolds) -> None:
        if not manifolds:
            return
        
        # get all body positions
        num_manifolds = len(manifolds)
        dpA = np.zeros((num_manifolds, 3), dtype='float32')
        dpB = np.zeros((num_manifolds, 3), dtype='float32')
        
        for i, force in enumerate(manifolds):
            manifold = self.forces[force]
            dpA[i] = manifold.body_a.pos - manifold.body_a.initial
            dpB[i] = manifold.body_b.pos - manifold.body_b.initial
            
        # get contact indices for all manifolds in list
        contact_indices = np.array([self.forces[idx].contact_index for idx in manifolds])
        num_contacts = self.contacts.num_contact[contact_indices]
            
        # Vectorized constraint computation
        JAn = self.contacts.JAn[contact_indices]
        JBn = self.contacts.JBn[contact_indices]
        JAt = self.contacts.JAt[contact_indices]
        JBt = self.contacts.JBt[contact_indices]
        C0 = self.contacts.C0[contact_indices]
        friction_coeffs = self.contacts.friction[contact_indices]
        
        # Broadcast position deltas
        dpA_expanded = dpA[:, None, :]
        dpB_expanded = dpB[:, None, :]
        
        # Compute all constraints at once
        C_normal = (C0[:, :, 0] * (1 - alpha) + 
                    np.sum(JAn * dpA_expanded, axis=2) + 
                    np.sum(JBn * dpB_expanded, axis=2))
        
        C_tangent = (C0[:, :, 1] * (1 - alpha) + 
                    np.sum(JAt * dpA_expanded, axis=2) + 
                    np.sum(JBt * dpB_expanded, axis=2))
        
        # Vectorized friction computation
        normal_forces = np.abs(self.lamb[manifolds][:, ::2])  # Even indices
        friction_bounds = normal_forces * friction_coeffs[:, None]
        
        # Update all constraints and friction bounds
        for i, force_idx in enumerate(manifolds):
            contact_idx = contact_indices[i]
            n_contacts = num_contacts[i]
            
            # Update constraints
            self.C[force_idx, ::2][:n_contacts] = C_normal[i, :n_contacts]  # Normal
            self.C[force_idx, 1::2][:n_contacts] = C_tangent[i, :n_contacts]  # Tangent
            
            # Update friction bounds
            self.fmax[force_idx, 1::2][:n_contacts] = friction_bounds[i, :n_contacts]
            self.fmin[force_idx, 1::2][:n_contacts] = -friction_bounds[i, :n_contacts]
            
            # Update stick conditions
            tangent_forces = np.abs(self.lamb[force_idx, 1::2][:n_contacts])
            tangent_violations = np.abs(C0[i, :n_contacts, 1])
            
            stick_mask = ((tangent_forces < friction_bounds[i, :n_contacts]) & 
                        (tangent_violations < STICK_THRESH))
            
            self.contacts.stick[contact_idx, :n_contacts] = stick_mask
        
        
    def compute_derivatives_manifold_vectorized(self, bodies, manifolds) -> None:
        """
        Fully vectorized version for better performance with large batches.
        """
        if not bodies or not manifolds or len(bodies) != len(manifolds):
            return
        
        # Get all manifold and contact data
        manifold_objects = [self.forces[idx] for idx in manifolds]
        contact_indices = np.array([m.contact_index for m in manifold_objects])
        body_a_indices = np.array([m.body_a_index for m in manifold_objects])
        
        # Determine which bodies are body A vs body B
        bodies_array = np.array(bodies)
        is_body_a = bodies_array == body_a_indices  # Boolean mask
        
        # Get contact counts and Jacobian data
        num_contacts = self.contacts.num_contact[contact_indices]
        JAn = self.contacts.JAn[contact_indices]  # (num_pairs, CONTACTS, 3)
        JAt = self.contacts.JAt[contact_indices]  # (num_pairs, CONTACTS, 3)
        JBn = self.contacts.JBn[contact_indices]  # (num_pairs, CONTACTS, 3)
        JBt = self.contacts.JBt[contact_indices]  # (num_pairs, CONTACTS, 3)
        
        # Update Jacobians for each pair
        for i, force_idx in enumerate(manifolds):
            n_contacts = num_contacts[i]
            
            if is_body_a[i]:
                # Use body A Jacobians
                self.J[force_idx, ::2][:n_contacts] = JAn[i, :n_contacts]    # Normal (even indices)
                self.J[force_idx, 1::2][:n_contacts] = JAt[i, :n_contacts]   # Tangent (odd indices)
            else:
                # Use body B Jacobians
                self.J[force_idx, ::2][:n_contacts] = JBn[i, :n_contacts]    # Normal (even indices)
                self.J[force_idx, 1::2][:n_contacts] = JBt[i, :n_contacts]   # Tangent (odd indices)
                
    def initialize_manifolds(self, manifolds):
        ...
        
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