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