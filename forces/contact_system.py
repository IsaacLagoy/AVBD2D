import numpy as np
from helper.constants import CONTACTS, DEBUG_TIMING
from helper.decorators import timer


class ContactSystem():
    
    def __init__(self, solver, max_contacts: int) -> None:
        self.solver = solver
        
        self.max_contacts = max_contacts
        self.size = 0
        
        # Contact properties
        self.normal = np.zeros((max_contacts, CONTACTS, 2), dtype='float32')  # vec2
        self.rA = np.zeros((max_contacts, CONTACTS, 2), dtype='float32')      # vec2
        self.rB = np.zeros((max_contacts, CONTACTS, 2), dtype='float32')      # vec2
        self.stick = np.zeros((max_contacts, CONTACTS), dtype=bool)           # boolean
        
        # Jacobians
        self.JAn = np.zeros((max_contacts, CONTACTS, 3), dtype='float32')     # vec3
        self.JBn = np.zeros((max_contacts, CONTACTS, 3), dtype='float32')     # vec3
        self.JAt = np.zeros((max_contacts, CONTACTS, 3), dtype='float32')     # vec3
        self.JBt = np.zeros((max_contacts, CONTACTS, 3), dtype='float32')     # vec3
        self.C0 = np.zeros((max_contacts, CONTACTS, 2), dtype='float32')      # vec2
        
        # simplex
        self.index_a = np.zeros((max_contacts, 4), dtype='int16')
        self.index_b = np.zeros((max_contacts, 4), dtype='int16')
        self.minkowski = np.zeros((max_contacts, 4, 3), dtype='float32') # minkowski space differences
        
        # manifold variables
        self.num_contact = np.zeros(max_contacts, dtype='int16') # only 0 - CONTACTS
        self.friction = np.zeros(max_contacts, dtype='float32')
        
        # track empty indices
        self.free_indices = set(range(max_contacts))
        
        # map index -> contact object
        self.contacts = {}
        
    def insert(self) -> int:
        """
        Insert Contact data into the arrays. This should only be called from the Contact constructor. Returns index inserted
        """
        # allocate new space if old space is exceeded
        if not self.free_indices:
            new_max = self.max_contacts * 2
            
            # allocate new arrays
            self.normal = np.vstack([self.normal, np.zeros((self.max_contacts, CONTACTS, 2), dtype='float32')])
            self.rA = np.vstack([self.rA, np.zeros((self.max_contacts, CONTACTS, 2), dtype='float32')])
            self.rB = np.vstack([self.rB, np.zeros((self.max_contacts, CONTACTS, 2), dtype='float32')])
            self.stick = np.vstack([self.stick, np.zeros((self.max_contacts, CONTACTS), dtype=bool)])
            
            self.JAn = np.vstack([self.JAn, np.zeros((self.max_contacts, CONTACTS, 3), dtype='float32')])
            self.JBn = np.vstack([self.JBn, np.zeros((self.max_contacts, CONTACTS, 3), dtype='float32')])
            self.JAt = np.vstack([self.JAt, np.zeros((self.max_contacts, CONTACTS, 3), dtype='float32')])
            self.JBt = np.vstack([self.JBt, np.zeros((self.max_contacts, CONTACTS, 3), dtype='float32')])
            self.C0 = np.vstack([self.C0, np.zeros((self.max_contacts, CONTACTS, 2), dtype='float32')])
            
            self.num_contact = np.hstack([self.num_contact, np.zeros(self.max_contacts, dtype='int16')])
            self.friction = np.hstack([self.friction, np.zeros(self.max_contacts, dtype='float32')])
            
            # add new free indices
            self.free_indices.update(range(self.max_contacts, new_max))
            self.max_contacts = new_max
        
        # add new contact
        index = self.free_indices.pop()
        
        # Reset to default values (important for reused indices)
        self.normal[index] = 0.0
        self.rA[index] = 0.0
        self.rB[index] = 0.0
        self.stick[index] = False
        
        self.JAn[index] = 0.0
        self.JBn[index] = 0.0
        self.JAt[index] = 0.0
        self.JBt[index] = 0.0
        self.C0[index] = 0.0
        
        self.num_contact[index] = 0
        self.friction[index] = 0
        
        self.size += 1
        return index
        # TODO Contact needs to add itself to the contacts list
        
    def delete(self, index) -> None:
        if index in self.contacts:
            del self.contacts[index]
            self.free_indices.add(index)
            self.size -= 1
            
    @timer('Compacting Contacts', on=DEBUG_TIMING)
    def compact(self) -> None:
        """
        Move active contacts to the contiguous front of the arrays using vectorized operations.
        Much faster than element-by-element swaps.
        """
        if not self.free_indices or self.size == 0:
            return

        # Create mapping from old indices to new indices
        active_indices = [i for i in range(self.max_contacts) if i not in self.free_indices]
        
        if len(active_indices) == 0:
            return
            
        # Only proceed if we actually need to move things
        if active_indices == list(range(len(active_indices))):
            return  # Already compact
            
        # Use numpy advanced indexing to reorder arrays (vectorized)
        self.normal[:self.size] = self.normal[active_indices]
        self.rA[:self.size] = self.rA[active_indices]
        self.rB[:self.size] = self.rB[active_indices]
        self.stick[:self.size] = self.stick[active_indices]
        
        self.JAn[:self.size] = self.JAn[active_indices]
        self.JBn[:self.size] = self.JBn[active_indices]
        self.JAt[:self.size] = self.JAt[active_indices]
        self.JBt[:self.size] = self.JBt[active_indices]
        self.C0[:self.size] = self.C0[active_indices]
        
        self.num_contact[:self.size] = self.num_contact[active_indices]
        self.friction[:self.size] = self.friction[active_indices]

        # Update contact objects and rebuild dictionary
        new_contacts = {}
        for new_idx, old_idx in enumerate(active_indices):
            if old_idx in self.contacts:
                contact = self.contacts[old_idx]
                contact.index = new_idx
                new_contacts[new_idx] = contact
        
        self.contacts = new_contacts

        # Update free indices to be the tail end
        self.free_indices = set(range(self.size, self.max_contacts))
            
    def is_compact(self) -> bool:
        """
        Verify if the contact system is compact (all active contacts are at the front).
        Returns True if compact, False otherwise.
        """
        # If no contacts or all slots are used, it's automatically compact
        if self.size == 0 or len(self.free_indices) == 0:
            return True
            
        # Check that all indices from 0 to size-1 are active (not in free_indices)
        for i in range(self.size):
            if i in self.free_indices:
                return False
                
        # Check that all indices from size to max_contacts-1 are free
        for i in range(self.size, self.max_contacts):
            if i not in self.free_indices:
                return False
                
        # Verify that contacts dictionary keys match the active range
        active_indices = set(range(self.size))
        if set(self.contacts.keys()) != active_indices:
            return False
            
        return True