import numpy as np


class Color():
    
    def __init__(self, color_id) -> None:
        self.id = color_id
        self.head = None
        self.count = 0
        self.indices = []
        self.force_indices = []
        
    def add_body(self, body) -> None:
        """Add a rigid body to this color's linked list"""
        # ignore bodies that cant be affected by force
        if body.mass <= 0:
            return
        
        # Insert at head for O(1) insertion
        body.color_next = self.head
        self.head = body
        self.count += 1
        self.indices.append(body.index)
        
    def is_empty(self) -> bool:
        """Check if this color has no assigned bodies"""
        return self.head is None
    
    def reserve_space(self) -> None:
        """
        Allocates the necessary amount of space for completing all solver operations
        """
        self.rhs = np.zeros((self.count, 3), dtype='float64')
        self.lhs = np.zeros((self.count, 3, 3), dtype='float64')
    
    # --------------------
    # Iterators
    # --------------------
        
    def get_bodies_iterator(self):
        """Generator to iterate through all bodies in this color"""
        current = self.head
        while current is not None:
            yield current
            current = current.color_next
            
    # --------------------
    # Magic Methods
    # --------------------
            
    def __len__(self) -> int:
        """Return the number of bodies in this color"""
        return self.count
    
# TODO check if we need proper destructor or if we can let garbage collection do its thing