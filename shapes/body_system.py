import numpy as np


class BodySystem():
    
    def __init__(self, solver, max_bodies: int) -> None:
        self.solver = solver
        
        self.max_bodies = max_bodies
        self.size = 0 # number of active bodies
        
        # position
        self.pos      = np.zeros((max_bodies, 3), dtype='float32')
        self.initial  = np.zeros((max_bodies, 3), dtype='float32')
        self.inertial = np.zeros((max_bodies, 3), dtype='float32')
        
        # velocity
        self.vel      = np.zeros((max_bodies, 3), dtype='float32')
        self.prev_vel = np.zeros((max_bodies, 3), dtype='float32')
        
        # attribute
        self.friction = np.ones(max_bodies, dtype='float32')
        self.mass     = np.ones(max_bodies, dtype='float32')
        self.moment   = np.ones(max_bodies, dtype='float32')
        
        self.scale = np.ones((max_bodies, 2), dtype='float32')
        self.radius = np.ones(max_bodies, dtype='float32')
        
        # object reference
        self.mesh = np.zeros(max_bodies, dtype='int32')
        
        # mats
        self.s_ir = np.zeros((max_bodies, 2, 2), dtype='float32')
        self.irs  = np.zeros((max_bodies, 2, 2), dtype='float32')
        
        # lazy updating
        self.updated  = np.zeros(max_bodies, dtype=np.bool_)
        
        # track empty indices
        self.free_indices = set(range(max_bodies))
        
        # map index -> body object
        self.bodies = {}
        
    def insert(self, pos, vel, friction, mass, moment, scale, mesh_idx) -> int:
        """
        Insert Rigid data into the arrays. This should only be called from the Rigid constructor. Returns index inserted
        """
        # allocate new space if old space is exceeded
        if not self.free_indices:
            new_max = self.max_bodies * 2
            
            # allocate new arrays
            self.pos      = np.vstack([self.pos,      np.zeros((self.max_bodies, 3), dtype='float32')])
            self.initial  = np.vstack([self.initial,  np.zeros((self.max_bodies, 3), dtype='float32')])
            self.inertial = np.vstack([self.inertial, np.zeros((self.max_bodies, 3), dtype='float32')])
            
            self.vel      = np.vstack([self.vel,      np.zeros((self.max_bodies, 3), dtype='float32')])
            self.prev_vel = np.vstack([self.prev_vel, np.zeros((self.max_bodies, 3), dtype='float32')])
            
            self.friction = np.hstack([self.friction, np.ones(self.max_bodies, dtype='float32')])
            self.mass     = np.hstack([self.mass,     np.ones(self.max_bodies, dtype='float32')])
            self.moment   = np.hstack([self.moment,   np.ones(self.max_bodies, dtype='float32')])
            
            self.scale = np.vstack([self.scale, np.ones((self.max_bodies, 2), dtype='float32')])
            self.radius = np.hstack((self.radius, np.ones(self.max_bodies, dtype='float32')))
            
            self.mesh = np.hstack((self.mesh, np.zeros(self.max_bodies, dtype='int32')))
            
            self.s_ir = np.vstack((self.s_ir, np.zeros((self.max_bodies, 2, 2), dtype='float32')))
            self.irs  = np.vstack((self.irs, np.zeros((self.max_bodies, 2, 2), dtype='float32')))
            
            self.updated  = np.hstack([self.updated,   np.ones(self.max_bodies, dtype=bool)])
            
            # add new free indices
            self.free_indices.update(range(self.max_bodies, new_max))
            self.max_bodies = new_max
        
        # add new rigid
        index = self.free_indices.pop()
        
        self.pos[index]      = [i for i in pos]
        self.initial[index]  = [i for i in pos]
        self.inertial[index] = [i for i in pos]
        
        self.vel[index]      = [i for i in vel]
        self.prev_vel[index] = [i for i in vel]
        
        self.friction[index] = friction
        self.mass[index]     = mass
        self.moment[index]   = moment
        
        self.scale[index] = [i for i in scale]
        self.radius[index] = np.linalg.norm(self.scale[index])
        
        self.mesh[index] = mesh_idx
        
        # NOTE will update matrices when needed, not on initialization
        
        self.updated[index]  = False
        
        self.size += 1
        return index
        
    def delete(self, index) -> None:
        if index in self.bodies:
            del self.bodies[index]
            self.free_indices.add(index)
            self.size -= 1
            
    def compact(self) -> None:
        """
        Move active bodies to the contiguous front of the arrays using an O(n) front-back swap.
        """
        if not self.free_indices or self.size == 0:
            return
        
        # Create mapping from old indices to new indices
        active_indices = [i for i in range(self.max_bodies) if i not in self.free_indices]
        
        if len(active_indices) == 0:
            return
        
        # Only proceed if we actually need to move things
        if active_indices == list(range(len(active_indices))):
            return  # Already compact
        
        # Use numpy advanced indexing to reorder arrays (vectorized)
        self.pos[:self.size] = self.pos[active_indices]
        self.initial[:self.size] = self.initial[active_indices]
        self.inertial[:self.size] = self.inertial[active_indices]
        
        self.vel[:self.size] = self.vel[active_indices]
        self.prev_vel[:self.size] = self.prev_vel[active_indices]
        
        self.friction[:self.size] = self.friction[active_indices]
        self.mass[:self.size] = self.mass[active_indices]
        self.moment[:self.size] = self.moment[active_indices]
        
        self.scale[:self.size] = self.scale[active_indices]
        self.radius[:self.size] = self.radius[active_indices]
        
        self.mesh[:self.size] = self.mesh[active_indices]
        
        self.s_ir[:self.size] = self.s_ir[active_indices]
        self.irs[:self.size] = self.irs[active_indices]
        
        self.updated[:self.size] = self.updated[active_indices]
        
        # update rigid objects and rebuild dictionary
        new_bodies = {}
        for new_idx, old_idx in enumerate(active_indices):
            if old_idx in self.bodies:
                body = self.bodies[old_idx]
                body.index = new_idx
                new_bodies[new_idx] = body
                
        self.bodies = new_bodies
        
        # update free indices with tail
        self.free_indices = set(range(self.size, self.max_bodies))