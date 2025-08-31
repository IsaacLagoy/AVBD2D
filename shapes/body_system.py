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
        
        # track empty indices
        self.free_indices = set(range(max_bodies))
        
        # map index -> body object
        self.bodies = {}
        
    def insert(self, pos, vel, friction, mass, moment) -> int:
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
        
        self.size += 1
        return index
        # TODO Rigid needs to add itself to the bodies list
        
    def delete(self, index) -> None:
        if index in self.bodies:
            del self.bodies[index]
            self.free_indices.add(index)
            self.size -= 1
            
    def compact(self) -> None:
        """
        Move active bodies to the contiguous front of the arrays using an O(n) front-back swap.
        """
        if not self.free_indices:
            return

        front = 0
        back = self.max_bodies - 1

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
            self.pos[front],      self.pos[back]      = self.pos[back],      self.pos[front]
            self.initial[front],  self.initial[back]  = self.initial[back],  self.initial[front]
            self.inertial[front], self.inertial[back] = self.inertial[back], self.inertial[front]

            self.vel[front],      self.vel[back]      = self.vel[back],      self.vel[front]
            self.prev_vel[front], self.prev_vel[back] = self.prev_vel[back], self.prev_vel[front]

            self.friction[front], self.friction[back] = self.friction[back], self.friction[front]
            self.mass[front],     self.mass[back]     = self.mass[back],     self.mass[front]
            self.moment[front],   self.moment[back]   = self.moment[back],   self.moment[front]

            # Update Body objects
            body = self.bodies.pop(back)
            body.index = front
            self.bodies[front] = body

            # Update free indices
            self.free_indices.remove(front)
            self.free_indices.add(back)

            # Move pointers
            front += 1
            back -= 1