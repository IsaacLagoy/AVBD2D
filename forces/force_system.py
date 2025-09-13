import numpy as np
from helper.constants import ROWS, DEBUG_TIMING
from helper.decorators import timer
from forces.contact_system import ContactSystem
from collision.collide import sat
from collision.gjk import gjk, epa
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
        # early terminate
        if len(self.pairs) == 0:
            return
        
        # NOTE ideas
        # allocate enough space for the largest SAT
        # allocate temp arrays for SAT
        # find a better way of accessing the data from each body and mesh
        # 2-3 ms to get info
        
        pos = self.body_system.pos
        irs = self.body_system.irs
        s_ir = self.body_system.s_ir
        
        # TODO find a smarter way to do this with multiple meshes
        mesh_system = self.solver.mesh_system
        
        mesh = self.solver.mesh_system.meshes[0]
        normals = mesh.normals
        vertices = mesh.vertices
        dots = mesh.dots
        
        n_normals = normals.shape[0]
        n_verts = vertices.shape[0]
        
        index_a = np.empty(3, dtype='int16')
        index_b = np.empty(3, dtype='int16')
        minks = np.empty((3, 2), dtype='float32')
        
        # update to fit number of iterations - 3 from gjk
        support_buffer = np.empty((15, 2), dtype='float32')
        normal_buffer = np.empty((15, 2), dtype='float32')
        face_buffer = np.empty((15, 2), dtype='uint8')
        distance_buffer = np.empty(15, dtype='float32')
        set_buffer = np.empty(15, dtype='uint8')

        for i, j in self.pairs:
            collided = gjk(
                pos_a = self.body_system.pos[i],
                pos_b = self.body_system.pos[j],
                sr_a = self.body_system.irs[i],
                sr_b = self.body_system.irs[j],
                verts_a = vertices,
                verts_b = vertices,
                index_a = index_a,
                index_b = index_b,
                minks = minks,
                free = 0
            )
            
            if not collided:
                continue
            
            self.body_system.bodies[i].color = (255, 0, 0)
            self.body_system.bodies[j].color = (255, 0, 0)
            
            mtv = epa(
                pos_a = self.body_system.pos[i],
                pos_b = self.body_system.pos[j],
                sr_a = self.body_system.irs[i],
                sr_b = self.body_system.irs[j],
                verts_a = vertices,
                verts_b = vertices,
                simplex = minks,
                faces = face_buffer,
                sps = support_buffer,
                normals = normal_buffer,
                dists = distance_buffer,
                set = set_buffer
            )
            
            # self.body_system.pos[i, :2] -= mtv / 2
            # self.body_system.pos[j, :2] += mtv / 2
        
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