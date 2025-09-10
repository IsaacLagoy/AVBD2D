import numpy as np

# NOTE removing a mesh is currently not supported by this class
# if mesh removing becomes a requirement, this will need to be changed

class MeshSystem():
    
    def __init__(self, solver, max_verts: int) -> None:
        self.solver = solver
    
        # controls the maximum amount of vertices contained across all meshes
        self.max_verts = max_verts
        self.vertices = np.zeros((self.max_verts, 2), dtype='float32')
        self.normals  = np.zeros((self.max_verts, 2), dtype='float32')
        
        # store near dot products for quick collisions
        self.dots = np.zeros(self.max_verts, dtype='float32')
        
        # each mesh must have at least 3 vertices (triangle)
        self.max_meshes = max_verts // 3
        self.starts  = np.zeros(self.max_meshes, dtype='int32')
        self.lengths = np.zeros(self.max_meshes, dtype='int32')
        
        # track size
        self.next_start = 0
        self.next_mesh = 0
        
        # map index -> mesh object
        self.meshes = {}

    def insert(self, vertices) -> int:        
        # compute edges and normals
        edges = np.roll(vertices, -1, axis=0) - vertices
        normals = np.stack([edges[:, 1], -edges[:, 0]], axis=1)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        if np.any(lengths == 0): raise ValueError("Zero-length edge detected: cannot normalize normal vector.")
        normals /= lengths
        
        # compute collision dots
        dots = np.sum(normals * vertices, axis=1)
        
        # allocate new space for mesh indexing
        if self.next_mesh == self.max_meshes:
            new_max = self.max_meshes * 2
            
            # allocate new arrays
            self.starts  = np.hstack((self.starts,  np.zeros(self.max_meshes, dtype='int32')))
            self.lengths = np.hstack((self.lengths, np.zeros(self.max_meshes, dtype='int32')))
            
            self.max_meshes = new_max
            
        # compute space to store mesh
        length = len(vertices)
        
        # allocate new space if old space is exceeded
        available_space = self.max_verts - length - self.next_start
        
        if available_space < 0:
            new_max = self.max_verts * 2
            
            # allocate new arrays
            self.vertices = np.vstack((self.vertices, np.zeros((self.max_verts, 2), dtype='float32')))
            self.normals  = np.vstack((self.normals,  np.zeros((self.max_verts, 2), dtype='float32')))
            
            self.dots = np.hstack((self.dots, np.zeros(self.max_verts, dtype='float32')))
            
            self.max_verts = new_max
            
        # insert new mesh
        end = self.next_start + length
        
        self.vertices[self.next_start:end] = vertices
        self.normals[self.next_start:end]  = normals
        self.dots[self.next_start:end]     = dots
        
        self.starts[self.next_mesh]  = self.next_start
        self.lengths[self.next_mesh] = length
        
        self.next_start += length
        self.next_mesh += 1
        
        # return the index of the current mesh
        return self.next_mesh - 1
        
        # NOTE the mesh object will add itself to the meshes dictionary