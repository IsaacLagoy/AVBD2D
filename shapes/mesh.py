import numpy as np


class Mesh():
    
    def __init__(self, vertices) -> None:
        # vertices must be wound counter clockwise
        self.vertices = vertices
        
        # compute normals
        edges = np.roll(self.vertices, -1, axis=0) - self.vertices
        self.normals = np.stack([edges[:, 1], -edges[:, 0]], axis=1)
        lengths = np.linalg.norm(self.normals, axis=1, keepdims=True)
        self.normals /= np.where(lengths == 0, 1, lengths)
    
        self.dots = np.sum(self.normals * self.vertices, axis=1)