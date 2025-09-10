from shapes.mesh_system import MeshSystem


# NOTE vertices must be wound counter clockwise
class Mesh():
    
    def __init__(self, system: MeshSystem, vertices) -> None:
        self.system = system
        
        # insert self into mesh system
        self.index = system.insert(vertices)
        self.system.meshes[self.index] = self
        
    # ------------------
    # System Access
    # ------------------
    @property
    def start(self) -> int:
        return self.system.starts[self.index]
    
    @property
    def length(self) -> int:
        return self.system.lengths[self.index]
    
    @property
    def vertices(self):
        return self.system.vertices[self.start : self.start + self.length]
    
    @property
    def normals(self):
        return self.system.normals[self.start : self.start + self.length]
    
    @property
    def dots(self):
        return self.system.dots[self.start : self.start + self.length]
        