from glm import vec2


class Mesh():
    
    def __init__(self, vertices: list[vec2]) -> None:
        # vertices must be wrapped counter clockwise
        self.vertices = vertices
        
    @property
    def edges(self) -> list[tuple[vec2, vec2]]:
        return [(self.vertices[i], self.vertices[(i + 1) % 4]) for i in range(len(self.vertices))]