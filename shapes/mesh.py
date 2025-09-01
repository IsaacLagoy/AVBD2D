import numpy as np


class Mesh():
    
    def __init__(self, vertices) -> None:
        # vertices must be wound counter clockwise
        self.vertices = vertices
        
    @property
    def edges(self):
        pass # TODO