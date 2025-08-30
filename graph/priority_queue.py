from shapes.rigid import Rigid


class DSaturPriorityQueue:
    """
    Max-heap priority queue for DSATUR algorithm.
    Priority: saturation_degree (primary), degree (secondary), id (tie-breaker)
    Uses negative values to simulate max-heap with Python's min-heap
    """
    
    def __init__(self) -> None:
        self.heap: list[tuple] = []  # main heap
        
    def compare(self, a: Rigid, b: Rigid) -> bool:
        if a.saturation_degree != b.saturation_degree:
            return a.saturation_degree < b.saturation_degree
        
        if a.degree != b.degree:
            return a.degree < b.degree
        
        return id(a) < id(b)
    
    