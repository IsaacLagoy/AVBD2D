from shapes.rigid import Rigid


class DSaturPriorityQueue:
    """
    Max-heap priority queue for DSATUR algorithm.
    Priority: saturation_degree (primary), degree (secondary), id (tie-breaker)
    Custom heap implementation without using heapq
    """
    
    def __init__(self) -> None:
        self.heap: list[Rigid] = []  # main heap of Rigid bodies
        # add hashmap from Rigid* -> int heap index
        
    def compare(self, a: Rigid, b: Rigid) -> bool:
        """Compare two bodies - returns True if a has lower priority than b"""
        if a.saturation_degree != b.saturation_degree:
            return a.saturation_degree < b.saturation_degree
        
        if a.degree != b.degree:
            return a.degree < b.degree
        
        return id(a) < id(b)
    
    def parent(self, index: int) -> int:
        """Get parent index"""
        return (index - 1) // 2
    
    def left_child(self, index: int) -> int:
        """Get left child index"""
        return 2 * index + 1
    
    def right_child(self, index: int) -> int:
        """Get right child index"""
        return 2 * index + 2
    
    def _swap(self, i: int, j: int) -> None:
        """Swap elements at indices i and j"""
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
    
    def _bubble_up(self, index: int) -> None:
        """Move element up to maintain max-heap property"""
        while index > 0:
            parent_idx = self.parent(index)
            # If current element has higher priority than parent, swap
            if self.compare(self.heap[parent_idx], self.heap[index]):
                self._swap(index, parent_idx)
                index = parent_idx
            else:
                break
    
    def _bubble_down(self, index: int) -> None:
        """Move element down to maintain max-heap property"""
        while True:
            largest = index
            left = self.left_child(index)
            right = self.right_child(index)
            
            # Find the largest among current, left, and right
            if (left < len(self.heap) and 
                self.compare(self.heap[largest], self.heap[left])):
                largest = left
                
            if (right < len(self.heap) and 
                self.compare(self.heap[largest], self.heap[right])):
                largest = right
            
            # If largest is not current, swap and continue
            if largest != index:
                self._swap(index, largest)
                index = largest
            else:
                break
    
    def size(self) -> int:
        """Return the number of elements in the heap"""
        return len(self.heap)
    
    def insert(self, body: Rigid) -> None:
        """Insert a body into the priority queue"""
        self.heap.append(body)
        self._bubble_up(len(self.heap) - 1)
    
    def search(self, body: Rigid) -> int:
        """Search for a body in the heap, return its index (-1 if not found)"""
        for i, heap_body in enumerate(self.heap):
            if heap_body is body:
                return i
        return -1
    
    def pop(self) -> Rigid:
        """Remove and return the body with highest priority"""
        if not self.heap:
            raise IndexError("pop from empty heap")
        
        if len(self.heap) == 1:
            return self.heap.pop()
        
        # Store the root (max element)
        max_body = self.heap[0]
        
        # Move last element to root and remove last
        self.heap[0] = self.heap.pop()
        
        # Restore heap property
        self._bubble_down(0)
        
        return max_body
    
    def update(self, body: Rigid) -> None:
        """Update a body's priority (useful when saturation changes)"""
        index = self.search(body)
        if index != -1:
            # Store original position to detect if we need to bubble
            original_index = index
            self._bubble_up(index)
            # If bubble_up didn't move it, try bubble_down
            if index == original_index:
                self._bubble_down(index)
    
    def clear(self) -> None:
        """Clear all elements from the heap"""
        self.heap.clear()
    
    def __len__(self) -> int:
        return len(self.heap)
    
    def __bool__(self) -> bool:
        return len(self.heap) > 0
    
    def __repr__(self) -> str:
        if not self.heap:
            return "DSaturPQ[]"
        
        entries = []
        for body in self.heap[:min(5, len(self.heap))]:  # Show first 5
            entries.append(f"Body({id(body)}: sat={body.saturation_degree}, deg={body.degree})")
        
        result = f"DSaturPQ[{', '.join(entries)}"
        if len(self.heap) > 5:
            result += f", ...+{len(self.heap) - 5} more"
        result += "]"
        return result