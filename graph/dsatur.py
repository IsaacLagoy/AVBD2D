from graph.color import Color
from graph.priority_queue import DSaturPriorityQueue
from shapes.rigid import Rigid

# TODO enure that this works on disconnected graphs
def dsatur_coloring(solver) -> list[Color]:
    # create priority queue
    pq = DSaturPriorityQueue()
    for rigid in solver.get_bodies_iterator():
        rigid.reset_coloring() # TODO ensure that this is the correct lovcation for resetting
        pq.insert(rigid)
        
    colors: list[Color] = []
       
    while len(pq):
        # get the vertex with the highest degree
        rigid: Rigid = pq.pop()
        
        # color rigid
        color = rigid.get_next_unused_color()
        to_update = rigid.assign_color(color)
        
        # append to next color
        if color > len(colors) - 1:
            colors.append(Color(color))
            
        # insert rigid into color linkedlist
        colors[color].add_body(rigid)
        
        # update adjacent bodies in the pq
        for adj in to_update:
            pq.update(adj)
            
    return colors