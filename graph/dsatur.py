from typing import List, Dict
from graph.color import Color
from graph.priority_queue import DSaturPriorityQueue
from shapes.rigid import Rigid


def dsatur_coloring(solver) -> tuple[int, List[Color]]:
    """
    DSATUR graph coloring algorithm using linked lists and priority queue.
    
    Args:
        solver: Physics solver containing linked list of bodies
        
    Returns:
        tuple: (chromatic_number, list_of_colors)
    """
    
    # Initialize priority queue and collect all bodies
    pq = DSaturPriorityQueue()
    bodies = []
    
    # Collect all bodies from solver's linked list
    current = solver.bodies
    while current is not None:
        current.reset_coloring()  # Reset any previous coloring
        bodies.append(current)
        current = current.next
    
    if not bodies:
        return 0, []
    
    # Initialize saturation degrees and add to priority queue
    for body in bodies:
        body.update_saturation()  # Should be 0 initially
        pq.push(body)
    
    # Initialize color tracking
    colors: List[Color] = []
    next_color_id = 0
    colored_count = 0
    total_bodies = len(bodies)
    
    # Main DSATUR loop
    while colored_count < total_bodies and not pq.is_empty():
        # Select body with highest saturation degree
        selected_body = pq.pop()
        if selected_body is None:
            break
            
        # Update saturation one more time to be safe
        selected_body.update_saturation()
        
        # Find the smallest available color
        available_color = selected_body.get_next_unused_color()
        
        # Ensure we have enough color objects
        while len(colors) <= available_color:
            colors.append(Color(len(colors)))
        
        # Assign body to color
        colors[available_color].add_body(selected_body)
        colored_count += 1
        
        # Update saturation degrees of adjacent uncolored bodies
        for adjacent_body in selected_body.get_adjacent_bodies():
            if not adjacent_body.is_colored():
                old_saturation = adjacent_body.saturation_degree
                adjacent_body.update_saturation()
                
                # Update priority queue if saturation changed
                if adjacent_body.saturation_degree != old_saturation:
                    pq.update(adjacent_body)
    
    # Calculate chromatic number (number of colors actually used)
    chromatic_number = sum(1 for color in colors if not color.is_empty())
    
    return chromatic_number, colors

def get_color_groups(colors: List[Color]) -> List[List]:
    """
    Convert Color objects to lists of Rigid bodies for easier processing.
    Returns only non-empty color groups.
    """
    color_groups = []
    
    for color in colors:
        if not color.is_empty():
            group = list(color.get_bodies_iterator())
            color_groups.append(group)
    
    return color_groups

def color_physics_graph(solver):
    """
    Main function to color the physics constraint graph and return color groups.
    """
    # Perform DSATUR coloring
    chromatic_number, colors = dsatur_coloring(solver)
    
    # # Verify the coloring is correct
    # if not verify_coloring(colors):
    #     print("ERROR: Invalid coloring detected!")
    #     return None
    
    # # Print results (optional)
    # print_coloring_results(chromatic_number, colors)
    
    # Return color groups for parallel processing
    return get_color_groups(colors)