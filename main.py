import pygame
from pygame.locals import QUIT
from shapes.rigid import Rigid
from solver import Solver
from glm import vec2, vec3
from random import uniform
from shapes.mesh import Mesh
from helper.constants import DRAW_FORCE
from graph.dsatur import color_physics_graph
from graph.visuals import get_color  # Assume this function is available

BODIES = 50

def update_body_colors(solver):
    """Update visual colors of all bodies based on their graph coloring"""
    # Perform graph coloring
    color_groups = color_physics_graph(solver)
    
    if not color_groups:
        print("Warning: Graph coloring failed, using default colors")
        return 0
    
    # Calculate chromatic number
    chromatic_number = len(color_groups)
    
    # Update visual colors for all bodies
    current = solver.bodies
    while current is not None:
        if hasattr(current, 'graph_color') and current.graph_color != -1:
            # Get RGB color based on graph color
            rgb_color = get_color(current.graph_color, chromatic_number)
            current.color = vec3(rgb_color[0], rgb_color[1], rgb_color[2])
        else:
            # Fallback color for uncolored bodies
            current.color = vec3(128, 128, 128)  # Gray
        current = current.next
    
    return chromatic_number

def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()
    
    # FPS counter variables
    fps_timer = 0
    current_fps = 0
    solver = Solver()
    
    cube_mesh = Mesh([vec2(-0.5, 0.5), vec2(-0.5, -0.5), vec2(0.5, -0.5), vec2(0.5, 0.5)])
    
    # add playbox (static bodies)
    Rigid(solver, cube_mesh, vec3(0, 0, 0), vec2(30, 0.75), color=vec3(100, 100, 100), density=-1)
    Rigid(solver, cube_mesh, vec3(15.5, 2, 0), vec2(0.75, 5), color=vec3(100, 100, 100), density=-1)
    Rigid(solver, cube_mesh, vec3(-15.5, 2, 0), vec2(0.75, 5), color=vec3(100, 100, 100), density=-1)
    
    # add random bodies
    for _ in range(BODIES):
        Rigid(solver, cube_mesh, 
              vec3(0, 6, 0) + vec3(uniform(-5, 5), uniform(-5, 5), uniform(-5, 5)), 
              vec2(uniform(1, 2), uniform(1, 2)), 
              color=vec3(150, 150, 150),  # Default color, will be updated
              density=1)
    
    # Perform initial graph coloring and update colors
    chromatic_number = 0
    coloring_timer = 0
    coloring_interval = 1.0  # Recolor every 1 second
    
    running = True
    while running:
        dt = clock.tick() / 1000.0  # seconds
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
                
        # Update FPS counter every 0.5 seconds
        fps_timer += dt
        if fps_timer >= 0.5:
            current_fps = clock.get_fps()
            title = f"AVBD2D - {current_fps:.1f} fps"
            if chromatic_number > 0:
                title += f" - Colors: {chromatic_number}"
            pygame.display.set_caption(title)
            fps_timer = 0
        
        # Update graph coloring periodically
        coloring_timer += dt
        if coloring_timer >= coloring_interval:
            chromatic_number = update_body_colors(solver)
            coloring_timer = 0
            print(f"Graph recolored with {chromatic_number} colors")
            
        # step physics
        solver.step(max(dt, 1e-8))
        
        # draw
        screen.fill((30, 30, 30))
        
        # Draw all bodies using linked list iterator
        for body in solver.get_bodies_iterator():
            body.draw(screen)
            
        # Draw all forces using linked list iterator (if enabled)
        if DRAW_FORCE:
            for force in solver.get_forces_iterator():
                force.draw(screen)
                
        pygame.display.flip()
        
    pygame.quit()

if __name__ == "__main__":
    main()