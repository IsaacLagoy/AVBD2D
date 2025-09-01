import pygame
from pygame.locals import QUIT
from shapes.rigid import Rigid
from solver import Solver
from glm import vec2, vec3
from random import uniform
from shapes.mesh import Mesh
from helper.constants import DRAW_FORCE
from graph.dsatur import dsatur_coloring
from graph.visuals import get_color


BODIES = 50

def update_body_colors(solver):
    """Update visual colors of all bodies based on their graph coloring"""
    # Perform graph coloring using the new DSATUR algorithm
    color_groups = dsatur_coloring(solver)
    
    if not color_groups:
        print("Warning: Graph coloring failed, using default colors")
        return 0
    
    # Calculate chromatic number (number of colors used)
    chromatic_number = len(color_groups)
    
    # Note: The dsatur_coloring function already assigns visual colors to rigid bodies
    # So we don't need to manually update colors here anymore
    
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
    Rigid(solver.body_system, cube_mesh, vec3(0, 0, 0), vec2(30, 0.75), color=vec3(0.4, 0.4, 0.4), density=-1)
    Rigid(solver.body_system, cube_mesh, vec3(15.5, 2, 0), vec2(0.75, 5), color=vec3(0.4, 0.4, 0.4), density=-1)
    Rigid(solver.body_system, cube_mesh, vec3(-15.5, 2, 0), vec2(0.75, 5), color=vec3(0.4, 0.4, 0.4), density=-1)
    
    # add random bodies
    for _ in range(BODIES):
        Rigid(solver.body_system, cube_mesh, 
              vec3(0, 6, 0) + vec3(uniform(-5, 5), uniform(-5, 5), 0), 
              vec2(uniform(1, 2), uniform(1, 2)), 
              color=vec3(0.6, 0.6, 0.6),  # Default color, will be updated by coloring
              density=1)
    
    # Perform initial graph coloring and update colors
    chromatic_number = 0
    coloring_timer = 0
    coloring_interval = 0
    
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
            
            for rigid in solver.get_bodies_iterator():
                rigid.color = get_color(rigid.graph_color, chromatic_number)
            
        # step physics
        solver.step(max(dt, 1e-4))
        
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