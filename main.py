import pygame
from pygame.locals import QUIT
from shapes.rigid import Rigid
from solver import Solver
from glm import vec2, vec3
from random import uniform
from shapes.mesh import Mesh
from helper.constants import DRAW_FORCE


BODIES = 50


def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()
    
    # FPS counter variables
    fps_font = pygame.font.Font(None, 36)
    fps_timer = 0
    current_fps = 0

    solver = Solver()
    
    cube_mesh = Mesh([vec2(-0.5, 0.5), vec2(-0.5, -0.5), vec2(0.5, -0.5), vec2(0.5, 0.5)])

    # add playbox
    solver.bodies.append(Rigid(cube_mesh, vec3(0, 0, 0), vec2(30, 0.75), color=vec3(150), density = -1))
    solver.bodies.append(Rigid(cube_mesh, vec3(15.5, 2, 0), vec2(0.75, 5), color=vec3(150), density = -1))
    solver.bodies.append(Rigid(cube_mesh, vec3(-15.5, 2, 0), vec2(0.75, 5), color=vec3(150), density = -1))
    
    # add random bodies
    for _ in range(BODIES):
        solver.bodies.append(Rigid(cube_mesh, vec3(0, 6, 0) + [uniform(-5, 5) for _ in range(3)], vec2([uniform(1, 2) for _ in range(2)]), color=vec3(150), density = 1))

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
            pygame.display.set_caption(f"AVBD2D - {current_fps:.1f} fps")
            fps_timer = 0

        # step physics
        solver.step(dt)

        # draw
        screen.fill((30, 30, 30))
        for body in solver.bodies:
            body.draw(screen)
            
        if DRAW_FORCE:
            for force in solver.forces:
                force.draw(screen)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()