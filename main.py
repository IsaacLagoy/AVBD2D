import pygame
from pygame.locals import QUIT
from rigid import Rigid
from solver import Solver
from glm import vec2, vec3
from random import uniform
from mesh import Mesh


def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()

    solver = Solver()
    
    cube_mesh = Mesh([vec2(-0.5, 0.5), vec2(-0.5, -0.5), vec2(0.5, -0.5), vec2(0.5, 0.5)])

    # add some bodies
    # solver.bodies.append(Rigid(vec3(0, 5, 1), vec2(0.5, 0.5), color=(200, 50, 50)))
    # solver.bodies.append(Rigid(vec3(2, 10, 0), vec2(1, 0.5), color=(50, 200, 50)))
    # solver.bodies.append(Rigid(vec3(-2, 15, 0), vec2(0.75, 0.75), color=(50, 80, 200)))
    solver.bodies.append(Rigid(cube_mesh, vec3(0, 0, 0), vec2(5, 0.75), color=(50, 50, 50), density = -1))
    solver.bodies.append(Rigid(cube_mesh, vec3(0, 1, 0) + vec3([uniform(0, 0) for _ in range(3)]), vec2([uniform(1, 2) for _ in range(2)]), color=(50, 80, 200), density = 1))

    running = True
    while running:
        dt = clock.tick(60) / 1000.0  # seconds

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

        # step physics
        solver.step(dt / 10)

        # draw
        screen.fill((30, 30, 30))
        for body in solver.bodies:
            body.draw(screen)
            
        print('num forces', len(solver.forces))
        for body in solver.bodies:
            print(len(body.forces))
            
        for force in solver.forces:
            force.draw(screen)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()