import pygame
import numpy as np
import math

# Initialize pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Normal Transformation Demo - Inverse Transpose vs Regular")
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)

class RigidBody:
    def __init__(self, x, y):
        self.pos = np.array([x, y])
        self.rotation = 0.0
        self.scale = np.array([1.0, 1.0])
        
        # Create a simple rectangle with normals pointing outward
        self.local_vertices = np.array([
            [-50, -30],  # bottom-left
            [50, -30],   # bottom-right
            [50, 30],    # top-right
            [-50, 30]    # top-left
        ])
        
        # Normals for each edge (pointing outward)
        self.local_normals = np.array([
            [0, -1],   # bottom edge normal (pointing down)
            [1, 0],    # right edge normal (pointing right)
            [0, 1],    # top edge normal (pointing up)
            [-1, 0]    # left edge normal (pointing left)
        ])
        
        # Edge midpoints for drawing normals
        self.edge_midpoints = np.array([
            [0, -30],   # bottom edge midpoint
            [50, 0],    # right edge midpoint
            [0, 30],    # top edge midpoint
            [-50, 0]    # left edge midpoint
        ])
    
    def get_matrices(self):
        # Rotation matrix
        cos_r = math.cos(self.rotation)
        sin_r = math.sin(self.rotation)
        rot_mat = np.array([
            [cos_r, -sin_r],
            [sin_r, cos_r]
        ])
        
        # Scale matrix
        sca_mat = np.array([
            [self.scale[0], 0],
            [0, self.scale[1]]
        ])
        
        return rot_mat, sca_mat
    
    def transform_vertices(self):
        rot_mat, sca_mat = self.get_matrices()
        transform_mat = rot_mat @ sca_mat
        
        # Transform vertices (regular transformation)
        world_vertices = self.local_vertices @ transform_mat.T + self.pos
        world_midpoints = self.edge_midpoints @ transform_mat.T + self.pos
        
        return world_vertices, world_midpoints
    
    def transform_normals_correct(self):
        """Transform normals using the correct inverse transpose method"""
        rot_mat, sca_mat = self.get_matrices()
        transform_mat = rot_mat @ sca_mat
        
        # Correct normal transformation: (M^-1)^T
        normal_transform = np.linalg.inv(transform_mat).T
        world_normals = self.local_normals @ normal_transform.T
        
        return world_normals
    
    def transform_normals_wrong(self):
        """Transform normals incorrectly (for comparison)"""
        rot_mat, sca_mat = self.get_matrices()
        transform_mat = rot_mat @ sca_mat
        
        # Wrong normal transformation: just apply the regular transformation
        world_normals = self.local_normals @ transform_mat.T
        
        return world_normals

def draw_normals(screen, positions, normals, color, scale=50):
    """Draw normal vectors at given positions"""
    for i, (pos, normal) in enumerate(zip(positions, normals)):
        start_pos = pos.astype(int)
        end_pos = (pos + normal * scale).astype(int)
        
        # Draw the normal vector
        pygame.draw.line(screen, color, start_pos, end_pos, 2)
        # Draw arrowhead
        pygame.draw.circle(screen, color, end_pos, 3)

def draw_text(screen, text, pos, color=BLACK):
    """Draw text on screen"""
    font = pygame.font.Font(None, 24)
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, pos)

def main():
    running = True
    body = RigidBody(WIDTH // 2, HEIGHT // 2)
    
    # Animation parameters
    time = 0
    
    while running:
        dt = clock.tick(60) / 1000.0
        time += dt
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Animate the rigid body
        body.rotation = time * 0.5
        body.scale[0] = 1.0 + 0.5 * math.sin(time * 2)  # Non-uniform scaling
        body.scale[1] = 1.0 + 0.3 * math.cos(time * 1.5)
        
        # Clear screen
        screen.fill(WHITE)
        
        # Transform everything
        world_vertices, world_midpoints = body.transform_vertices()
        correct_normals = body.transform_normals_correct()
        wrong_normals = body.transform_normals_wrong()
        
        # Draw the shape
        pygame.draw.polygon(screen, GRAY, world_vertices, 2)
        
        # Draw correct normals (green)
        draw_normals(screen, world_midpoints, correct_normals, GREEN, scale=40)
        
        # Draw incorrect normals (red) - offset slightly for visibility
        offset_positions = world_midpoints + np.array([10, 10])
        draw_normals(screen, offset_positions, wrong_normals, RED, scale=40)
        
        # Draw legend
        draw_text(screen, "Green: Correct normals (inverse transpose)", (10, 10), GREEN)
        draw_text(screen, "Red: Incorrect normals (regular transform)", (10, 35), RED)
        draw_text(screen, "Gray: Shape outline", (10, 60), GRAY)
        draw_text(screen, f"Scale: ({body.scale[0]:.2f}, {body.scale[1]:.2f})", (10, 85))
        draw_text(screen, f"Rotation: {math.degrees(body.rotation):.1f}°", (10, 110))
        
        # Instructions
        draw_text(screen, "Notice how red normals become skewed with non-uniform scaling", (10, HEIGHT - 40))
        draw_text(screen, "Green normals remain perpendicular to the surface", (10, HEIGHT - 20))
        
        pygame.display.flip()
    
    pygame.quit()

if __name__ == "__main__":
    main()