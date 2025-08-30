import random
import glm
from glm import vec2
from shapes.mesh import Mesh


def get_convex(num_points: int = 6, max_dim: float = 3.0) -> Mesh:
    """
    Generate a random convex polygon with counter-clockwise vertices.
    The polygon fits inside a bounding box with dimension <= max_dim.
    """
    # Step 1: random points in square
    pts = [vec2(random.uniform(-max_dim/2, max_dim/2),
                random.uniform(-max_dim/2, max_dim/2))
           for _ in range(num_points * 3)]  # oversample to get good hull

    # Step 2: convex hull (Andrew's monotone chain)
    pts = sorted(pts, key=lambda p: (p.x, p.y))

    def cross(o, a, b):
        return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]  # counter-clockwise

    # Step 3: ensure within max_dim box
    xs = [p.x for p in hull]
    ys = [p.y for p in hull]
    width, height = max(xs) - min(xs), max(ys) - min(ys)
    scale = min(1.0, max_dim / max(width, height))
    hull = [p * scale for p in hull]

    return Mesh(hull)
