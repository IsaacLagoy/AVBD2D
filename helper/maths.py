import numpy as np

def rotate_scale(angle: float, scale, v):
    s = scale * v
    return rotate(angle, s)

def rotate(angle, v):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype='float32')

def cross(a, b) -> float:
    return a[0] * b[1] - a[1] * b[0]

def inside(p, a, b) -> bool:
    return cross(b - a, p - a) >= 0

def segment_intersect(p1, p2, q1, q2):
    d1 = p2 - p1
    d2 = q2 - q1
    
    denom = cross(d1, d2)
    if abs(denom) < 1e-8:
        return None
    
    t = cross(q1 - p1, d2) / denom
    return p1 + t * d1

def inverse_transform(pos, sca, p):
    local = p - np.array([pos[0], pos[1]], dtype='float32')
    local = rotate(-pos[2], local)
    local /= sca

    return local

def transform(pos, sca, r):
    scaled_r = r * sca
    rotated_r = rotate(pos[2], scaled_r)
    return pos[:2] + rotated_r