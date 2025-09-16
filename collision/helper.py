import numpy as np
import numba as nb

@nb.njit(inline='always', fastmath=True)
def clamp(i, low, high):
    if i < low:
        return low
    
    if i > high:
        return high
    
    return i

@nb.njit(inline='always', fastmath=True)
def dot(a: np.ndarray, b: np.ndarray) -> np.float32:
    return a[0] * b[0] + a[1] * b[1]

@nb.njit(inline='always', fastmath=True)
def triple_product(a, b, c):
    return dot(a, c) * b - dot(a, b) * c

@nb.njit(inline='always', fastmath=True)
def mat_x_vec(m, v):
    return np.array([
        m[0][0] * v[0] + m[0][1] * v[1],
        m[1][0] * v[0] + m[1][1] * v[1],
    ], dtype='float32')
    
@nb.njit(inline='always', fastmath=True)
def perp_towards(vec, to):
    """
    Returns a perpendicular to 'vec' pointing towards 'to'.
    """
    perp = np.array([-vec[1], vec[0]], dtype=np.float32)  # 90° CCW
    if dot(perp, to) < 0:
        perp = -perp
    return perp

@nb.njit(inline='always', fastmath=True)
def transform(pos, sr_mat, verts, idx):
    """
    Transforms a given point into world space
    """
    return mat_x_vec(sr_mat, verts[idx]) + pos[:2]

@nb.njit(inline='always', fastmath=True)
def transform_direct(pos, sr_mat, vert):
    """
    Transforms a given point into world space
    """
    return mat_x_vec(sr_mat, vert) + pos[:2]

@nb.njit(inline='always', fastmath=True)
def get_far(verts, dir):
    """
    Finds the index of the vertex with the highest dot product with dir
    """
    cur = 0
    here = dot(dir, verts[0])
    
    # pick search direction
    roll = dot(dir, verts[-1])
    right = dot(dir, verts[1])
    
    # early out, already found best index
    if here > roll and here > right:
        return cur
    
    l_less = len(verts) - 1
    
    if roll > right:
        walk = -1
        cur = l_less
        here = roll
    else:
        walk = 1
        cur = 1
        here = right
        
    # walk until we find a worse vertex
    while 0 <= cur <= l_less:
        next_idx = cur + walk
        if not (0 <= next_idx <= l_less):
            return cur  # hit the boundary, must be max
        next_dot = dot(dir, verts[next_idx])
        if next_dot < here:
            return cur
        cur = next_idx
        here = next_dot

        
    return cur