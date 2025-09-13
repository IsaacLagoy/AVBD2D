import numpy as np
import numba as nb

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