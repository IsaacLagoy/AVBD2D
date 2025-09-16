from collision.helper import dot, triple_product, mat_x_vec, perp_towards, transform, get_far
from helper.constants import COLLISION_MARGIN

import numpy as np
import numba as nb

@nb.njit(fastmath=True)
def get_support(pos_a, pos_b, sr_a, sr_b, verts_a, verts_b, index_a, index_b, minks, dir, idx):
    """
    Loads the next best support point for dir into the given index
    """
    # find best indices
    dir_a = mat_x_vec(sr_a.T,  dir)
    dir_b = mat_x_vec(sr_b.T, -dir)
    
    index_a[idx] = get_far(verts_a,  dir_a)
    index_b[idx] = get_far(verts_b, dir_b)
    
    # transform selected points to world space
    minks[idx] = transform(pos_a, sr_a, verts_a, index_a[idx]) - transform(pos_b, sr_b, verts_b, index_b[idx]) 
    
@nb.njit(fastmath=True)
def handle_0(pos_a, pos_b):
    # NOTE the origin in model space must be enclosed by the polygon
    dir = pos_b[:2] - pos_a[:2]
    
    # check if shapes are overlapping
    if np.allclose(dir, 0):
        dir = np.array([1.0, 0.0], dtype=pos_a.dtype)
        
    return dir

@nb.njit(fastmath=True)
def handle_1(minks):
    return -minks[0]

@nb.njit(fastmath=True)
def handle_2(minks):
    CB = minks[1] - minks[0]
    CO =          - minks[0]
    
    # TODO Add robust same direction case
    
    return triple_product(CB, CO, CB)

@nb.njit(fastmath=True)
def handle_3(index_a, index_b, minks):
    AB = minks[1] - minks[2]
    AC = minks[0] - minks[2]
    AO =          - minks[2]
    
    ab_perp = perp_towards(AB, -minks[0])
    if dot(ab_perp, AO) > -COLLISION_MARGIN:
        # remove 0
        index_a[0] = index_a[2]
        index_b[0] = index_b[2]
        minks[0] = minks[2]
        
        return 2, ab_perp
    
    ac_perp = perp_towards(AC, -minks[1])
    if dot(ac_perp, AO) > -COLLISION_MARGIN:
        # remove 1
        index_a[1] = index_a[2]
        index_b[1] = index_b[2]
        minks[1] = minks[2]
        
        return 2, ac_perp
    
    return -1, AO # we return AO but it will not be used, dummy return value for njit
    
@nb.njit(fastmath=True)
def handle_simplex(pos_a, pos_b, index_a, index_b, minks, free):
    """
    Finds the next best direction for search
    Updates simplex if necessary
    Return next direction and the next free index for the simplex (-1 means hit)
    """
    if free == 0:
        return 0, handle_0(pos_a, pos_b)
    if free == 1:
        return 1, handle_1(minks)
    if free == 2:
        return 2, handle_2(minks)
    if free == 3:
        return handle_3(index_a, index_b, minks)
    else:
        raise RuntimeError("GJK simplex out of bounds")

@nb.njit(fastmath=True)
def gjk(pos_a, pos_b, sr_a, sr_b, verts_a, verts_b, index_a, index_b, minks, free) -> bool:
    """
    Determines if two convex CCW wound polygons are intersecting
    Updates simplex and returns a collision boolean
    free is the next index that is free in the simplex. 0 if empty. 3 if full. Those should be the only options
    """
        
    # TODO find amount of iterations for conclusive results
    for _ in range(15):
        # get next search direction
        free, dir = handle_simplex(pos_a, pos_b, index_a, index_b, minks, free)
        if free == -1: 
            return True
        
        # add next support point
        get_support(pos_a, pos_b, sr_a, sr_b, verts_a, verts_b, index_a, index_b, minks, dir, free)
        if dot(minks[free], dir) < COLLISION_MARGIN: 
            return False
        
        free += 1
        
    return False