from helper.decorators import timer

import numpy as np
import numba as nb

# @nb.njit(fastmath=True)
def get_far(verts, dir):
    """
    Finds the index of the vertex with the highest dot product with dir
    """
    cur = 0
    here = np.dot(dir, verts[0])
    
    # pick search direction
    roll = np.dot(dir, verts[-1])
    right = np.dot(dir, verts[1])
    
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
    while 0 < cur < l_less:
        next_dot = np.dot(dir, verts[cur + walk])
        
        # next vertex was worse
        if next_dot < here:
            return cur
        
        # keep walking
        cur += walk
        here = next_dot
        
    return cur

# @nb.njit(fastmath=True)
def transform(pos, sr_mat, verts, idx):
    """
    Transforms a given point into world space
    """
    return verts[idx] @ sr_mat + pos[:2]

# @nb.njit(fastmath=True)
def get_support(pos_a, pos_b, sr_a, sr_b, verts_a, verts_b, index_a, index_b, minks, dir, idx):
    """
    Loads the next best support point for dir into the given index
    """
    # find best indices
    index_a[idx] = get_far(verts_a,  dir)
    index_b[idx] = get_far(verts_b, -dir)
    
    # transform selected points to world space
    minks[idx] = transform(pos_a, sr_a, verts_a, index_a[idx]) - transform(pos_b, sr_b, verts_b, index_b[idx]) 
    
# @nb.njit(fastmath=True)
def get_perpendicular(vec, to):
    """
    Get a non-normalized vector that is perpendicular to vec and facing to
    """
    perp = np.array([-vec[1], vec[0]])
    if np.dot(perp, to) >= 0:
        return perp
    return -perp
    
# @nb.njit(fastmath=True)
def handle_0(pos_a, pos_b):
    # NOTE the origin in model space must be enclosed by the polygon
    dir = pos_b[:2] - pos_a[:2]
    
    # check if shapes are overlapping
    if np.allclose(dir, 0):
        dir = np.array([1.0, 0.0])
        
    return dir

# @nb.njit(fastmath=True)
def handle_1(minks):
    return -minks[1]

# @nb.njit(fastmath=True)
def handle_2(minks):
    AB = minks[0] - minks[1]
    AO = -minks[1]
    
    # if near zero, point to AO
    if np.allclose(AB, 0):
        return AO
    
    return get_perpendicular(AB, AO)

# @nb.njit(fastmath=True)
def handle_3(index_a, index_b, minks):
    AB = minks[1] - minks[2]
    AC = minks[0] - minks[2]
    AO = -minks[2]
    
    ab_perp = get_perpendicular(AB, AO)
    if np.dot(ab_perp, AO) > 0 and np.dot(AB, AO) > 0:
        # remove 0, swap 0 with 2 to free up 2 TODO there may be a more clever way to do this without swaps. 
        index_a[0] = index_a[2]
        index_b[0] = index_b[2]
        minks[0] = minks[2]
        
        return ab_perp
    
    ac_perp = get_perpendicular(AC, AO)
    if np.dot(ac_perp, AO) > 0 and np.dot(AC, AO) > 0:
        # remove 1, swap 1 with 2 to free up 2
        index_a[1] = index_a[2]
        index_b[1] = index_b[2]
        minks[1] = minks[2]
        
        return ac_perp
    
    return None
    
# @nb.njit(fastmath=True)
def handle_simplex(pos_a, pos_b, index_a, index_b, minks, free):
    """
    Finds the next best direction for search
    Updates simplex if necessary
    Return next direction and the next free index for the simplex (-1 means hit)
    """
    match free:
        case 0: # empty
            return 0, handle_0(pos_a, pos_b)
        case 1:
            return 1, handle_1(minks)
        case 2: 
            return 2, handle_2(minks)
        case 3: # full
            return 2, handle_3(index_a, index_b, minks)
        case _:
            raise RuntimeError("GJK simplex out of bounds")

# TODO get forward scale and rotate matrix for each body
@timer('GJK', on=True)
def gjk(pos_a, pos_b, sr_a, sr_b, verts_a, verts_b, index_a, index_b, minks, free) -> bool:
    """
    Determines if two convex CCW wound polygons are intersecting
    Updates simplex and returns a collision boolean
    free is the next index that is free in the simplex. 0 if empty. 3 if full. Those should be the only options
    """
    EPSILON = 1e-8
        
    # TODO find amount of iterations for conclusive results
    for _ in range(15):
        # get next search direction
        free, dir = handle_simplex(pos_a, pos_b, index_a, index_b, minks, free)
        if free == -1: 
            return True
        if dir is None: 
            return False
        
        # degenerate direction, return collision
        if np.dot(dir, dir) < EPSILON:
            return True
        
        # add next support point
        get_support(pos_a, pos_b, sr_a, sr_b, verts_a, verts_b, index_a, index_b, minks, dir, free)
        free += 1
        if np.dot(minks[free], dir) < 0: 
            return False
        
    return False