from helper.decorators import timer
from collision.helper import dot, triple_product, mat_x_vec, perp_towards, transform

import numpy as np
import numba as nb

COLLISION_MARGIN = 1e-8

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

@timer('GJK', on=False, us=True)
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

# return collision normal
@timer('EPA', on=False, us=True)
@nb.njit(fastmath=True)
def epa(pos_a, pos_b, sr_a, sr_b, verts_a, verts_b, simplex, faces, sps, normals, dists, set):
    
    # load polytope
    # deep copy 2d array, ensure that this is correct, sps will not modify
    sps[:3] = simplex
    
    build_face(sps, faces, normals, dists, 0, 1, 0)
    build_face(sps, faces, normals, dists, 1, 2, 1)
    build_face(sps, faces, normals, dists, 2, 0, 2)
    
    cloud_size = 3
    size = 3
    
    # NOTE, we never remove support points
    for _ in range(12):
        # get front of polytope
        idx = front(dists, size)
        
        # find new support point
        get_support_mink_only(pos_a, pos_b, sr_a, sr_b, verts_a, verts_b, sps, normals[idx], cloud_size)
        
        # check if newly added point is not in the cloud
        for i in range(0, cloud_size):
            if np.abs(sps[i, 0] - sps[cloud_size, 0]) < COLLISION_MARGIN and \
               np.abs(sps[i, 1] - sps[cloud_size, 1]) < COLLISION_MARGIN:
                   return normals[idx]
                    
        # check that the new found point is past the face
        if dists[idx] > dot(normals[idx], sps[cloud_size]):
            return normals[idx]
        
        # expand the polytope
        set_size = 0
        
        # look at each edge and determine if it is visible
        i = 0
        while i < size:
            if dot(normals[i], sps[cloud_size]) > 0:
                # add vertices to the set of horizons
                set_size = manage_cloud(set, faces[i, 0], set_size)
                set_size = manage_cloud(set, faces[i, 1], set_size)
                
                # remove visible face from the polytope
                remove(normals, dists, faces, i, size)
                size -= 1
            else:
                i += 1
        
        build_face(sps, faces, normals, dists, set[0], cloud_size, size    )
        build_face(sps, faces, normals, dists, set[1], cloud_size, size + 1)
        size += 2
            
        # increment cloud size for adding the new vertex
        cloud_size += 1
        
    # if algorithm didn't naturally terminate, return nearest guess
    idx = front(dists, size)
    print(_)
    return normals[idx]
        
@nb.njit(fastmath=True, inline='always')
def manage_cloud(set, e, size):
    if discard(set, e, size):
        size -= 1
    else:
        set[size] = e
        size += 1
                    
    return size

@nb.njit(fastmath=True, inline='always')
def discard(set, e, size):
    """
    Removes the element with a swap and pop if the element exists
    """
    if size == 0:
        return False
    
    for i in range(size - 1):
        if set[i] == e:
            set[i], set[size - 1] = set[size - 1], set[i]
            return True
            
    # no need to swap but we still need to check if it should be removed
    return set[size - 1] == e
        
@nb.njit(fastmath=True, inline='always')
def front(dists, size):
    """
    Find the "front" most index of the parallel arrays
    """
    best_idx = 0
    best_val = dists[0]
    for i in range(1, size):
        val = dists[i]
        if val < best_val:
            best_val = val
            best_idx = i
    return best_idx
        
@nb.njit(fastmath=True, inline='always')
def remove(normals, dists, faces, idx, size):
    """
    Removes the element at the given index with a swap and pop
    """
    normals[idx], normals[size - 1] = normals[size - 1], normals[idx]
    dists[idx], dists[size - 1] = dists[size - 1], dists[idx]
    faces[idx], faces[size - 1] = faces[size - 1], faces[idx]
    
@nb.njit(fastmath=True)
def get_support_mink_only(pos_a, pos_b, sr_a, sr_b, verts_a, verts_b, sps, dir, idx):
    """
    Loads the next best support point for dir into the given index
    This function does not track the witness indices from each body
    """
    # find best indices
    dir_a = mat_x_vec(sr_a.T,  dir)
    dir_b = mat_x_vec(sr_b.T, -dir)
    
    idx_a = get_far(verts_a,  dir_a)
    idx_b = get_far(verts_b, dir_b)
    
    # transform selected points to world space
    sps[idx] = transform(pos_a, sr_a, verts_a, idx_a) - transform(pos_b, sr_b, verts_b, idx_b) 
    
@nb.njit(fastmath=True, inline='always')
def build_face(sps, faces, normals, dists, idx_a, idx_b, idx_log):
    faces[idx_log, 0] = idx_a
    faces[idx_log, 1] = idx_b

    # Edge vector
    ex = sps[idx_b, 0] - sps[idx_a, 0]
    ey = sps[idx_b, 1] - sps[idx_a, 1]

    # Perpendicular facing sps[idx_a]
    nx = -ey 
    ny = ex
    if nx * sps[idx_a, 0] + ny * sps[idx_a, 1] < 0:
        nx = -nx
        ny = -ny
        
    len2 = nx * nx + ny * ny

    # sqaure normalize the normal vector
    normals[idx_log, 0] = nx / len2
    normals[idx_log, 1] = ny / len2

    # squared distance for comparisons
    n_dot_a = normals[idx_log, 0] * sps[idx_a, 0] + normals[idx_log, 1] * sps[idx_a, 1] 
    dists[idx_log] = n_dot_a
    

def sat(pos_a, pos_b, sr_a, sr_b, verts_a, verts_b, mtv):
    """
    Returns the starting indices of each edge 
    """
    # get the MTV in local space
    mtv_la = transform(sr_a.T, mtv)
    mtv_lb = transform(sr_b.T, mtv)
    
    