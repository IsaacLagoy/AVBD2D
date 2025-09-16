import numpy as np
import numba as nb
from helper.constants import COLLISION_MARGIN
from collision.helper import mat_x_vec, transform, get_far, dot


# return collision normal
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