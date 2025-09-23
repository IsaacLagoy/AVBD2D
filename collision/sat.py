import numpy as np
import numba as nb
from collision.helper import transform, transform_direct, dot, mat_x_vec, clamp
from helper.constants import COLLISION_MARGIN



# TODO create idcs and world buffers externally
# TODO pass contact rA and rB as reference
@nb.njit(fastmath=True)
def sat(pos_a, pos_b, rs_a, rs_b, ir_a, ir_b, verts_a, verts_b, mtv, idcs, world, rA, rB):
    """
    Returns the starting indices of each edge 
    mtv should be normalized
    """
    
    # print('-------- SAT ---------')
    
    # # get the MTV in local space
    # mtv_la = mat_x_vec(rs_a.T, mtv)
    # mtv_lb = mat_x_vec(rs_b.T, mtv)
    
    # get world location of vertices
    t_verts_a = np.empty((len(verts_a), 2), dtype='float32')
    for i in range(len(verts_a)):
        t_verts_a[i] = mat_x_vec(rs_a, verts_a[i]) + pos_a[:2]
    
    t_verts_b = np.empty((len(verts_b), 2), dtype='float32')
    for i in range(len(verts_b)):
        t_verts_b[i] = mat_x_vec(rs_b, verts_b[i]) + pos_b[:2]
    
    # take the dot product of all vertices with their "local" vector
    dots_a = np.empty(len(verts_a), dtype='float32')
    for i in range(len(verts_a)):
        dots_a[i] = dot(t_verts_a[i], mtv)
        
    dots_b = np.empty(len(verts_b), dtype='float32')
    for i in range(len(verts_b)):
        dots_b[i] = dot(t_verts_b[i], mtv)
    
    # # should point in direction of mtv
    # dx = pos_b[:2] - pos_a[:2] # NOTE because of checked EPA output, dx dot mtv is always negative
    
    # # reference is A
    # dx = mat_x_vec(ir_a, dx)
    # dx = dot(dx, mtv)
    # dots_b += dx
    
    intersect(pos_b, rs_b, t_verts_b, dots_b, pos_a, rs_a, t_verts_a, dots_a, mtv, idcs, world, rB, rA)
       
@nb.njit(fastmath=True, inline='always')
def intersect(pos_ref, rs_ref, verts_ref, dots_ref, pos_inc, rs_inc, verts_inc, dots_inc, mtv, idcs, world, r_ref, r_inc):
    # idcs breakdown
    # max      ref - [0, 1]
    # edg strt ref - [2, 3]
    # min      inc - [4, 5]
    # edg strt inc - [6, 7]

    # rules
    # [2, 3] and [6, 7] cannot be the same! Degenerate
    
    print('Dots Ref')
    print(dots_ref)
    print('Dots Inc')
    print(dots_inc)
    
    # find extreme values
    max_ref = find_maxs(dots_ref, idcs[0:2])
    min_inc = find_mins(dots_inc, idcs[4:6])
    
    print('(inc above) (ref below) thresh')
    print(max_ref, min_inc)
    
    # find bounding values
    find_max_bounds(dots_ref, idcs[2:4], min_inc)
    find_min_bounds(dots_inc, idcs[6:8], max_ref)
    
    # load world locations
    world[0] = verts_ref[idcs[0]]
    world[1] = verts_ref[idcs[1]]
    world[4] = verts_inc[idcs[4]]
    world[5] = verts_inc[idcs[5]]
    
    # find all local space edge intersections
    if idcs[2] != idcs[3]:
        world[2] = dot_edge_intersect(idcs[2], verts_ref, dots_ref, min_inc, mtv)
        world[3] = dot_edge_intersect(idcs[3], verts_ref, dots_ref, min_inc, mtv)
    
    if idcs[6] != idcs[7]:
        world[6] = dot_edge_intersect(idcs[6], verts_inc, dots_inc, max_ref, mtv)
        world[7] = dot_edge_intersect(idcs[7], verts_inc, dots_inc, max_ref, mtv)
        
    # clamp is it is a non-inclusive collision
    if idcs[6] != idcs[7]:
        edge_clamp(world[6:8], world[0:2]) # clamp min inc to edge ref
        
    if idcs[2] != idcs[3]:
        edge_clamp(world[2:4], world[4:6]) # clamp max ref to edge inc
    
    # convert all points to worlds space
    # transform_quad(rs_ref, pos_ref, world[:4], verts_ref, idcs[:4])
    # transform_quad(rs_inc, pos_inc, world[4:], verts_inc, idcs[4:])

    # print('clamp done')
    
    # add contacts to r lists
    # TODO check contact ordering
    r_rs_inv = np.linalg.inv(rs_ref)
    i_rs_inv = np.linalg.inv(rs_inc)
    
    r_ref[0] = mat_x_vec(r_rs_inv, world[0] - pos_ref[:2]) 
    r_ref[1] = mat_x_vec(r_rs_inv, world[1] - pos_ref[:2])
    
    r_inc[0] = mat_x_vec(i_rs_inv, world[4] - pos_inc[:2])
    r_inc[1] = mat_x_vec(i_rs_inv, world[5] - pos_inc[:2])
    
    
@nb.njit(fastmath=True, inline='always')
def transform_quad(rs, pos, worlds, verts, idcs):
    # NOTE worlds 0 and 3 will be in local space
    # worlds 1 and 2 will not be defined and need to be gathered from idcs
    
    # transform edge intresections
    worlds[0] = transform_direct(pos, rs, worlds[0])
    worlds[3] = transform_direct(pos, rs, worlds[3])
    
    # transform maximum vertices, upper half 
    worlds[1] = transform(pos, rs, verts, idcs[2])
    worlds[2] = transform(pos, rs, verts, idcs[3])
    
@nb.njit(fastmath=True, inline='always')
def edge_clamp(e, m):
    """
    Clamp the segment m0, m1 to e0, e1
    """
    print('clamping')
    print(e)
    print(m)
    
    v = e[1] - e[0]
    vv = dot(v, v)
    
    # parametric coordinates of m relative to e
    t0 = dot(m[0] - e[0], v) / vv
    t1 = dot(m[1] - e[0], v) / vv
    
    # clamp to [0, 1]
    t0 = clamp(t0, 0.0, 1.0)
    t1 = clamp(t1, 0.0, 1.0)
    
    # reconstruct clamped points
    m[0] = e[0] + t0 * v
    m[1] = e[0] + t1 * v
    
@nb.njit(fastmath=True, inline='always')
def dot_edge_intersect(start, verts, dots, thresh, mtv):
    """
    Returns the local space of the edge with dot intersection
    """
    # get end index of the edge
    end = start + 1 if start < len(verts) - 1 else 0
    
    # get references to variables
    P1 = verts[start]
    P2 = verts[end]
    
    l1 = dots[start]
    l2 = dots[end]
    
    # print(l1, l2, start, end)
    # print(dots)
    # print(verts)
    # print(mtv)
    
    t = (thresh - l1) / (l2 - l1)
    
    # clamp to avoid errors
    t = clamp(t, 0.0, 1.0)
    return P1 + t * (P2 - P1)
        
# -----------------------------
# Reference geometry functions
# -----------------------------

@nb.njit(fastmath=True, inline='always')
def find_maxs(arr, idcs) -> float:
    # constants
    l = len(arr)
    
    m = arr[0]
    idcs[0] = 0
    idcs[1] = 0
    
    for i in range(1, l):
        c = arr[i]
        
        # check if current value exceeds maximum
        if c > m + COLLISION_MARGIN:
            idcs[0] = i
            idcs[1] = i
            m = c
            
        # check if current value matches maximum
        elif abs(c - m) <= COLLISION_MARGIN:
            idcs[1] = i
            
    return m

@nb.njit(fastmath=True, inline='always')
def find_max_bounds(arr, idcs, thresh) -> None:
    # constants
    l = len(arr)
    
    # loop checks
    is_in = arr[0] >= thresh
    
    # final checks
    end_found = False
    begin_found = False
    
    for i in range(1, l):
        c = arr[i]

        if c >= thresh:
            
            # we are entering the new threshold
            if not is_in:
                idcs[0] = i - 1
                is_in = True
                begin_found = True
                
            continue
        
        # below threshold
        # check to see if we are leaving
        if is_in:
            is_in = False
            idcs[1] = i - 1
            end_found = True
            
    # edge case - end is at 0 [----++++]
    if not end_found:
        idcs[1] = l - 1
        
    # edge case - beginning is at 0 [++++----]
    # element before begin is -1 
    if not begin_found:
        idcs[0] = l - 1

# -----------------------------
# Incident geometry functions
# -----------------------------

@nb.njit(fastmath=True, inline='always')
def find_mins(arr, idcs) -> float:
    # constants
    l = len(arr)
    
    m = arr[0]
    idcs[0] = 0
    idcs[1] = 0
    
    for i in range(1, l):
        c = arr[i]
        
        # check if current value exceeds maximum
        if c + COLLISION_MARGIN < m:
            idcs[0] = i
            idcs[1] = i
            m = c
            
        # check if current value matches maximum
        elif abs(c - m) <= COLLISION_MARGIN:
            idcs[1] = i
            
    return m

@nb.njit(fastmath=True, inline='always')
def find_min_bounds(arr, idcs, thresh) -> None:
    # constants
    l = len(arr)
    
    # loop checks
    is_in = arr[0] <= thresh
    
    # final checks
    end_found = False
    begin_found = False
    
    for i in range(1, l):
        c = arr[i]

        if c <= thresh:
            
            # we are entering the new threshold
            if not is_in:
                idcs[0] = i - 1
                is_in = True
                begin_found = True
                
            continue
        
        # below threshold
        # check to see if we are leaving
        if is_in:
            is_in = False
            idcs[1] = i - 1
            end_found = True
            
    # edge case - end is at 0 [----++++]
    if not end_found:
        idcs[1] = l - 1
        
    # edge case - beginning is at 0 [++++----]
    # element before begin is -1 
    if not begin_found:
        idcs[0] = l - 1