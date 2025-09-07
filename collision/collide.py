from shapes.rigid import Rigid
import numpy as np
from helper.decorators import timer
import numba as nb


@timer('Collision', on=True)
# @nb.njit(fastmath=True)
def sat(pos_a, pos_b, irs_a, norms_a, s_b, ir_b, verts_b, dots_a):    
    """This function runs a vectorized and optimized version of the Seperating Axis Theorem (SAT). This function only runs SAT from the perspective of the reference object, it will have to be run again with the arguments reversed to complete a full SAT. 

    Args:
        body_a (Rigid): This is the reference rigid body
        body_b (Rigid): This is the incident rigid body

    Returns:
        row (int): This is the index of the normal used to find the minimum axis of penetration on body_a
        col (int): This is the index of the most penetrating vertex on body_b
        dots (np.ndarray): An array containing all of the dot products of body_b.vertices with body_a.normals[row]
        delta_pos_minus_dot_a (float): dot products greater than this value are outside of body_a. 
    """    
    # vector from A to B in world space
    delta_pos_world = pos_b[:2] - pos_a[:2]

    # convert to local space of body_a
    delta_pos_local_a = delta_pos_world @ irs_a.T
    
    # prune normals that have a negative dot product
    fltrd_idcs = np.where(np.dot(norms_a, delta_pos_local_a) > 0)[0]
    fltrd_norms = norms_a[fltrd_idcs]
    
    # find local A dot product between difference in position and each filtered normal
    delta_pos_dots = np.dot(fltrd_norms, delta_pos_local_a)
    
    # subtract A's influence in each normal direction
    delta_pos_dots -= dots_a[fltrd_idcs] # contains the highest dot product of A's vertices with each of its normals in A's local space

    mat = s_b @ ir_b @ irs_a.T
    
    # transform normals to be in b's local space but are scaled inversely. If A is larger then the dot product contribution should be proportionally less
    norms_r_sinv = fltrd_norms @ mat.T
    
    # create a matrix containing the dot products of transformed normals and every vertex in B's local space
    # each row contains the dot products from all vertices with a single normal
    # these dot products are taken in B's local space but the normals should be rotated forward and scaled backward so that they are proportional in A's local space
    r_sinv_dots = np.dot(norms_r_sinv, verts_b.T)
    
    # select the minimum dot product in each row, this is the penetration value for object b
    max_pen_col_idcs = np.argmin(r_sinv_dots, axis=1)
    max_pen_vals = r_sinv_dots[np.arange(r_sinv_dots.shape[0]), max_pen_col_idcs]
    
    # find overlap on all axes in r sinv space
    max_pen_vals += delta_pos_dots
    
    # find the penetration with the maximum value = minimum penetration
    min_pen_row_idx_fltrd = np.argmax(max_pen_vals)
    minmax_pen_val = max_pen_vals[min_pen_row_idx_fltrd]
    
    # determine if there is overlap and break if not
    if minmax_pen_val >= 0:
        return None
    
    # find the unfiltered index of the minimum penetration normal
    min_pen_row_idx = fltrd_idcs[min_pen_row_idx_fltrd]
    
    # find the vertex index of the maximally penetrating vertex
    minmax_pen_col_idx = max_pen_col_idcs[min_pen_row_idx_fltrd]
    
    # extract the row of the most penetrating vertex
    minmax_pen_row_dots = r_sinv_dots[min_pen_row_idx_fltrd]
    
    # find the delta_pos - argmax(A) that corresponds with the edge of A
    minmax_delta_pos_dot = delta_pos_dots[min_pen_row_idx_fltrd]
    
    return min_pen_row_idx, minmax_pen_col_idx, minmax_pen_row_dots, minmax_delta_pos_dot
    
def collide(manifold) -> bool:
    body_a: Rigid = manifold.body_a
    body_b: Rigid = manifold.body_b
    
    data_a = sat(body_a.pos, body_b.pos, body_a.inv_rot_sca_mat, body_a.mesh.normals, body_a.sca_mat, np.linalg.inv(body_b.rot_mat), body_b.mesh.vertices, body_a.mesh.dots)
    data_b = sat(body_b.pos, body_a.pos, body_b.inv_rot_sca_mat, body_b.mesh.normals, body_b.sca_mat, np.linalg.inv(body_a.rot_mat), body_a.mesh.vertices, body_b.mesh.dots)
    
    # check for no collision
    if data_a is None or data_b is None:
        return 0
    
    body_a.color = (255, 0, 0)
    body_b.color = (255, 0, 0)
    
    return 0
    
    # no penetration from rigid b
    if data_b is None:
        norm_idx, vert_idx, dots, delta = data_a
        reference: Rigid = manifold.body_a
        incident: Rigid = manifold.body_b

    # no penetration from object a
    elif data_a is None:
        norm_idx, vert_idx, dots, delta = data_b
        reference: Rigid = manifold.body_b
        incident: Rigid = manifold.body_a
        
    # both a and b must have collided
    # select data from smallest penetration (least negative)
    elif data_a[2][data_a[1]] > data_b[2][data_b[1]]:
        norm_idx, vert_idx, dots, delta = data_a
        reference: Rigid = manifold.body_a
        incident: Rigid = manifold.body_b
        
    else:
        norm_idx, vert_idx, dots, delta = data_b
        reference: Rigid = manifold.body_b
        incident: Rigid = manifold.body_a
        

    
    return 0
        
    # collect the necessary data
    normal_local = reference[norm_idx]
    normal = (reference.rot @ reference.scale).T_inv @ normal_local
    
    # there are always 2 contact points
    # broadcast assign each normal to be the same value
    manifold.normal[:] = normal
    
    # identify intersecting edges
    insides = dots < delta
    diff = np.diff(insides.astype(int))
    trans = np.where(diff != 0)[0]
    
    if len(trans) == 1:
        trans = np.append(trans, len(diff))
    
    # convert selected points on incident object to reference space
    # even indices are one edge, odd indices are another
    trans = np.concatenate([trans, (trans + 1) % len(insides)])
    edge_verts = incident.vertices[trans]
    
    # construct transformation
    inc_mat = incident.rot_mat @ incident.sca_mat
    ref_mat = reference.rot_mat @ reference.sca_mat
    
    # project to world space
    edge_verts = inc_mat @ edge_verts + incident.pos[:2] # pos[:2] is only x y translational
    edge_verts = np.linalg.inv(ref_mat) @ (edge_verts - reference.pos[:2])
    
    
    
    assert len(trans) == 2, f'Wrong number of contact edges: {len(trans)}'
    
    
    
