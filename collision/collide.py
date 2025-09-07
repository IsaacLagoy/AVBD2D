from shapes.rigid import Rigid
import numpy as np
from helper.decorators import timer
import numba as nb
import pygame
from glm import vec2, vec3
import glm


# @timer('Collision', on=True)
# @nb.njit(fastmath=True)
def sat(screen, body_a: Rigid, body_b: Rigid):    
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
    # print('\n', body_a.scale, body_b.scale)
    
    # vector from A to B in world space
    delta_pos_world = body_b.pos[:2] - body_a.pos[:2]
    
    # print('delta_pos_world', delta_pos_world)

    # convert to local space of body_a
    delta_pos_local_a = delta_pos_world @ np.linalg.inv(body_a.rot_mat @ body_a.sca_mat).T
    
    # print('delta_pos_local_a', delta_pos_local_a)
    # print('body_a.inv_rot_sca_mat', body_a.inv_rot_sca_mat)
    
    # prune normals that have a negative dot product
    fltrd_idcs = np.where(np.dot(body_a.mesh.normals, delta_pos_local_a) > 0)[0]
    fltrd_norms = body_a.mesh.normals[fltrd_idcs]
    
    # print('selected norms', fltrd_norms)
    
    # find local A dot product between difference in position and each filtered normal
    delta_pos_dots = np.dot(fltrd_norms, delta_pos_local_a)
    
    # print('pos_dots', delta_pos_dots)
    
    # subtract A's influence in each normal direction
    delta_pos_dots -= body_a.mesh.dots[fltrd_idcs] # contains the highest dot product of A's vertices with each of its normals in A's local space
    
    # print('dp_dots', delta_pos_dots)
    
    mat_a = np.linalg.inv(body_a.rot_mat @ body_a.sca_mat).T  # Normal transform: a local -> world (inverse scale)
    mat_b_rot = np.linalg.inv(body_b.rot_mat)  # Rotate from world -> b's local space
    mat_b_scale = body_b.sca_mat  # Forward scale in b's local space

    mat = mat_b_scale @ mat_b_rot @ mat_a
    
    # transform normals to be in b's local space but are scaled inversely. If A is larger then the dot product contribution should be proportionally less
    norms_r_sinv = fltrd_norms @ mat.T
    
    # print()
    
    # print('trans_norms',norms_r_sinv)
    
    # create a matrix containing the dot products of transformed normals and every vertex in B's local space
    # each row contains the dot products from all vertices with a single normal
    # these dot products are taken in B's local space but the normals should be rotated forward and scaled backward so that they are proportional in A's local space
    r_sinv_dots = np.dot(norms_r_sinv, body_b.mesh.vertices.T)
    # print('dots', r_sinv_dots)
    
    # select the minimum dot product in each row, this is the penetration value for object b
    max_pen_col_idcs = np.argmin(r_sinv_dots, axis=1)
    max_pen_vals = r_sinv_dots[np.arange(r_sinv_dots.shape[0]), max_pen_col_idcs]
    
    # print('max_pen_vals', max_pen_vals)
    
    # find overlap on all axes in r sinv space
    max_pen_vals += delta_pos_dots
    
    # print('max_pen_vals', max_pen_vals)
    
    # find the penetration with the maximum value = minimum penetration
    min_pen_row_idx_fltrd = np.argmax(max_pen_vals)
    minmax_pen_val = max_pen_vals[min_pen_row_idx_fltrd]
    
    # print('minmax', minmax_pen_val)
    
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

# def sat(screen, body_a, body_b):
#     """
#     Standard vectorized SAT implementation using world-space transformations.
#     Uses the body's precomputed matrices for consistency.
    
#     Args:
#         body_a: Reference rigid body
#         body_b: Incident rigid body
        
#     Returns:
#         tuple: (normal_idx, vertex_idx, separations, delta_pos_projection) or None if no overlap
#     """
    
#     # Transform both bodies' vertices and normals to world space using precomputed matrices
    
#     # Body A: vertices and normals to world space
#     world_vertices_a = body_a.mesh.vertices @ body_a.rot_sca_mat.T + body_a.pos[:2]
#     world_normals_a = body_a.mesh.normals @ body_a.inv_rot_sca_mat.T
    
#     # Body B: vertices to world space  
#     world_vertices_b = body_b.mesh.vertices @ body_b.rot_sca_mat.T + body_b.pos[:2]
    
#     # Vector from A's center to B's center in world space
#     delta_pos_world = body_b.pos[:2] - body_a.pos[:2]
    
#     # Filter normals: only test normals pointing toward body_b
#     # Project delta_pos onto each normal to determine direction
#     delta_projections = world_normals_a @ delta_pos_world
#     facing_indices = np.where(delta_projections > 0)[0]
    
#     if len(facing_indices) == 0:
#         print('No Axes')
#         return None
    
#     # Get filtered normals
#     filtered_normals = world_normals_a[facing_indices]
    
#     # Project all vertices onto each filtered normal
#     # Shape: (num_normals, num_vertices_a)
#     projections_a = filtered_normals @ world_vertices_a.T
#     # Shape: (num_normals, num_vertices_b) 
#     projections_b = filtered_normals @ world_vertices_b.T
    
#     # For each normal, find the range of projections for each body
#     # Body A: max projection (furthest in normal direction)
#     max_proj_a = np.max(projections_a, axis=1)
#     min_proj_a = np.min(projections_a, axis=1)
#     # Body B: min projection (closest to body A along normal)
#     max_proj_b = np.max(projections_b, axis=1)
#     min_proj_b = np.min(projections_b, axis=1)
#     min_proj_b_indices = np.argmin(projections_b, axis=1)
    
#     if np.any(min_proj_a > max_proj_b) or np.any(max_proj_a < min_proj_b):
#         return None
    
#     # Positive overlap amount, negative = gap
#     separations = np.minimum(max_proj_a, max_proj_b) - np.maximum(min_proj_a, min_proj_b)

#     min_sep_idx = np.argmin(separations)   # smallest overlap/gap
#     min_separation = separations[min_sep_idx]

#     if min_separation <= 0:  # gap -> no collision
#         return None

    
#     # Get the actual indices (unfiltered)
#     actual_normal_idx = facing_indices[min_sep_idx]
#     actual_vertex_idx = min_proj_b_indices[min_sep_idx]
    
#     # Get the full row of projections for the separating axis
#     separation_row = projections_b[min_sep_idx]
    
#     # Project delta_pos onto the separating normal for additional info
#     delta_pos_projection = delta_projections[facing_indices[min_sep_idx]]
    
#     return actual_normal_idx, actual_vertex_idx, separation_row, delta_pos_projection

# def axes(body: Rigid) -> list[vec3]:
#     mesh = body.vertices
#     seps = []
#     for i in range(4):
#         edge = glm.normalize(mesh[(i + 1) % len(mesh)] - mesh[i])
#         seps.append((vec2(edge.y, -edge.x), mesh[i], mesh[(i + 1) % len(body.mesh.vertices)]))
#     return seps


# def project(body: Rigid, axis: vec2) -> tuple[float, float]:
#     """
#     Projects a rigid body's mesh onto a given axis and returns the min and max scalar values.
#     """
#     transformed_mesh = body.trans_vertices
    
#     # Initialize min and max with the projection of the first vertex
#     projection_values = [glm.dot(v, axis) for v in transformed_mesh]
    
#     min_projection = min(projection_values)
#     max_projection = max(projection_values)
    
#     return min_projection, max_projection


# def sat(screen, body_a: Rigid, body_b: Rigid) -> vec2 | None:
#     axs = axes(body_a)
    
#     min_overlap = float('inf')
#     min_axis = None
    
#     for (axis, e1, e2) in axs:
#         proj_a_min, proj_a_max = project(body_a, axis)
#         proj_b_min, proj_b_max = project(body_b, axis)
        
#         overlap = 0
        
#         if proj_a_min < proj_b_max and proj_a_max > proj_b_min:
#             overlap = min(proj_a_max, proj_b_max) - max(proj_a_min, proj_b_min)
#             assert overlap >= 0, 'Overlap is negative'
#             if glm.dot(body_b.pos[:2] - body_a.pos[:2], axis) < 0:
#                 overlap *= -1
#         else:
#             return None
            
#         current_overlap = abs(overlap)
#         if current_overlap < min_overlap:
#             min_overlap = current_overlap
#             min_axis = axis
            
#             if overlap < 0:
#                 min_axis = -min_axis
                
#     if min_axis is not None:
#         return min_axis * min_overlap

#     return None

# def sat(screen, body_a: Rigid, body_b: Rigid) -> np.ndarray | None:
#     """
#     Vectorized SAT that exactly matches the non-vectorized version 
#     (only testing axes from body_a).
#     """
#     # === Get transformed vertices ===
#     verts_a = body_a.trans_vertices   # shape (Na, 2)
#     verts_b = body_b.trans_vertices   # shape (Nb, 2)
    
#     # === Generate axes exactly like non-vectorized version ===
#     # Compute edges for body_a
#     edges_a = np.roll(verts_a, -1, axis=0) - verts_a  # shape (Na, 2)
#     # Normalize edges
#     edge_lengths = np.linalg.norm(edges_a, axis=1, keepdims=True)
#     edges_a_normalized = edges_a / edge_lengths
#     # Get perpendicular vectors (normals) - equivalent to vec2(edge.y, -edge.x)
#     normals_a = np.column_stack([edges_a_normalized[:, 1], -edges_a_normalized[:, 0]])
    
#     min_overlap = float('inf')
#     min_axis = None
    
#     for i, axis in enumerate(normals_a):
#         # Project both bodies onto this axis
#         proj_a = np.dot(verts_a, axis)  # (Na,)
#         proj_b = np.dot(verts_b, axis)  # (Nb,)
        
#         min_a, max_a = np.min(proj_a), np.max(proj_a)
#         min_b, max_b = np.min(proj_b), np.max(proj_b)
        
#         # Check for separation (matches non-vectorized logic exactly)
#         if min_a < max_b and max_a > min_b:
#             overlap = min(max_a, max_b) - max(min_a, min_b)
            
#             # Apply direction logic exactly like non-vectorized
#             delta_pos = body_b.pos[:2] - body_a.pos[:2]
#             if np.dot(delta_pos, axis) < 0:
#                 overlap *= -1
#         else:
#             return None  # Separated
        
#         # Track minimum overlap
#         current_overlap = abs(overlap)
#         if current_overlap < min_overlap:
#             min_overlap = current_overlap
#             min_axis = axis.copy()
            
#             if overlap < 0:
#                 min_axis = -min_axis
    
#     if min_axis is not None:
#         return min_axis * min_overlap
#     return None

    
def collide(manifold) -> bool:
    body_a: Rigid = manifold.body_a
    body_b: Rigid = manifold.body_b
    
    # run vectorized SAT from both object    
    data_a = sat(manifold.system.solver.screen, body_a, body_b)
    data_b = sat(manifold.system.solver.screen, body_b, body_a)
    
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
    
    
    
