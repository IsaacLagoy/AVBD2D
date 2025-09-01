from shapes.rigid import Rigid
import numpy as np
from collision.sutherland_hodgman import sutherland_hodgman
from helper.maths import inverse_transform


def axes(body: Rigid):
    mesh = body.vertices
    num_verts = len(mesh)
    seps = []
    
    for i in range(num_verts):
        edge = mesh[(i + 1) % num_verts] - mesh[i]
        denom = np.linalg.norm(edge)
        edge /= denom
        
        seps.append(np.array([edge[1], -edge[0]], dtype='float32'))
        
    return seps

def project(body: Rigid, axis):
    """
    Projects a rigid body's mesh onto a given axis and returns the min and max scalar values.
    """
    # TODO get real world location of vertices
    projs = np.dot(body.vertices, axis)
    min_proj = np.min(projs)
    max_proj = np.max(projs)
    return min_proj, max_proj

def sat(body_a: Rigid, body_b: Rigid):
    axs = axes(body_a)
   
    min_overlap = float('inf')
    min_axis = None
    
    for axis in axs:
        proj_a_min, proj_a_max = project(body_a, axis)
        proj_b_min, proj_b_max = project(body_b, axis)
        
        overlap = 0
        
        if proj_a_min < proj_b_max and proj_a_max > proj_b_min:
            overlap = min(proj_a_max, proj_b_max) - max(proj_a_min, proj_b_min)
            assert overlap >= 0, 'Overlap is negative'
            if np.dot(body_b.pos[:2] - body_a.pos[:2], axis) < 0:
                overlap *= -1
        else:
            return None
            
        current_overlap = abs(overlap)
        if current_overlap < min_overlap:
            min_overlap = current_overlap
            min_axis = axis
            
            if overlap < 0:
                min_axis = -min_axis
                
    if min_axis is not None:
        return min_axis * min_overlap

    return None

def collide(manifold) -> bool:
    # easy access variables
    body_a = manifold.body_a
    body_b = manifold.body_b
    
    pen_a = sat(body_a, body_b) # a has minimum axis
    pen_b = sat(body_b, body_a) # b has minimum axis
    
    # no collision
    if pen_a is None or pen_b is None:
        return 0  # No collision if either SAT test fails
    
    is_a = False
    if pen_b is None:
        is_a = True
        depth = np.linalg.norm(pen_a)
        normal = pen_a / depth
        
    elif pen_a is None:
        is_a = False
        depth = np.linalg.norm(pen_b)
        normal = pen_b / depth
        
    else:
        ag = np.dot(pen_a, pen_a) < np.dot(pen_b, pen_b)
        depth = np.linalg.norm(pen_a) if ag else np.linalg.norm(pen_b)
        normal = pen_a / depth if ag else pen_b / depth
        
        is_a = ag
    
    clipped = sutherland_hodgman(body_b, body_a) if is_a else sutherland_hodgman(body_a, body_b)
    
    if not len(clipped):
        return 0
    
    margin = depth * 0.02
    c_normal = normal if np.dot(body_a.pos[:2] - body_b.pos[:2], normal) < 0 else -normal
    
    clipped.sort(key=lambda c: np.dot(c, c_normal))

    # project points and find the closest
    rA = clipped[-1]
    rB = clipped[0]
    
    manifold.normal[0] = -c_normal
    manifold.rA[0] = inverse_transform(body_a.pos, body_a.scale, rA)
    manifold.rB[0] = inverse_transform(body_b.pos, body_b.scale, rB)
    
    rA2 = clipped[-2] if np.dot(clipped[-2], c_normal) > np.dot(clipped[-1], c_normal) - margin else rA
    rB2 = clipped[1] if np.dot(clipped[1], c_normal) < np.dot(clipped[0], c_normal) + margin else rB
    
    # check if there is only on unique contact point
    # TODO this is going to throw errors
    if (np.allclose(rA2, rA) and np.allclose(rB2, rB)) or (np.allclose(rB2, rA) or np.allclose(rA2, rB)):
        return 1
    
    manifold.normal[1] = -c_normal
    manifold.rA[1] = inverse_transform(body_a.pos, body_a.scale, rA2)
    manifold.rB[1] = inverse_transform(body_b.pos, body_b.scale, rB2)
    
    return 2