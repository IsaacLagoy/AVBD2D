import glm
from glm import vec3, vec2
from shapes.rigid import Rigid
from helper.maths import inverse_transform
from collision.sutherland_hodgman import sutherland_hodgmen
import numpy as np


def axes(body: Rigid) -> list[vec3]:
    mesh = body.vertices
    seps = []
    for i in range(4):
        edge = glm.normalize(mesh[(i + 1) % len(mesh)] - mesh[i])
        seps.append((vec2(edge.y, -edge.x), mesh[i], mesh[(i + 1) % len(body.mesh.vertices)]))
    return seps


def project(body: Rigid, axis: vec2) -> tuple[float, float]:
    """
    Projects a rigid body's mesh onto a given axis and returns the min and max scalar values.
    """
    transformed_mesh = body.vertices
    
    # Initialize min and max with the projection of the first vertex
    projection_values = [glm.dot(v, axis) for v in transformed_mesh]
    
    min_projection = min(projection_values)
    max_projection = max(projection_values)
    
    return min_projection, max_projection


def sat(body_a: Rigid, body_b: Rigid) -> vec2 | None:
    axs = axes(body_a)
    
    min_overlap = float('inf')
    min_axis = None
    
    for (axis, e1, e2) in axs:
        proj_a_min, proj_a_max = project(body_a, axis)
        proj_b_min, proj_b_max = project(body_b, axis)
        
        overlap = 0
        
        if proj_a_min < proj_b_max and proj_a_max > proj_b_min:
            overlap = min(proj_a_max, proj_b_max) - max(proj_a_min, proj_b_min)
            assert overlap >= 0, 'Overlap is negative'
            if glm.dot(body_b.xy - body_a.xy, axis) < 0:
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


def collide(body_a: Rigid, body_b: Rigid, contacts) -> bool:
    pen_a = sat(body_a, body_b) # a has minimum axis
    pen_b = sat(body_b, body_a) # b has minimum axis
    
    # no collision
    if pen_a is None or pen_b is None:
        return 0  # No collision if either SAT test fails
    
    is_a = False
    if pen_b is None:
        is_a = True
        depth = glm.length(pen_a)
        normal = glm.normalize(pen_a)
        
    elif pen_a is None:
        is_a = False
        depth = glm.length(pen_b)
        normal = glm.normalize(pen_b)
        
    else:
        ag = glm.length2(pen_a) < glm.length2(pen_b)
        normal = glm.normalize(pen_a) if ag else glm.normalize(pen_b)
        depth = glm.length(pen_a) if ag else glm.length(pen_b)
        is_a = ag
    
    clipped = sutherland_hodgmen(body_b, body_a) if is_a else sutherland_hodgmen(body_a, body_b)
    
    if not len(clipped):
        return 0
    
    margin = depth * 0.02
    c_normal = normal if glm.dot(body_a.xy - body_b.xy, normal) < 0 else -normal
    
    clipped.sort(key=lambda c: glm.dot(c, c_normal))

    # project points and find the closest
    rA = clipped[-1]
    rB = clipped[0]
    
    contacts[0].normal = -c_normal
    contacts[0].rA = inverse_transform(body_a.pos, body_a.scale, rA)
    contacts[0].rB = inverse_transform(body_b.pos, body_b.scale, rB)
    
    rA2 = clipped[-2] if glm.dot(clipped[-2], c_normal) > glm.dot(clipped[-1], c_normal) - margin else rA
    rB2 = clipped[1] if glm.dot(clipped[1], c_normal) < glm.dot(clipped[0], c_normal) + margin else rB
    
    # check if there is only on unique contact point
    if (np.allclose(rA2, rA) and np.allclose(rB2, rB)) or \
       (np.allclose(rB2, rA) or np.allclose(rA2, rB)):

        return 1
    
    contacts[1].normal = -c_normal
    contacts[1].rA = inverse_transform(body_a.pos, body_a.scale, rA2)
    contacts[1].rB = inverse_transform(body_b.pos, body_b.scale, rB2)
    
    return 2