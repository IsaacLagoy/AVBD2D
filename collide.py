import glm
from glm import vec3, vec2
from rigid import Rigid
from maths import transform, inverse_transform
from sutherland_hodgman import sutherland_hodgmen


def axes(body: Rigid) -> list[vec3]:
    mesh = body.vertices
    seps = []
    for i in range(4):
        edge = glm.normalize(mesh[i] - mesh[(i + 1) % len(body.mesh.vertices)])
        seps.append((vec2(-edge.y, edge.x), mesh[i], mesh[(i + 1) % len(body.mesh.vertices)]))
    return seps


def project(body: Rigid, axis: vec2) -> tuple[float, float]:
    """
    Projects a rigid body's mesh onto a given axis and returns the min and max scalar values.
    """
    transformed_mesh = body.vertices
    
    # Initialize min and max with the projection of the first vertex
    projection_values = [(glm.dot(v, axis), r) for v, r in zip(transformed_mesh, transformed_mesh)]
    
    min_projection = min(projection_values)
    max_projection = max(projection_values)
    
    return min_projection, max_projection


def sat(body_a: Rigid, body_b: Rigid) -> vec2 | None:
    axs = axes(body_a)
    
    min_overlap = float('inf')
    min_axis = None
    min_a = None
    min_edge = None
    
    for (axis, e1, e2) in axs:
        projs_a = project(body_a, axis)
        projs_b = project(body_b, axis)
        
        proj_a_min, proj_a_max = projs_a[0][0], projs_a[1][0]
        proj_b_min, proj_b_max = projs_b[0][0], projs_b[1][0]
        
        overlap = 0
        
        if proj_a_min < proj_b_max and proj_a_max > proj_b_min:
            overlap = min(proj_a_max, proj_b_max) - max(proj_a_min, proj_b_min)
            assert overlap >= 0, 'Overlap is negative'
            if glm.dot(body_b.pos.xy - body_a.pos.xy, axis) < 0:
                overlap *= -1
        else:
            return None, None, None
            
        current_overlap = abs(overlap)
        if current_overlap < min_overlap:
            min_overlap = current_overlap
            min_axis = axis
            min_a = projs_b[0][1]
            min_edge = (e1, e2)
            
            if overlap < 0:
                min_axis = -min_axis
                
    if min_axis is not None:
        return min_axis * min_overlap, min_a, min_edge

    return None, None, None


def collide(body_a: Rigid, body_b: Rigid, contacts) -> bool:
    pen_a, min_b, edge_a = sat(body_a, body_b) # a has minimum axis
    pen_b, min_a, edge_b = sat(body_b, body_a) # b has minimum axis
    
    # no collision
    if not pen_a and not pen_b:
        return 0
    
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
    
    print('collide')
    
    if not len(clipped):
        return 0
    
    print(normal)
    
    body_a.color = (255, 0, 0) # TODO remove after debug
    
    clipped.sort(key=lambda c: glm.dot(c, normal))

    # project points and find the closest
    rA = clipped[-1]
    rB = clipped[0]
    
    margin = depth * 0.02
    
    contacts[0].normal = -normal
    contacts[0].rA = inverse_transform(body_a.pos, body_a.scale, rA)
    contacts[0].rB = inverse_transform(body_b.pos, body_b.scale, rB)
    
    rA2 = clipped[-2] if glm.dot(clipped[-2], normal) > glm.dot(clipped[-1], normal) - margin else rA
    rB2 = clipped[1] if glm.dot(clipped[1], normal) < glm.dot(clipped[0], normal) + margin else rB
    
    # check if there is only on unique contact point
    if (rA2 == rA and rB2 == rB) or (rB2 == rA or rA2 == rB):
        return 1
    
    contacts[1].normal = -normal
    contacts[1].rA = inverse_transform(body_a.pos, body_a.scale, rA2)
    contacts[1].rB = inverse_transform(body_b.pos, body_b.scale, rB2)
    
    return 2