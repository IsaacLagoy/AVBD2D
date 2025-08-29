import glm
from glm import vec2, vec3


def cross(a: vec2, b: vec2) -> float:
    return a.x * b.y - a.y * b.x

def rotate(angle: float, v: vec2) -> vec2:
    """Rotate vec2 v by angle (in radians)."""
    c, s = glm.cos(angle), glm.sin(angle)
    return vec2(c * v.x - s * v.y, s * v.x + c * v.y)

def rotate_scale(angle: float, scale: vec2, v: vec2) -> vec2:
    """Scale vec2 v by scale, then rotate by angle (in radians)."""
    # Apply scaling first
    sx, sy = scale.x * v.x, scale.y * v.y
    
    # Apply rotation
    c, s = glm.cos(angle), glm.sin(angle)
    return vec2(c * sx - s * sy, s * sx + c * sy)


def transform(pos: vec3, sca: vec2, r: vec2) -> vec2:
    scaled_r = r * sca
    rotated_r = rotate(pos.z, scaled_r)
    return vec2(pos.x, pos.y) + rotated_r

def inverse_transform(pos: vec3, sca: vec2, p: vec2) -> vec2:
    local = p - vec2(pos.x, pos.y)
    local = rotate(-pos.z, local)
    local = vec2(local.x / sca.x, local.y / sca.y)

    return local

def sign(i: float) -> int:
    return 2 * (i >= 0) - 1

def closest_point_on_segment(a: vec2, b: vec2, p: vec2) -> vec2:
    ab = b - a
    ap = p - a
    
    ab2 = glm.dot(ab, ab)
    if ab2 < 1e-8:
        return a
    
    t = glm.dot(ap, ab)
    
    if t < 0: return a
    if t > 1: return b
    return a + t * ab

def segment_intersect(p1: vec2, p2: vec2, q1: vec2, q2: vec2) -> vec2 | None:
    d1 = p2 - p1
    d2 = q2 - q1
    
    denom = cross(d1, d2)
    if abs(denom) < 1e-8:
        return None
    
    t = cross(q1 - p1, d2) / denom
    return p1 + t * d1

def inside(p: vec2, a: vec2, b: vec2) -> bool:
    return cross(b - a, p - a) >= 0