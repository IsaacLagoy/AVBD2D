from glm import vec3, mat3x3, inverse

def solve(a: mat3x3, b: vec3) -> vec3:
    """
    Solve linear system Ax = b using LDL^T decomposition
    """
    # Compute LDL^T decomposition
    D1 = a[0][0]
    L21 = a[1][0] / a[0][0]
    L31 = a[2][0] / a[0][0]
    D2 = a[1][1] - L21 * L21 * D1
    L32 = (a[2][1] - L21 * L31 * D1) / D2
    D3 = a[2][2] - (L31 * L31 * D1 + L32 * L32 * D2)
    
    # Forward substitution: Solve Ly = b
    y1 = b.x
    y2 = b.y - L21 * y1
    y3 = b.z - L31 * y1 - L32 * y2
    
    # Diagonal solve: Solve Dz = y
    z1 = y1 / D1
    z2 = y2 / D2
    z3 = y3 / D3
    
    # Backward substitution: Solve L^T x = z
    x = vec3()
    x[2] = z3
    x[1] = z2 - L32 * x[2]
    x[0] = z1 - L21 * x[1] - L31 * x[2]
    
    return x

# This is the simplist way to solve. 3x3 inverse may be faster than ldlt
# This function is currently overriding the one above
def solve(a: mat3x3, b: vec3) -> vec3:
    return inverse(a) * b