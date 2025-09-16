import numpy as np

def intersect_perpendicular(P1, P2, v, d):
    """
    Find intersection of a line segment P1-P2 with a line perpendicular to vector v
    that passes through an unknown point U with dot product v.U = d.
    
    Parameters:
    - P1, P2: endpoints of the line segment as 1D numpy arrays
    - v: 2D vector as a numpy array
    - d: scalar, dot product v.U
    
    Returns:
    - Intersection point as a numpy array
    """
    l1 = np.dot(v, P1)
    l2 = np.dot(v, P2)

    if np.isclose(l2, l1):
        raise ValueError("The line segment is perpendicular to v; cannot determine unique intersection.")

    t = (d - l1) / (l2 - l1)
    
    # Clip t to [0, 1] in case of numerical errors
    t = np.clip(t, 0.0, 1.0)
    
    X = P1 + t * (P2 - P1)
    return X


# ----------------- TEST CASES -----------------

def test_intersect_perpendicular():
    P1 = np.array([0.0, 0.0])
    P2 = np.array([10.0, 0.0])
    v = np.array([0.5, 0.5])
    
    # U has dot product 5 with v
    d = 5.0
    X = intersect_perpendicular(P1, P2, v, d)
    print("Test 1 Intersection:", X)
    
    # U has dot product 0 with v
    d = 0.0
    X = intersect_perpendicular(P1, P2, v, d)
    print("Test 2 Intersection:", X)
    
    # U has dot product 10 with v
    d = 10.0
    X = intersect_perpendicular(P1, P2, v, d)
    print("Test 3 Intersection:", X)
    
    # Vertical segment
    P1 = np.array([2.0, 0.0])
    P2 = np.array([2.0, 5.0])
    v = np.array([1.0, 0.0])
    d = 2.0
    X = intersect_perpendicular(P1, P2, v, d)
    print("Test 4 Intersection (vertical segment):", X)

test_intersect_perpendicular()
