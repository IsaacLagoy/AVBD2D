import numpy as np
import numba as nb

def solve(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    N = lhs.shape[0]
    out = np.empty((N, 3), dtype=lhs.dtype)

    for i in nb.prange(N):  # parallel loop over systems
        a = lhs[i]
        b = rhs[i]

        # LDLᵀ decomposition (unrolled for 3x3)
        D1 = a[0, 0]
        L21 = a[1, 0] / D1
        L31 = a[2, 0] / D1
        D2 = a[1, 1] - L21 * L21 * D1
        L32 = (a[2, 1] - L21 * L31 * D1) / D2
        D3 = a[2, 2] - (L31 * L31 * D1 + L32 * L32 * D2)

        # Forward substitution: Ly = b
        y1 = b[0]
        y2 = b[1] - L21 * y1
        y3 = b[2] - L31 * y1 - L32 * y2

        # Diagonal solve: Dz = y
        z1 = y1 / D1
        z2 = y2 / D2
        z3 = y3 / D3

        # Backward substitution: Lᵀx = z
        x2 = z3
        x1 = z2 - L32 * x2
        x0 = z1 - L21 * x1 - L31 * x2

        out[i, 0] = x0
        out[i, 1] = x1
        out[i, 2] = x2

    return out
