import numpy as np
import numba as nb

@nb.njit(parallel=True)
def scatter_add(A, mapping_keys, mapping_values, offsets):
    for i in nb.prange(len(mapping_keys)):
        k = mapping_keys[i]
        start, end = offsets[i], offsets[i+1]
        for j in range(start, end):
            A[k] += A[mapping_values[j]]