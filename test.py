import numpy as np

def normalize(vectors):
    """Normalize an array of row vectors."""
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

# -------------------------------------------------
# Step 1: normals (as row vectors)
# -------------------------------------------------
normals = np.array([
    [1.0, 0.0],   # x-axis
    [0.0, 1.0],   # y-axis
    [-1.0, 0.0],  # -x
    [0.0, -1.0]   # -y
])

# -------------------------------------------------
# Step 2: build transformation (scale + rotation)
# -------------------------------------------------
scale = (2.0, 1.0)        # non-uniform scale
theta = np.pi / 4         # 45° rotation

S = np.array([[scale[0], 0.0],
              [0.0,       scale[1]]])

R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

M = R @ S   # model matrix (no translation for normals)

# -------------------------------------------------
# Step 3: transform normals
# -------------------------------------------------
# Apply "normal matrix" (inverse transpose of M)
normal_matrix = np.linalg.inv(M)
transformed_normals = normalize(normals @ normal_matrix)

# -------------------------------------------------
# Step 4: apply the *inverse* transform to recover
# -------------------------------------------------
# The inverse of the normal_matrix is (M.T)
inverse_normal_matrix = np.linalg.inv(normal_matrix)
recovered_normals = normalize(normals @ normal_matrix @ inverse_normal_matrix)

# -------------------------------------------------
# Display
# -------------------------------------------------
print("Original normals:\n", normals)
print("\nAfter transform:\n", transformed_normals)
print("\nAfter applying inverse transform:\n", recovered_normals)

print("\nRecovered ≈ Original? ->", np.allclose(normals, recovered_normals))
