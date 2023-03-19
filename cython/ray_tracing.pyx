import numpy as np
cimport numpy as np
from libc.math cimport sqrt, fmax

#cython: boundscheck=False
def normalize(np.ndarray[float, ndim=1] x):
        # This function normalizes a vector.
        x /= np.linalg.norm(x)
        return x

#cython: boundscheck=False
cdef float intersect_sphere(np.ndarray[float, ndim=1] O, np.ndarray[float, ndim=1] D, np.ndarray[float, ndim=1] S, float R):
    # Return the distance from O to the intersection
    # of the ray (O, D) with the sphere (S, R), or
    # +inf if there is no intersection.
    # O and S are 3D points, D (direction) is a
    # normalized vector, R is a scalar.
    a = np.dot(D, D)
    OS = O - S
    b = 2 * np.dot(D, OS)
    c = np.dot(OS, OS) - R * R
    disc = b * b - 4 * a * c
    if disc > 0:
        distSqrt = sqrt(disc)
        q = (-b - distSqrt) / 2.0 if b < 0 else (-b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return np.inf


#cython: boundscheck=False
def trace_ray(np.ndarray[float, ndim=1] O, np.ndarray[float, ndim=1] D, float ambient, np.ndarray[float, ndim=1] color_light, np.ndarray[float, ndim=1] L, np.ndarray[float, ndim=1] position, float radius, np.ndarray[float, ndim=1] color, float diffuse, float specular_c, float specular_k):
    # Find first point of intersection with the scene.
    t = intersect_sphere(O, D, position, radius)
    # No intersection?
    if t == np.inf:
        return None
    # Find the point of intersection on the object.
    M = O + D * t
    N = normalize(M - position)
    toL = normalize(L - M)
    toO = normalize(O - M)
    # Ambient light.
    col = ambient
    # Lambert shading (diffuse).
    col += diffuse * fmax(np.dot(N, toL), 0) * color
    # Blinn-Phong shading (specular).
    col += specular_c * color_light * fmax(np.dot(N, normalize(toL + toO)), 0) ** specular_k
    return col



#cython: boundscheck=False
# Define Cython types
ctypedef np.float64_t dtype_t
ctypedef np.int_t index_t

def run(index_t w, index_t h):
    # Declare Cython types
    cdef np.ndarray[dtype_t, ndim=3] img
    cdef np.ndarray[dtype_t, ndim=1] position, color, color_light, O, Q, toL, toO, N
    cdef dtype_t radius, diffuse, specular_c
    cdef index_t i, j
    cdef dtype_t x, y, ambient
    cdef np.ndarray[dtype_t, ndim=1] L

    img = np.zeros((h, w, 3), dtype=dtype_t)
    position = np.array([0., 0., 1.], dtype=dtype_t)
    radius = 1.
    color = np.array([0., 0., 1.], dtype=dtype_t)
    diffuse = 1.
    specular_c = 1.
    specular_k = 50
    L = np.array([5., 5., -10.], dtype=dtype_t)
    color_light = np.ones(3, dtype=dtype_t)
    ambient = .05
    
    O = np.array([0., 0., -1.], dtype=dtype_t)  # Position.
    Q = np.array([0., 0., 0.], dtype=dtype_t)  # Pointing to.


    # Loop through all pixels.
    for i in range(w):
        x = (i / (w - 1)) * 2 - 1
        for j in range(h):
            y = (j / (h - 1)) * 2 - 1
            # Position of the pixel.
            Q[0], Q[1] = x, y
            # Direction of the ray going through
            # the optical center.
            D = normalize(Q - O)
            # Launch the ray and get the color
            # of the pixel.
            col = trace_ray(O, D, ambient, color_light, L, position, radius, color, diffuse, specular_c, specular_k)
            
            if col is None:
                continue
            img[h - j - 1, i, :] = np.clip(col, 0, 1)
    return img