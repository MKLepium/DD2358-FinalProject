import numpy as np
from numpy.linalg import norm

def normalize(x):
        # This function normalizes a vector.
        x /= np.linalg.norm(x)
        return x

def intersect_sphere(O, D, S, R):
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
            distSqrt = np.sqrt(disc)
            q = (-b - distSqrt) / 2.0 if b < 0 \
                else (-b + distSqrt) / 2.0
            t0 = q / a
            t1 = c / q
            t0, t1 = min(t0, t1), max(t0, t1)
            if t1 >= 0:
                return t1 if t0 < 0 else t0
        return np.inf

def trace_ray(O, D, ambient, color_light, L, position, radius, color, diffuse, specular_c, specular_k):
        # Find first point of intersection with the scene.
        t = intersect_sphere(O, D, position, radius)
        # No intersection?
        if t == np.inf:
            return
        # Find the point of intersection on the object.
        M = O + D * t
        N = normalize(M - position)
        toL = normalize(L - M)
        toO = normalize(O - M)
        # Ambient light.
        col = ambient
        # Lambert shading (diffuse).
        col += diffuse * max(np.dot(N, toL), 0) * color
        # Blinn-Phong shading (specular).
        col += specular_c * color_light * \
            max(np.dot(N, normalize(toL + toO)), 0) \
            ** specular_k
        return col

def run(w, h):
    img = np.zeros((h, w, 3))
    position = np.array([0., 0., 1.])
    radius = 1.
    color = np.array([0., 0., 1.])
    diffuse = 1.
    specular_c = 1.
    specular_k = 50
    L = np.array([5., 5., -10.])
    color_light = np.ones(3)
    ambient = .05
    
    O = np.array([0., 0., -1.])  # Position.
    Q = np.array([0., 0., 0.])  # Pointing to.


    # Loop through all pixels.
    for i, x in enumerate(np.linspace(-1, 1, w)):
        for j, y in enumerate(np.linspace(-1, 1, h)):
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


def main():
    import matplotlib.pyplot as plt
    import time
    import raytracer
    times_run = []
    times_cpp = []
    ws = [5,20,100, 500, 1000, 2000, 5000, 10000]
    hs = [5,20,100, 500, 1000, 2000, 5000, 10000]
    for w, h in zip(ws, hs):
        start = time.time()
        img = run(w, h)
        end = time.time()
        times_run.append(end - start)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(img)
        ax.set_axis_off()
        plt.savefig(f'ray_python_tracing_{w}x{h}.png')
        plt.clf()

        start = time.time()
        # use gdb to set breakpoint


        



        img = raytracer.run(w, h)
        end = time.time()
        times_cpp.append(end - start)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(img)
        ax.set_axis_off()
        plt.imshow(img)
        plt.savefig(f'ray_cpp_tracing_{w}x{h}.png')
        plt.clf()

    plt.plot(ws, times_run, label='Ray Tracing pure Python')
    plt.plot(ws, times_cpp, label='Ray Tracing C++')
    plt.xlabel('Image Width')
    plt.ylabel('Time (s)')
    plt.savefig('ray_tracing_time.png')

if __name__ == '__main__':
    main()