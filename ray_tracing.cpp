
//#include <pybind11/pybind11.h>
//#include <pybind11/numpy.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
#include <iostream>
//namespace py = pybind11;


std::vector<float> normalize(std::vector<float> v);
float dot(std::vector<float> v1, std::vector<float> v2);
float intersect_sphere(const std::vector<float>& O, const std::vector<float>& D,
                       const std::vector<float>& S, float R);

std::vector<float> trace_ray(
        std::vector<float> O,
        float ambient,
        std::vector<float> color_light,
        std::vector<float> L,
        std::vector<float> position,
        float radius,
        std::vector<float> color,
        float diffuse,
        float specular_c,
        float specular_k,
        std::vector<float> D
);
std::vector<float> create_image(int h, int w);


std::vector<float> create_image(int h, int w) {
    std::vector<float> img(h * w * 3);
    for (int i = 0; i < h * w * 3; ++i) {
        img[i] = 0.0f;
    }
    return img;
}


std::vector<float> normalize(std::vector<float> v) {
    float norm = std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0f));
    std::vector<float> x(3);
    std::transform(v.begin(), v.end(), x.begin(), [norm](float element) {
        return element / norm;
    });
    return x;
}

float intersect_sphere(const std::vector<float>& O, const std::vector<float>& D,
                       const std::vector<float>& S, float R) {
    // Return the distance from O to the intersection
    // of the ray (O, D) with the sphere (S, R), or
    // +inf if there is no intersection.
    // O and S are 3D points, D (direction) is a
    // normalized vector, R is a scalar.
    float a = dot(D, D);
    std::vector<float> OS = {O[0] - S[0], O[1] - S[1], O[2] - S[2]};
    float b = 2 * dot(D, OS);
    float c = dot(OS, OS) - R * R;
    float disc = b * b - 4 * a * c;
    if (disc > 0) {
        float distSqrt = std::sqrt(disc);
        float q;
        if (b < 0) {
            q = (-b - distSqrt) / 2.0f;
        } else {
            q = (-b + distSqrt) / 2.0f;
        }
        float t0 = q / a;
        float t1 = c / q;
        t0 = std::min(t0, t1);
        t1 = std::max(t0, t1);
        if (t1 >= 0) {
            if (t0 < 0) {
                return t1;
            } else {
                return t0;
            }
        }
    }
    return std::numeric_limits<float>::infinity();
}


float dot(std::vector<float> v1, std::vector<float> v2) {
    float result = 0.0f;
    for (std::size_t i = 0; i < v1.size(); ++i) {
        result += v1[i] * v2[i];
    }

    return result;
}

std::vector<float> trace_ray(
    std::vector<float> O, float ambient, std::vector<float> color_light, std::vector<float> L, std::vector<float> position, float radius, std::vector<float> color, 
    float diffuse, float specular_c, float specular_k, std::vector<float> D) 
{

    float t = intersect_sphere(O, D, position, radius);
    if (t == std::numeric_limits<float>::infinity()) {
        return {std::numeric_limits<float>::infinity(), 
                std::numeric_limits<float>::infinity(), 
                std::numeric_limits<float>::infinity()};
    }
    std::vector<float> M = {O[0] + D[0] * t, O[1] + D[1] * t, O[2] + D[2] * t};
    std::vector<float> N = {M[0] - position[0], M[1] - position[1], M[2] - position[2]};
    N = normalize(N);
    std::vector<float> toL = normalize({L[0] - M[0], L[1] - M[1], L[2] - M[2]});
    std::vector<float> toO = normalize({O[0] - M[0], O[1] - M[1], O[2] - M[2]});
    std::vector<float> col = {ambient, ambient, ambient};
    // Lambert shading (diffuse).
    col[0] += diffuse * std::max(dot(N, toL), 0.0f) * color[0];
    col[1] += diffuse * std::max(dot(N, toL), 0.0f) * color[1];
    col[2] += diffuse * std::max(dot(N, toL), 0.0f) * color[2];

    // Blinn-Phong shading (specular).
    col[0] += specular_c * color_light[0] * std::pow(std::max(dot(N, normalize({toL[0] + toO[0], toL[1] + toO[1], toL[2] + toO[2]})), 0.0f), specular_k);
    col[1] += specular_c * color_light[1] * std::pow(std::max(dot(N, normalize({toL[0] + toO[0], toL[1] + toO[1], toL[2] + toO[2]})), 0.0f), specular_k);
    col[2] += specular_c * color_light[2] * std::pow(std::max(dot(N, normalize({toL[0] + toO[0], toL[1] + toO[1], toL[2] + toO[2]})), 0.0f), specular_k);
    return col;
}

std::vector<float> run(int w, int h) {
    // Initialize image array with zeros
    // position = np.array([0., 0., 1.])
    std::vector<float> position = {0.0f, 0.0f, 1.0f};
    // radius = 1.
    float radius = 0.1f;
    // color = np.array([0., 0., 1.])
    std::vector<float> color = {0.0f, 0.0f, 1.0f};
    // diffuse = 1.
    float diffuse = 1.0f;
    // specular_c = 1.
    float specular_c = 1.0f;
    // specular_k = 50
    float specular_k = 50.0f;
    // L = np.array([5., 5., -10.])
    std::vector<float> L = {5.0f, 5.0f, -10.0f};
    // color_light = np.ones(3)
    std::vector<float> color_light = {1.0f, 1.0f, 1.0f};
    // ambient = .05
    float ambient = 0.05f;
    // img = np.zeros((h, w, 3))
    // O = np.array([0., 0., -1.])  # Position.
    std::vector<float> O = {0.0f, 0.0f, -1.0f};
    std::vector<float> Q = {0, 0, 0}; 

    // get pointer from img_out
    std::vector<float> img_out = create_image(w, h);
    // Loop through all pixels
    for (int i = 0; i < w; i++) {
        double x = -1.0 + (2.0 * i) / (w - 1);
        for (int j = 0; j < h; j++) {
            double y = -1.0 + (2.0 * j) / (h - 1);

            // Position of the pixel.
            Q[0] = x;
            Q[1] = y;

            // Direction of the ray going through the optical center
            
            Q[0] = Q[0] - O[0];
            Q[1] = Q[1] - O[1];
            Q[2] = Q[2] - O[2];
            auto D = normalize(Q);

/*
    std::vector<float> O, 
    float ambient, 
    std::vector<float> color_light, 
    std::vector<float> L, 
    std::vector<float> position, 
    float radius, 
    std::vector<float> color, 
    float diffuse, 
    float specular_c, 
    float specular_k, 
    std::vector<float> D
*/
            // Launch the ray and get the color of the pixel
            auto col = trace_ray(
                O,              //  std::vector<float> O, 
                ambient,        // float ambient, 
                color_light,    // std::vector<float> color_light,
                L,              // std::vector<float> L,
                position,       // std::vector<float> position,
                radius,         // float radius,
                color,          // std::vector<float> color,
                diffuse,        // float diffuse,
                specular_c,     // float specular_c,
                specular_k,     // float specular_k,
                D               // std::vector<float> D
            );


            if (col[0] == std::numeric_limits<float>::infinity()) {
                // Background color.
                int index = (h - j - 1) * w * 3 + i * 3;
                img_out[index] = 0.0f;
                img_out[index + 1] = 0.0f;
                img_out[index + 2] = 0.0f;
            } else {
                // Color of the sphere.
                int index = (h - j - 1) * w * 3 + i * 3;
                img_out[index] = std::max(0.0f, std::min(col[0], 1.0f)); 
                img_out[index + 1] = std::max(0.0f, std::min(col[1], 1.0f)); 
                img_out[index + 2] = std::max(0.0f, std::min(col[2], 1.0f));
            }


        }
    }

    
/* w, h = 5 Expected:
[[[0.         0.         0.        ]
  [0.         0.         0.        ]
  [0.         0.         0.        ]
  [0.         0.         0.        ]
  [0.         0.         0.        ]]

 [[0.         0.         0.        ]
  [0.         0.         0.        ]
  [0.05000051 0.05000051 0.9365578 ]
  [0.         0.         0.        ]
  [0.         0.         0.        ]]

 [[0.         0.         0.        ]
  [0.05       0.05       0.42899987]
  [0.14018093 0.14018093 0.95667751]
  [0.05000051 0.05000051 0.9365578 ]
  [0.         0.         0.        ]]

 [[0.         0.         0.        ]
  [0.         0.         0.        ]
  [0.05       0.05       0.42899987]
  [0.         0.         0.        ]
  [0.         0.         0.        ]]

 [[0.         0.         0.        ]
  [0.         0.         0.        ]
  [0.         0.         0.        ]
  [0.         0.         0.        ]
  [0.         0.         0.        ]]]

0 0 0
0 0 0
0 0 0
0 0 0
0 0 0
0 0 0
0 0 0
0.0500563
0.0500563
0.921588
0.050533
0.050533
1
0.05
0.05
0.859375
0
0
0
0
0
0
0.172574
0.172574
1
0.0604168
0.0604168
0.974853
0.05
0.05
0.818471
0
0
0
0
0
0
0.05
0.05
0.227984
0.05
0.05
0.540575
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0

Process finished with exit code 0


Process finished with exit code 0

 */

    for(auto i : img_out){
        std::cout << i << std::endl;
    }
    return img_out;
}


int main(){
    run(5,5);
    return 0;
}

/*
 *
 *
PYBIND11_MODULE(raytracer, m) {

    m.def("run", &run, py::return_value_policy::reference);
    py::bind_vector<std::vector<float>>(m, "FloatVector");
}*/