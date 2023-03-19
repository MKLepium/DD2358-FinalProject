//
// Created by Maxi on 18.03.2023.
//
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#ifndef FINALPROJECT_RAY_TRACING_MODULE_H
#define FINALPROJECT_RAY_TRACING_MODULE_H

py::array_t<float> run(int w, int h);



#endif //FINALPROJECT_RAY_TRACING_MODULE_H
