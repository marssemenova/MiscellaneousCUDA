cmake_minimum_required(VERSION 3.31)
project(NBodySim CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(NBodySim main.cu)

set_target_properties(NBodySim PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
