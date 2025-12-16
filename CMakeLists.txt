cmake_minimum_required(VERSION 3.20)
project(MeanShiftParallel)

set(CMAKE_CXX_STANDARD 17)

if(APPLE)
    set(CMAKE_PREFIX_PATH "/opt/homebrew/opt/libomp" ${CMAKE_PREFIX_PATH})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp")
endif()
# -----------------------------------

find_package(OpenMP REQUIRED)

add_executable(MeanShiftParallel main.cpp
        stb_image.h
        stb_image_write.h)

if(OpenMP_CXX_FOUND)
    target_link_libraries(MeanShiftParallel PUBLIC OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_options(MeanShiftParallel PRIVATE /O2 /arch:AVX2 /openmp)
else()
    target_compile_options(MeanShiftParallel PRIVATE -O3 -march=native)
    if(NOT APPLE)
        target_compile_options(MeanShiftParallel PRIVATE -fopenmp)
    endif()
endif()