cmake_minimum_required(VERSION 3.20)
project(MeanShiftParallel CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Apple clang non include OpenMP: si usa libomp di Homebrew (brew install libomp)
if(APPLE)
    set(CMAKE_PREFIX_PATH "/opt/homebrew/opt/libomp" ${CMAKE_PREFIX_PATH})
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    include_directories(SYSTEM "/opt/homebrew/opt/libomp/include")
endif()
find_package(OpenMP REQUIRED)

add_executable(MeanShiftParallel
        src/main.cpp
        src/mean_shift.cpp
        src/image_io.cpp
        src/benchmark.cpp)
target_include_directories(MeanShiftParallel PRIVATE src)
target_include_directories(MeanShiftParallel SYSTEM PRIVATE third_party)
target_link_libraries(MeanShiftParallel PRIVATE OpenMP::OpenMP_CXX)

if(MSVC)
    target_compile_options(MeanShiftParallel PRIVATE /O2 /arch:AVX2 /openmp:experimental)
else()
    # Niente -ffast-math: la riduzione floating point viene riordinata solo dove lo chiede
    # esplicitamente #pragma omp simd reduction (la baseline sequenziale resta scalare)
    target_compile_options(MeanShiftParallel PRIVATE -O3 -march=native)
endif()

# Report di vettorizzazione del compilatore: cmake -DVEC_REPORT=ON ...
option(VEC_REPORT "Stampa i report di vettorizzazione (clang)" OFF)
if(VEC_REPORT AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(MeanShiftParallel PRIVATE
            -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize)
endif()
