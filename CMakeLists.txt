cmake_minimum_required(VERSION 3.20)
project(MeanShiftParallel CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)

add_executable(MeanShiftParallel
        src/main.cpp
        src/mean_shift.cpp
        src/mean_shift_seq.cpp
        src/image_io.cpp
        src/benchmark.cpp)
target_include_directories(MeanShiftParallel PRIVATE src)
target_include_directories(MeanShiftParallel SYSTEM PRIVATE third_party)
target_link_libraries(MeanShiftParallel PRIVATE OpenMP::OpenMP_CXX)

# Niente -ffast-math: la riduzione floating point viene riordinata solo dove lo chiede
# esplicitamente #pragma omp simd reduction.
# -fno-trapping-math: senza, GCC non trasforma il confronto del cutoff in una select vettoriale
target_compile_options(MeanShiftParallel PRIVATE -O3 -march=native -fno-trapping-math)
# La baseline sequenziale è compilata senza vettorizzazione automatica
set_source_files_properties(src/mean_shift_seq.cpp PROPERTIES COMPILE_OPTIONS "-fno-tree-vectorize")

# Report di vettorizzazione di GCC: cmake -DVEC_REPORT=ON ...
option(VEC_REPORT "Stampa i report di vettorizzazione" OFF)
if(VEC_REPORT)
    target_compile_options(MeanShiftParallel PRIVATE -fopt-info-vec-optimized -fopt-info-vec-missed)
endif()
