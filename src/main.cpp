// Mean shift clustering per la segmentazione di immagini: versione sequenziale, vettorizzata
// (SIMD) e parallela (OpenMP), con harness di benchmark.
//
// Uso:
//   MeanShiftParallel segment <input> <output.png> [--bandwidth H] [--variant V] [--threads T]
//                                                  [--schedule S]
//   MeanShiftParallel bench [all|correctness|scaling|bandwidth|schedule|layout]
//                           [--images DIR] [--results DIR] [--warmup N] [--reps N]

#include <omp.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "benchmark.h"
#include "image_io.h"
#include "mean_shift.h"

namespace {

const int DEFAULT_WARMUP = 1;   // run iniziali scartati (cold start)
const int DEFAULT_REPS = 5;     // run misurati per ogni configurazione

void usage() {
    std::cerr << "Uso:\n"
                 "  MeanShiftParallel segment <input> <output.png> [--bandwidth H] [--variant seq|simd|par|par_aos]\n"
                 "                                                [--threads T] [--schedule S]\n"
                 "  MeanShiftParallel bench [all|correctness|scaling|bandwidth|schedule|layout]\n"
                 "                          [--images DIR] [--results DIR] [--warmup N] [--reps N]\n";
}

// Restituisce il valore dell'opzione `name` (es. "--threads"), oppure `fallback` se assente.
const char* option(int argc, char** argv, const char* name, const char* fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
    }
    return fallback;
}

int cmd_segment(int argc, char** argv) {
    if (argc < 4) { usage(); return 1; }
    Params params;
    params.bandwidth = std::atof(option(argc, argv, "--bandwidth", "12"));
    params.threads = std::atoi(option(argc, argv, "--threads", "8"));
    params.schedule = option(argc, argv, "--schedule", params.schedule.c_str());
    Variant variant;
    if (!parse_variant(option(argc, argv, "--variant", "par"), variant) || params.bandwidth <= 0 ||
        params.threads < 1 || !set_schedule(params.schedule)) {
        usage();
        return 1;
    }

    Dataset data;
    if (!load_image(argv[2], data)) {
        std::cerr << "ERRORE: impossibile leggere l'immagine " << argv[2] << "\n";
        return 1;
    }
    std::cout << "Immagine " << argv[2] << ": " << data.width << "x" << data.height << " ("
              << data.size() << " pixel), variante " << variant_name(variant) << ", h = "
              << params.bandwidth << "\n";

    const double t0 = omp_get_wtime();
    const Result res = mean_shift(data, params, variant);
    const double elapsed = omp_get_wtime() - t0;

    std::cout << "Cluster trovati: " << res.centers.size() << ", iterazioni medie per punto: "
              << static_cast<double>(res.total_iters) / data.size() << "\n"
              << "Tempo: " << elapsed << " s (shift " << res.t_shift << " s, modi " << res.t_modes << " s)\n";
    if (!save_segmentation(argv[3], data, res)) {
        std::cerr << "ERRORE: impossibile scrivere " << argv[3] << "\n";
        return 1;
    }
    std::cout << "Immagine segmentata salvata in " << argv[3] << "\n";
    return 0;
}

int cmd_bench(int argc, char** argv) {
    const std::string which = (argc >= 3 && argv[2][0] != '-') ? argv[2] : "all";
    const int warmup = std::atoi(option(argc, argv, "--warmup", std::to_string(DEFAULT_WARMUP).c_str()));
    const int reps = std::atoi(option(argc, argv, "--reps", std::to_string(DEFAULT_REPS).c_str()));
    if (warmup < 0 || reps < 1) { usage(); return 1; }
    const bool ok = run_benchmarks(which, option(argc, argv, "--images", "images"),
                                   option(argc, argv, "--results", "results"), warmup, reps);
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "segment") == 0) return cmd_segment(argc, argv);
    if (argc >= 2 && std::strcmp(argv[1], "bench") == 0) return cmd_bench(argc, argv);
    usage();
    return 1;
}
