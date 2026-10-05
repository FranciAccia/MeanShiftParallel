#pragma once

#include <string>

// Esegue gli esperimenti di benchmark e scrive un CSV per esperimento in results_dir.
// which: "all" oppure il nome di un singolo esperimento
// (correctness, scaling, bandwidth, schedule, layout).
// Restituisce false se la verifica di correttezza fallisce.
bool run_benchmarks(const std::string& which, const std::string& images_dir,
                    const std::string& results_dir, int warmup, int reps);
