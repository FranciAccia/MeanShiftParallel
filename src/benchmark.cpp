#include "benchmark.h"

#include <omp.h>
#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

#include "image_io.h"
#include "mean_shift.h"

namespace {

// ------------------------------------------------------------------------------------------
// Configurazione degli esperimenti
// ------------------------------------------------------------------------------------------

const char* MAIN_IMAGE = "118015";                    // immagine principale (barche)
const std::vector<std::string> ALL_IMAGES = {"118015", "108004", "145059"};
const int MAIN_SIZE = 128;                            // 128x128: sequenziale ≥ 10 s con h = 12
const float MAIN_BANDWIDTH = 12.0f;
const int MAIN_THREADS = 8;                           // numero di core logici dell'M2

const std::vector<int> SCALING_SIZES = {64, 96, 128};
const std::vector<int> SCALING_THREADS = {1, 2, 4, 6, 8, 12, 16};   // fino a 2x i core
const int BANDWIDTH_SIZE = 96;                        // più piccola: lo sweep esegue anche seq
const std::vector<float> BANDWIDTHS = {6, 8, 12, 16, 24, 32};
const std::vector<std::string> SCHEDULES = {
    "static", "static,1", "static,64",
    "dynamic,1", "dynamic,16", "dynamic,256",
    "guided",
};
// Pausa dopo ogni run pari alla sua durata (max 5 s): evita il thermal throttling del
// MacBook Air M2 (senza ventola), che sotto carico continuo allunga i tempi fino al 50%.
const double MAX_COOLDOWN_S = 5.0;

const int CORRECTNESS_SIZE = 64;
// Differenza massima tra i modi, come frazione di h: la riduzione SIMD somma in ordine diverso,
// quindi le varianti si fermano a pochi eps·h di distanza. Cluster ed etichette devono coincidere.
const float CORRECTNESS_TOL = 0.01f;

// ------------------------------------------------------------------------------------------
// Misura dei tempi
// ------------------------------------------------------------------------------------------

// Tempo CPU del processo (user + system), sommato su tutti i thread.
double cpu_seconds() {
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_utime.tv_sec + usage.ru_utime.tv_usec * 1e-6 +
           usage.ru_stime.tv_sec + usage.ru_stime.tv_usec * 1e-6;
}

struct Measurement {
    int reps = 0;
    double mean = 0, stdev = 0, min = 0, max = 0, ci95 = 0;   // wall-clock (s)
    double cpu_mean = 0;                                        // tempo CPU medio (s)
    double modes_mean = 0;                                      // tempo medio della fase 2 (s)
    size_t clusters = 0;
    double avg_iter = 0;
};

// Esegue `warmup` run scartati e `reps` run misurati. Si misura solo il clustering:
// caricamento dell'immagine e conversione nei due layout avvengono prima.
void cooldown(double elapsed) {
    std::this_thread::sleep_for(std::chrono::duration<double>(std::min(elapsed, MAX_COOLDOWN_S)));
}

Measurement measure(const Dataset& data, const Params& params, Variant variant, int warmup, int reps) {
    for (int i = 0; i < warmup; ++i) {
        const double t0 = omp_get_wtime();
        mean_shift(data, params, variant);
        cooldown(omp_get_wtime() - t0);
    }

    std::vector<double> wall, cpu, modes;
    Result last;
    for (int i = 0; i < reps; ++i) {
        const double c0 = cpu_seconds();
        const double t0 = omp_get_wtime();
        last = mean_shift(data, params, variant);
        wall.push_back(omp_get_wtime() - t0);
        cpu.push_back(cpu_seconds() - c0);
        modes.push_back(last.t_modes);
        cooldown(wall.back());
    }

    auto mean_of = [](const std::vector<double>& v) {
        double s = 0;
        for (double x : v) s += x;
        return s / v.size();
    };
    Measurement m;
    m.reps = reps;
    m.mean = mean_of(wall);
    double var = 0;
    for (double x : wall) var += (x - m.mean) * (x - m.mean);
    m.stdev = reps > 1 ? std::sqrt(var / (reps - 1)) : 0.0;
    m.min = *std::min_element(wall.begin(), wall.end());
    m.max = *std::max_element(wall.begin(), wall.end());
    m.ci95 = 1.96 * m.stdev / std::sqrt(static_cast<double>(reps));   // approssimazione normale
    m.cpu_mean = mean_of(cpu);
    m.modes_mean = mean_of(modes);
    m.clusters = last.centers.size();
    m.avg_iter = static_cast<double>(last.total_iters) / data.size();
    return m;
}

// ------------------------------------------------------------------------------------------
// Output: una riga CSV per misura (stesso schema per tutti gli esperimenti) + stampa a terminale
// ------------------------------------------------------------------------------------------

class CsvWriter {
public:
    explicit CsvWriter(const std::string& path) : out_(path) {
        out_ << "image,size,n_pixels,variant,threads,schedule,bandwidth,reps,"
                "mean_s,std_s,min_s,max_s,ci95_s,cpu_s,modes_s,clusters,avg_iter\n";
    }
    void row(const std::string& image, const Dataset& data, Variant variant, const Params& p,
             const Measurement& m) {
        const bool parallel = (variant == Variant::Par || variant == Variant::ParAoS);
        char line[512];
        std::snprintf(line, sizeof(line),
                      "%s,%d,%d,%s,%d,\"%s\",%g,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%zu,%.2f\n",
                      image.c_str(), data.width, data.size(), variant_name(variant),
                      parallel ? p.threads : 1, parallel ? p.schedule.c_str() : "-", p.bandwidth,
                      m.reps, m.mean, m.stdev, m.min, m.max, m.ci95, m.cpu_mean, m.modes_mean,
                      m.clusters, m.avg_iter);
        out_ << line;
        out_.flush();
    }
    bool ok() const { return out_.good(); }

private:
    std::ofstream out_;
};

void print_measurement(const char* label, const Measurement& m, double baseline = 0.0) {
    std::printf("  %-28s %8.3f s ± %.3f  [min %.3f, max %.3f]  CPU %7.2f s  K=%zu  iter=%.1f",
                label, m.mean, m.ci95, m.min, m.max, m.cpu_mean, m.clusters, m.avg_iter);
    if (baseline > 0) std::printf("  speedup %.2fx", baseline / m.mean);
    std::printf("\n");
    std::fflush(stdout);
}

std::string image_path(const std::string& dir, const std::string& id, int size) {
    return dir + "/" + id + "_" + std::to_string(size) + ".png";
}

bool load_or_complain(const std::string& path, Dataset& data) {
    if (load_image(path, data)) return true;
    std::cerr << "ERRORE: immagine non trovata: " << path
              << " (generarla con scripts/prepare_images.py)\n";
    return false;
}

Params make_params(float bandwidth, int threads = 1, const std::string& schedule = "dynamic,16") {
    Params p;
    p.bandwidth = bandwidth;
    p.threads = threads;
    p.schedule = schedule;
    return p;
}

// ------------------------------------------------------------------------------------------
// Esperimenti
// ------------------------------------------------------------------------------------------

// [0] Verifica di correttezza: stessi modi (entro la tolleranza), stessi cluster e stesse
// etichette della baseline sequenziale.
bool exp_correctness(const std::string& images_dir) {
    std::printf("\n[0] Verifica di correttezza (immagini %dx%d, h = %g)\n", CORRECTNESS_SIZE,
                CORRECTNESS_SIZE, MAIN_BANDWIDTH);
    bool all_ok = true;
    for (const std::string& id : ALL_IMAGES) {
        Dataset data;
        if (!load_or_complain(image_path(images_dir, id, CORRECTNESS_SIZE), data)) return false;
        const Result ref = mean_shift(data, make_params(MAIN_BANDWIDTH), Variant::Seq);
        for (Variant v : {Variant::Simd, Variant::Par, Variant::ParAoS}) {
            const Result res = mean_shift(data, make_params(MAIN_BANDWIDTH, MAIN_THREADS), v);
            float max_diff = 0;
            int same_labels = 0;
            for (int i = 0; i < data.size(); ++i) {
                max_diff = std::max({max_diff, std::abs(res.modes[i].r - ref.modes[i].r),
                                     std::abs(res.modes[i].g - ref.modes[i].g),
                                     std::abs(res.modes[i].b - ref.modes[i].b)});
                same_labels += (res.labels[i] == ref.labels[i]);
            }
            const bool ok = max_diff <= CORRECTNESS_TOL * MAIN_BANDWIDTH && res.centers.size() == ref.centers.size() &&
                            same_labels == data.size();
            all_ok = all_ok && ok;
            std::printf("  %s  %-8s vs seq: max |Δmodo| = %.2e, cluster %zu/%zu, etichette uguali %d/%d  %s\n",
                        id.c_str(), variant_name(v), max_diff, res.centers.size(), ref.centers.size(),
                        same_labels, data.size(), ok ? "OK" : "FALLITO");
        }
    }
    return all_ok;
}

// [1] Strong scaling: dimensione fissa, numero di thread crescente, per più dimensioni
// dell'input. Per ogni dimensione: baseline seq, simd (solo vettorizzazione) e par.
bool exp_scaling(const std::string& images_dir, const std::string& results_dir, int warmup, int reps) {
    std::printf("\n[1] Strong scaling: speedup vs thread per più dimensioni (h = %g)\n", MAIN_BANDWIDTH);
    CsvWriter csv(results_dir + "/scaling.csv");
    for (int size : SCALING_SIZES) {
        Dataset data;
        if (!load_or_complain(image_path(images_dir, MAIN_IMAGE, size), data)) return false;
        std::printf(" %dx%d (%d pixel)\n", size, size, data.size());

        Params p = make_params(MAIN_BANDWIDTH);
        const Measurement seq = measure(data, p, Variant::Seq, warmup, reps);
        csv.row(MAIN_IMAGE, data, Variant::Seq, p, seq);
        print_measurement("seq", seq);

        const Measurement simd = measure(data, p, Variant::Simd, warmup, reps);
        csv.row(MAIN_IMAGE, data, Variant::Simd, p, simd);
        print_measurement("simd", simd, seq.mean);

        for (int t : SCALING_THREADS) {
            p.threads = t;
            const Measurement par = measure(data, p, Variant::Par, warmup, reps);
            csv.row(MAIN_IMAGE, data, Variant::Par, p, par);
            const std::string label = "par " + std::to_string(t) + " thread";
            print_measurement(label.c_str(), par, seq.mean);
        }
    }
    return csv.ok();
}

// [2] Effetto del bandwidth: numero di cluster, iterazioni medie, tempi seq e par.
// Salva anche le immagini segmentate per ogni bandwidth (figure della relazione).
bool exp_bandwidth(const std::string& images_dir, const std::string& results_dir, int warmup, int reps) {
    std::printf("\n[2] Effetto del bandwidth (%dx%d, par con %d thread)\n", BANDWIDTH_SIZE,
                BANDWIDTH_SIZE, MAIN_THREADS);
    CsvWriter csv(results_dir + "/bandwidth.csv");
    Dataset data;
    if (!load_or_complain(image_path(images_dir, MAIN_IMAGE, BANDWIDTH_SIZE), data)) return false;
    for (float h : BANDWIDTHS) {
        Params p = make_params(h, MAIN_THREADS);
        const Measurement seq = measure(data, p, Variant::Seq, warmup, reps);
        csv.row(MAIN_IMAGE, data, Variant::Seq, p, seq);
        const Measurement par = measure(data, p, Variant::Par, warmup, reps);
        csv.row(MAIN_IMAGE, data, Variant::Par, p, par);
        std::printf(" h = %g\n", h);
        print_measurement("seq", seq);
        print_measurement("par", par, seq.mean);
    }

    // Segmentazioni di esempio di tutte le immagini (dimensione principale), fuori dalle misure
    for (const std::string& id : ALL_IMAGES) {
        Dataset img;
        if (!load_or_complain(image_path(images_dir, id, MAIN_SIZE), img)) return false;
        for (float h : BANDWIDTHS) {
            const Result res = mean_shift(img, make_params(h, MAIN_THREADS), Variant::Par);
            char name[128];
            std::snprintf(name, sizeof(name), "/seg_%s_h%02d.png", id.c_str(), static_cast<int>(h));
            save_segmentation(results_dir + name, img, res);
        }
    }
    std::printf("  immagini segmentate salvate in %s/seg_*.png\n", results_dir.c_str());
    return csv.ok();
}

// [3] Strategie di scheduling OpenMP: il lavoro per punto non è uniforme (numero di
// iterazioni diverso), quindi il bilanciamento del carico conta.
bool exp_schedule(const std::string& images_dir, const std::string& results_dir, int warmup, int reps) {
    std::printf("\n[3] Scheduling OpenMP (%dx%d, h = %g, %d thread)\n", MAIN_SIZE, MAIN_SIZE,
                MAIN_BANDWIDTH, MAIN_THREADS);
    CsvWriter csv(results_dir + "/schedule.csv");
    Dataset data;
    if (!load_or_complain(image_path(images_dir, MAIN_IMAGE, MAIN_SIZE), data)) return false;
    for (const std::string& s : SCHEDULES) {
        const Params p = make_params(MAIN_BANDWIDTH, MAIN_THREADS, s);
        const Measurement m = measure(data, p, Variant::Par, warmup, reps);
        csv.row(MAIN_IMAGE, data, Variant::Par, p, m);
        print_measurement(s.c_str(), m);
    }
    return csv.ok();
}

// [4] Layout di memoria: AoS (r g b interleaved) vs SoA (tre array separati), stesso codice SIMD.
bool exp_layout(const std::string& images_dir, const std::string& results_dir, int warmup, int reps) {
    std::printf("\n[4] Layout di memoria AoS vs SoA (%dx%d, h = %g)\n", MAIN_SIZE, MAIN_SIZE, MAIN_BANDWIDTH);
    CsvWriter csv(results_dir + "/layout.csv");
    Dataset data;
    if (!load_or_complain(image_path(images_dir, MAIN_IMAGE, MAIN_SIZE), data)) return false;
    for (int t : {1, MAIN_THREADS}) {
        for (Variant v : {Variant::Par, Variant::ParAoS}) {
            const Params p = make_params(MAIN_BANDWIDTH, t);
            const Measurement m = measure(data, p, v, warmup, reps);
            csv.row(MAIN_IMAGE, data, v, p, m);
            const std::string label = std::string(v == Variant::Par ? "SoA" : "AoS") + ", " +
                                      std::to_string(t) + " thread";
            print_measurement(label.c_str(), m);
        }
    }
    return csv.ok();
}

}  // namespace

bool run_benchmarks(const std::string& which, const std::string& images_dir,
                    const std::string& results_dir, int warmup, int reps) {
    const bool all = (which == "all");
    if (!all && which != "correctness" && which != "scaling" && which != "bandwidth" &&
        which != "schedule" && which != "layout") {
        std::cerr << "ERRORE: esperimento sconosciuto '" << which << "'\n";
        return false;
    }
    std::filesystem::create_directories(results_dir);
    std::printf("Benchmark mean shift — warm-up %d, ripetizioni %d, core logici %d\n",
                warmup, reps, omp_get_num_procs());

    // La verifica di correttezza precede sempre le misure: se fallisce non si misura nulla
    if (!exp_correctness(images_dir)) {
        std::cerr << "ERRORE: verifica di correttezza fallita, benchmark interrotto\n";
        return false;
    }
    bool ok = true;
    if (all || which == "scaling")   ok = exp_scaling(images_dir, results_dir, warmup, reps) && ok;
    if (all || which == "bandwidth") ok = exp_bandwidth(images_dir, results_dir, warmup, reps) && ok;
    if (all || which == "schedule")  ok = exp_schedule(images_dir, results_dir, warmup, reps) && ok;
    if (all || which == "layout")    ok = exp_layout(images_dir, results_dir, warmup, reps) && ok;
    std::printf("\nRisultati salvati in %s/*.csv\n", results_dir.c_str());
    return ok;
}
