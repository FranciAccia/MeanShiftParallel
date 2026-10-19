#include "mean_shift.h"

#include <omp.h>

#include <cstdlib>

#include "kernel.h"

namespace {

// Convergenza di un singolo punto: y viene spostato iterativamente verso la media pesata
// di tutti gli N punti del dataset (gradient ascent sulla KDE). Restituisce le iterazioni.
// La versione scalare di riferimento è in mean_shift_seq.cpp.

// Versione SIMD (SoA): accessi contigui ai tre array; il cutoff diventa una select branchless,
// così il loop è vettorizzabile (SSE/AVX: 4 o 8 float per istruzione).
int converge_simd_soa(const float* __restrict R, const float* __restrict G,
                      const float* __restrict B, int n, Pixel& y, const Kernel& k, int max_iter) {
    int it = 0;
    while (it < max_iter) {
        ++it;
        const float yr = y.r, yg = y.g, yb = y.b;
        float sr = 0.0f, sg = 0.0f, sb = 0.0f, sw = 0.0f;
        #pragma omp simd reduction(+ : sr, sg, sb, sw)
        for (int j = 0; j < n; ++j) {
            const float dr = R[j] - yr, dg = G[j] - yg, db = B[j] - yb;
            const float d2 = dr * dr + dg * dg + db * db;
            const float w = (d2 <= k.cutoff2) ? fast_exp(d2 * k.neg_inv_2h2) : 0.0f;
            sr += w * R[j];
            sg += w * G[j];
            sb += w * B[j];
            sw += w;
        }
        if (shift_and_check(y, sr, sg, sb, sw, k.eps2)) break;
    }
    return it;
}

// Versione SIMD sul layout AoS: stesso loop, ma accessi interleaved (stride di 3 float).
int converge_simd_aos(const Pixel* __restrict pts, int n, Pixel& y, const Kernel& k, int max_iter) {
    int it = 0;
    while (it < max_iter) {
        ++it;
        const float yr = y.r, yg = y.g, yb = y.b;
        float sr = 0.0f, sg = 0.0f, sb = 0.0f, sw = 0.0f;
        #pragma omp simd reduction(+ : sr, sg, sb, sw)
        for (int j = 0; j < n; ++j) {
            const float dr = pts[j].r - yr, dg = pts[j].g - yg, db = pts[j].b - yb;
            const float d2 = dr * dr + dg * dg + db * db;
            const float w = (d2 <= k.cutoff2) ? fast_exp(d2 * k.neg_inv_2h2) : 0.0f;
            sr += w * pts[j].r;
            sg += w * pts[j].g;
            sb += w * pts[j].b;
            sw += w;
        }
        if (shift_and_check(y, sr, sg, sb, sw, k.eps2)) break;
    }
    return it;
}

// Fase 1: shift di tutti i punti. Ogni punto è indipendente (si scrive solo modes[i]): nessuna
// sincronizzazione. Le iterazioni variano per punto, quindi lo scheduling è configurabile.

long long shift_simd(const Dataset& d, const Params& p, const Kernel& k, std::vector<Pixel>& modes) {
    const int n = d.size();
    long long total = 0;
    for (int i = 0; i < n; ++i) {
        Pixel y = d.aos[i];
        total += converge_simd_soa(d.r.data(), d.g.data(), d.b.data(), n, y, k, p.max_iter);
        modes[i] = y;
    }
    return total;
}

long long shift_par(const Dataset& d, const Params& p, const Kernel& k, std::vector<Pixel>& modes) {
    const int n = d.size();
    long long total = 0;
    #pragma omp parallel for default(none) shared(d, p, k, modes, n) schedule(runtime) \
            reduction(+ : total)
    for (int i = 0; i < n; ++i) {
        Pixel y = d.aos[i];
        total += converge_simd_soa(d.r.data(), d.g.data(), d.b.data(), n, y, k, p.max_iter);
        modes[i] = y;
    }
    return total;
}

long long shift_par_aos(const Dataset& d, const Params& p, const Kernel& k, std::vector<Pixel>& modes) {
    const int n = d.size();
    long long total = 0;
    #pragma omp parallel for default(none) shared(d, p, k, modes, n) schedule(runtime) \
            reduction(+ : total)
    for (int i = 0; i < n; ++i) {
        Pixel y = d.aos[i];
        total += converge_simd_aos(d.aos.data(), n, y, k, p.max_iter);
        modes[i] = y;
    }
    return total;
}

// Fase 2: rilevamento dei modi. Un modo diventa un nuovo centro se dista almeno merge_factor·h
// dai centri già trovati.

inline float dist2(const Pixel& a, const Pixel& b) {
    const float dr = a.r - b.r, dg = a.g - b.g, db = a.b - b.b;
    return dr * dr + dg * dg + db * db;
}

// Sequenziale in ordine di indice: i centri non dipendono dal numero di thread.
std::vector<Pixel> find_centers(const std::vector<Pixel>& modes, float radius2) {
    std::vector<Pixel> centers;
    for (const Pixel& m : modes) {
        bool found = false;
        for (const Pixel& c : centers) {
            if (dist2(m, c) < radius2) { found = true; break; }
        }
        if (!found) centers.push_back(m);
    }
    return centers;
}

inline int nearest_center(const Pixel& m, const std::vector<Pixel>& centers) {
    int best = 0;
    float best_d2 = dist2(m, centers[0]);
    for (int c = 1; c < static_cast<int>(centers.size()); ++c) {
        const float d2 = dist2(m, centers[c]);
        if (d2 < best_d2) { best_d2 = d2; best = c; }
    }
    return best;
}

}

Result mean_shift(const Dataset& data, const Params& params, Variant variant) {
    const Kernel k = make_kernel(params);
    const int n = data.size();
    const bool parallel = (variant == Variant::Par || variant == Variant::ParAoS);
    if (parallel) {
        omp_set_num_threads(params.threads);
        set_schedule(params.schedule);
    }

    Result res;
    res.modes.resize(n);
    res.labels.resize(n);

    const double t0 = omp_get_wtime();
    switch (variant) {
        case Variant::Seq:    res.total_iters = shift_seq(data, params, k, res.modes); break;
        case Variant::Simd:   res.total_iters = shift_simd(data, params, k, res.modes); break;
        case Variant::Par:    res.total_iters = shift_par(data, params, k, res.modes); break;
        case Variant::ParAoS: res.total_iters = shift_par_aos(data, params, k, res.modes); break;
    }
    const double t1 = omp_get_wtime();

    const float radius = params.merge_factor * params.bandwidth;
    res.centers = find_centers(res.modes, radius * radius);
    // Etichettatura: ogni pixel al centro più vicino al suo modo.
    #pragma omp parallel for default(none) shared(res, n) schedule(static) if (parallel)
    for (int i = 0; i < n; ++i) {
        res.labels[i] = nearest_center(res.modes[i], res.centers);
    }
    const double t2 = omp_get_wtime();

    res.t_shift = t1 - t0;
    res.t_modes = t2 - t1;
    return res;
}

const char* variant_name(Variant variant) {
    switch (variant) {
        case Variant::Seq:    return "seq";
        case Variant::Simd:   return "simd";
        case Variant::Par:    return "par";
        case Variant::ParAoS: return "par_aos";
    }
    return "?";
}

Variant parse_variant(const std::string& name) {
    if (name == "seq")     return Variant::Seq;
    if (name == "simd")    return Variant::Simd;
    if (name == "par_aos") return Variant::ParAoS;
    return Variant::Par;
}

void set_schedule(const std::string& schedule) {
    const std::string::size_type comma = schedule.find(',');
    const std::string kind = schedule.substr(0, comma);
    const int chunk = (comma == std::string::npos) ? 0 : std::atoi(schedule.c_str() + comma + 1);

    omp_sched_t sched;
    if (kind == "static")       sched = omp_sched_static;
    else if (kind == "dynamic") sched = omp_sched_dynamic;
    else                        sched = omp_sched_guided;
    omp_set_schedule(sched, chunk);  // chunk 0 = dimensione di default dell'implementazione
}
