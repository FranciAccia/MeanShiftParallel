#pragma once

#include <string>
#include <vector>

// Un pixel nello spazio colore RGB (valori in [0, 255]). È l'elemento del layout AoS.
struct Pixel {
    float r, g, b;
};

// Il dataset (i pixel di un'immagine) memorizzato in entrambi i layout di memoria.
// Viene costruito una sola volta, prima delle misure: la conversione non entra nei tempi.
struct Dataset {
    int width = 0;
    int height = 0;
    std::vector<Pixel> aos;          // Array of Structures: r g b r g b ...
    std::vector<float> r, g, b;      // Structure of Arrays: r r r ... g g g ... b b b ...

    int size() const { return static_cast<int>(aos.size()); }
};

// Le varianti implementate:
//  - Seq    : sequenziale scalare, layout AoS (baseline)
//  - Simd   : 1 thread, loop interno vettorizzato con #pragma omp simd, layout SoA
//  - Par    : OpenMP parallel for sui punti + loop interno SIMD, layout SoA
//  - ParAoS : come Par ma sul layout AoS (solo per il confronto AoS vs SoA)
enum class Variant { Seq, Simd, Par, ParAoS };

struct Params {
    float bandwidth = 20.0f;         // h: deviazione standard del kernel gaussiano
    float eps = 1e-3f;               // convergenza quando lo shift è < eps * h
    int max_iter = 100;              // limite di iterazioni per punto
    float merge_factor = 0.5f;       // modi distanti meno di merge_factor * h sono lo stesso cluster
    int threads = 1;                 // usato solo dalle varianti parallele
    std::string schedule = "dynamic,16";  // scheduling OpenMP delle varianti parallele
};

struct Result {
    std::vector<Pixel> modes;        // modo (punto di convergenza) di ogni pixel
    std::vector<Pixel> centers;      // un centro per ogni cluster trovato
    std::vector<int> labels;         // cluster di appartenenza di ogni pixel
    long long total_iters = 0;       // somma delle iterazioni di tutti i punti
    double t_shift = 0.0;            // tempo della fase di shift (s)
    double t_modes = 0.0;            // tempo del rilevamento dei modi + etichettatura (s)
};

// Esegue il mean shift clustering completo: shift di ogni punto fino a convergenza,
// rilevamento dei modi e assegnazione di ogni pixel a un cluster.
Result mean_shift(const Dataset& data, const Params& params, Variant variant);

const char* variant_name(Variant variant);
Variant parse_variant(const std::string& name);

// Imposta lo scheduling usato da schedule(runtime), es. "static", "dynamic,16", "guided".
void set_schedule(const std::string& schedule);
