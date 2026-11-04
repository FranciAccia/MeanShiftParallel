#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "mean_shift.h"

// Parti comuni a tutte le varianti: costanti del kernel, esponenziale e passo di shift.

// Costanti del kernel gaussiano K(d²) = exp(-d² / (2h²)), calcolate una volta per esecuzione.
struct Kernel {
    float neg_inv_2h2;   // -1 / (2h²)
    float cutoff2;       // (3h)²: oltre 3 deviazioni standard il peso è trascurabile (< 1.2e-2)
    float eps2;          // (eps * h)²: soglia di convergenza sullo shift
};

inline Kernel make_kernel(const Params& p) {
    const float h2 = p.bandwidth * p.bandwidth;
    return {-1.0f / (2.0f * h2), 9.0f * h2, p.eps * p.eps * h2};
}

// exp(x) per x in [-4.5, 0]: 2^k (nell'esponente IEEE 754) · polinomio di grado 5, errore < 4e-7.
// A differenza di std::exp è inline e senza branch, quindi vettorizzabile. La usano tutte le
// varianti, così seq vs simd misura solo l'effetto della vettorizzazione.
inline float fast_exp(float x) {
    // Limite inferiore per evitare overflow (i punti oltre il cutoff vengono poi scartati).
    // Select invece di std::fmax: GCC non vettorizza fmax su x86
    x = (x < -4.5f) ? -4.5f : x;
    const float t = x * 1.44269504f;
    const float k = std::floor(t);
    const float f = t - k;
    float p = 1.8775767e-3f;
    p = p * f + 8.9893397e-3f;
    p = p * f + 5.5826318e-2f;
    p = p * f + 2.4015361e-1f;
    p = p * f + 6.9315308e-1f;
    p = p * f + 1.0f;
    std::int32_t bits;
    std::memcpy(&bits, &p, sizeof(bits));
    bits += static_cast<std::int32_t>(k) << 23;   // moltiplica per 2^k
    std::memcpy(&p, &bits, sizeof(p));
    return p;
}

// Sposta y nella media pesata (sr, sg, sb) / sw e restituisce true se lo shift è sotto soglia.
inline bool shift_and_check(Pixel& y, float sr, float sg, float sb, float sw, float eps2) {
    if (sw <= 0.0f) return true;
    const Pixel next{sr / sw, sg / sw, sb / sw};
    const float dr = next.r - y.r, dg = next.g - y.g, db = next.b - y.b;
    y = next;
    return dr * dr + dg * dg + db * db < eps2;
}

// Baseline sequenziale scalare (mean_shift_seq.cpp, compilato con -fno-tree-vectorize).
long long shift_seq(const Dataset& d, const Params& p, const Kernel& k, std::vector<Pixel>& modes);
