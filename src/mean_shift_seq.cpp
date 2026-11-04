// Baseline sequenziale. Questo file è compilato con -fno-tree-vectorize (vedi CMakeLists.txt),
// così il loop interno resta scalare.

#include "kernel.h"

namespace {

// Convergenza di un singolo punto: y viene spostato iterativamente verso la media pesata
// di tutti gli N punti del dataset (gradient ascent sulla KDE). Restituisce le iterazioni.
// Layout AoS; i punti oltre il cutoff vengono saltati con un branch.
int converge_scalar(const Pixel* pts, int n, Pixel& y, const Kernel& k, int max_iter) {
    int it = 0;
    while (it < max_iter) {
        ++it;
        float sr = 0.0f, sg = 0.0f, sb = 0.0f, sw = 0.0f;
        for (int j = 0; j < n; ++j) {
            const float dr = pts[j].r - y.r, dg = pts[j].g - y.g, db = pts[j].b - y.b;
            const float d2 = dr * dr + dg * dg + db * db;
            if (d2 > k.cutoff2) continue;
            const float w = fast_exp(d2 * k.neg_inv_2h2);
            sr += w * pts[j].r;
            sg += w * pts[j].g;
            sb += w * pts[j].b;
            sw += w;
        }
        if (shift_and_check(y, sr, sg, sb, sw, k.eps2)) break;
    }
    return it;
}

}  // namespace

long long shift_seq(const Dataset& d, const Params& p, const Kernel& k, std::vector<Pixel>& modes) {
    const int n = d.size();
    long long total = 0;
    for (int i = 0; i < n; ++i) {
        Pixel y = d.aos[i];
        total += converge_scalar(d.aos.data(), n, y, k, p.max_iter);
        modes[i] = y;
    }
    return total;
}
