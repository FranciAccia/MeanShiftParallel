#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>

// --- LIBRERIE ESTERNE SINGLE-HEADER ---
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Struttura che rappresenta un colore nello spazio RGB
struct Point {
    double r, g, b; // Usiamo double per precisione durante i calcoli
};

// Distanza Euclidea al quadrato tra colori
inline double get_dist_sq(const Point& p1, const Point& p2) {
    double dr = p1.r - p2.r;
    double dg = p1.g - p2.g;
    double db = p1.b - p2.b;
    return dr*dr + dg*dg + db*db;
}

inline double gaussian_kernel(double dist_sq, double bandwidth) {
    return std::exp(-dist_sq / (2 * bandwidth * bandwidth));
}

// --- CORE DELL'ALGORITMO (PARALLELO) ---
void mean_shift_par(const std::vector<Point>& points, std::vector<Point>& shifted_points, double bandwidth) {
    int n = points.size();

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        double new_r = 0, new_g = 0, new_b = 0;
        double total_weight = 0;

        // Vettorizzazione SIMD sui colori
        #pragma omp simd reduction(+:new_r, new_g, new_b, total_weight)
        for (int j = 0; j < n; ++j) {
            double dist_sq = get_dist_sq(points[i], points[j]);

            // Ottimizzazione cutoff: se il colore è troppo diverso, ignoralo
            if (dist_sq > 9 * bandwidth * bandwidth) continue;

            double weight = gaussian_kernel(dist_sq, bandwidth);
            new_r += points[j].r * weight;
            new_g += points[j].g * weight;
            new_b += points[j].b * weight;
            total_weight += weight;
        }

        if (total_weight > 0) {
            shifted_points[i].r = new_r / total_weight;
            shifted_points[i].g = new_g / total_weight;
            shifted_points[i].b = new_b / total_weight;
        }
    }
}

int main() {
    // 1. CARICAMENTO IMMAGINE
    const char* input_file = "input.jpg"; // Assicurati che il file esista!
    int width, height, channels;

    // Carichiamo l'immagine
    unsigned char* img_data = stbi_load(input_file, &width, &height, &channels, 3); // Forziamo 3 canali (RGB)
    if (!img_data) {
        std::cerr << "Errore: Impossibile caricare l'immagine " << input_file << std::endl;
        std::cerr << "Suggerimento: Controlla il percorso o metti l'immagine nella cartella cmake-build-release." << std::endl;
        return 1;
    }

    int num_pixels = width * height;
    std::cout << "Immagine caricata: " << width << "x" << height << " (" << num_pixels << " pixel)" << std::endl;

    if (num_pixels > 20000) {
        std::cout << "ATTENZIONE: L'immagine e' grande. Il Mean Shift O(N^2) sara' lento." << std::endl;
        std::cout << "Consiglio: Usa un'immagine ridimensionata (es. 100x100) per test veloci." << std::endl;
    }

    // 2. CONVERSIONE PIXEL -> PUNTI (double)
    std::vector<Point> points(num_pixels);
    for (int i = 0; i < num_pixels; ++i) {
        points[i].r = static_cast<double>(img_data[i * 3 + 0]);
        points[i].g = static_cast<double>(img_data[i * 3 + 1]);
        points[i].b = static_cast<double>(img_data[i * 3 + 2]);
    }
    stbi_image_free(img_data); // Liberiamo la memoria originale

    // 3. ESECUZIONE MEAN SHIFT
    // Parametri: Bandwidth più alto = colori più appiattiti (es. 30.0 o 50.0)
    double bandwidth = 35.0;
    int iterations = 5;

    std::vector<Point> result = points;
    std::vector<Point> next_points = points;

    std::cout << "Inizio elaborazione parallela (" << iterations << " iterazioni)..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        std::cout << "Iterazione " << iter + 1 << "/" << iterations << "..." << std::endl;
        mean_shift_par(result, next_points, bandwidth);
        result = next_points;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Tempo totale: " << diff.count() << " s" << std::endl;

    // 4. SALVATAGGIO IMMAGINE
    // Riconvertiamo i double in unsigned char (0-255)
    std::vector<unsigned char> out_data(num_pixels * 3);
    for (int i = 0; i < num_pixels; ++i) {
        out_data[i * 3 + 0] = static_cast<unsigned char>(std::min(255.0, std::max(0.0, result[i].r)));
        out_data[i * 3 + 1] = static_cast<unsigned char>(std::min(255.0, std::max(0.0, result[i].g)));
        out_data[i * 3 + 2] = static_cast<unsigned char>(std::min(255.0, std::max(0.0, result[i].b)));
    }

    stbi_write_png("output.png", width, height, 3, out_data.data(), width * 3);
    std::cout << "Fatto! Immagine salvata come 'output.png'" << std::endl;

    return 0;
}