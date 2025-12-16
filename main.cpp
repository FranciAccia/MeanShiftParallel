#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <algorithm> // per std::min, std::max

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

struct Point {
    double r, g, b;
};

inline double get_dist_sq(const Point& p1, const Point& p2) {
    double dr = p1.r - p2.r;
    double dg = p1.g - p2.g;
    double db = p1.b - p2.b;
    return dr*dr + dg*dg + db*db;
}

inline double gaussian_kernel(double dist_sq, double bandwidth) {
    return std::exp(-dist_sq / (2 * bandwidth * bandwidth));
}

// ==========================================
// 1. VERSIONE SEQUENZIALE
// ==========================================
void mean_shift_seq(const std::vector<Point>& points, std::vector<Point>& shifted_points, double bandwidth) {
    int n = points.size();

    for (int i = 0; i < n; ++i) {
        double new_r = 0, new_g = 0, new_b = 0;
        double total_weight = 0;

        for (int j = 0; j < n; ++j) {
            double dist_sq = get_dist_sq(points[i], points[j]);

            // Cutoff optimization
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

// ==========================================
// 2. VERSIONE PARALLELA
// ==========================================
void mean_shift_par(const std::vector<Point>& points, std::vector<Point>& shifted_points, double bandwidth) {
    int n = points.size();

    // schedule(dynamic): alcuni pixel richiedono meno tempo rispetto ad altri
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        double new_r = 0, new_g = 0, new_b = 0;
        double total_weight = 0;

        // Vettorizzazione SIMD sui calcoli interni
        #pragma omp simd reduction(+:new_r, new_g, new_b, total_weight)
        for (int j = 0; j < n; ++j) {
            double dist_sq = get_dist_sq(points[i], points[j]);

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
    // --- CONFIG ---
    const char* input_file = "input.jpg";
    double bandwidth = 20.0; // Più alto = colori più "piatti" (posterizzazione)
    int iterations = 5;      // Bastano poche iterazioni

    int width, height, channels;
    unsigned char* img_data = stbi_load(input_file, &width, &height, &channels, 3);

    if (!img_data) {
        std::cerr << "ERRORE: Immagine non trovata! Inserisci 'input.jpg' nella cartella di build." << std::endl;
        return 1;
    }

    int num_pixels = width * height;
    std::cout << "Immagine caricata: " << width << "x" << height << " (" << num_pixels << " pixel)" << std::endl;

    // Safety check
    if (num_pixels > 10000) {
        std::cout << "\n[ATTENZIONE] L'immagine e' grande (" << num_pixels << " px)." << std::endl;
        std::cout << "L'algoritmo O(N^2) sara' LENTO in sequenziale." << std::endl;
        std::cout << "Si consiglia di ridimensionare l'immagine a max 100x100 pixel per i test." << std::endl;
    }

    // Data conversion for vefiry points
    std::vector<Point> points_initial(num_pixels);
    for (int i = 0; i < num_pixels; ++i) {
        points_initial[i].r = static_cast<double>(img_data[i * 3 + 0]);
        points_initial[i].g = static_cast<double>(img_data[i * 3 + 1]);
        points_initial[i].b = static_cast<double>(img_data[i * 3 + 2]);
    }
    stbi_image_free(img_data);

    // Two different copies of the image for the diff tests
    std::vector<Point> data_seq = points_initial;
    std::vector<Point> buffer_seq = points_initial;

    std::vector<Point> data_par = points_initial;
    std::vector<Point> buffer_par = points_initial;

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Inizio Benchmark (Iterations: " << iterations << ", Bandwidth: " << bandwidth << ")" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // --- 2. SEQUENTIAL TEST ---
    std::cout << "Esecuzione Sequenziale in corso..." << std::endl;
    auto start_seq = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        mean_shift_seq(data_seq, buffer_seq, bandwidth);
        data_seq = buffer_seq;
    }

    auto end_seq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_seq = end_seq - start_seq;
    std::cout << ">> Tempo Sequenziale: " << time_seq.count() << " s" << std::endl;

    // --- 3. PARALLEL TEST ---
    std::cout << "Esecuzione Parallela in corso (" << omp_get_max_threads() << " threads)..." << std::endl;
    auto start_par = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        mean_shift_par(data_par, buffer_par, bandwidth);
        data_par = buffer_par;
    }

    auto end_par = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_par = end_par - start_par;
    std::cout << ">> Tempo Parallelo:   " << time_par.count() << " s" << std::endl;

    // UTILS FOR THE REPORT
    double speedup = time_seq.count() / time_par.count();
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "SPEEDUP OTTENUTO: " << speedup << "x" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // OUTPUT SAVE (JUST THE PARALLEL ONE)
    std::vector<unsigned char> out_data(num_pixels * 3);
    for (int i = 0; i < num_pixels; ++i) {
        out_data[i * 3 + 0] = static_cast<unsigned char>(std::min(255.0, std::max(0.0, data_par[i].r)));
        out_data[i * 3 + 1] = static_cast<unsigned char>(std::min(255.0, std::max(0.0, data_par[i].g)));
        out_data[i * 3 + 2] = static_cast<unsigned char>(std::min(255.0, std::max(0.0, data_par[i].b)));
    }

    stbi_write_png("output_segmented.png", width, height, 3, out_data.data(), width * 3);
    std::cout << "Immagine salvata come 'output_segmented.png'" << std::endl;

    return 0;
}