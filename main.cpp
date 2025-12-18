#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <filesystem> // Richiede C++17

// Alias per brevità
namespace fs = std::filesystem;

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// --- CONFIGURAZIONE ---
const int TARGET_PIXELS = 10000; // Riduciamo un po' per processare molte immagini velocemente
const double BANDWIDTH = 25.0;
const std::string DATASET_FOLDER = "coco_dataset"; // Nome della cartella con le immagini
const std::string OUTPUT_FOLDER = "output_coco";   // Dove salvare i risultati

struct Pixel { double r, g, b; };

inline double get_color_dist_sq(const Pixel& p1, const Pixel& p2) {
    double dr = p1.r - p2.r;
    double dg = p1.g - p2.g;
    double db = p1.b - p2.b;
    return dr*dr + dg*dg + db*db;
}

inline double gaussian_kernel(double dist_sq, double bandwidth) {
    return std::exp(-dist_sq / (2 * bandwidth * bandwidth));
}

// --- ALGORITMO SEQUENZIALE ---
void mean_shift_seq(const std::vector<Pixel>& pixels, std::vector<Pixel>& shifted_pixels, double bandwidth) {
    int n = pixels.size();
    for (int i = 0; i < n; ++i) {
        double new_r = 0, new_g = 0, new_b = 0;
        double total_weight = 0;
        for (int j = 0; j < n; ++j) {
            double dist_sq = get_color_dist_sq(pixels[i], pixels[j]);
            if (dist_sq > 9 * bandwidth * bandwidth) continue;
            double weight = gaussian_kernel(dist_sq, bandwidth);
            new_r += pixels[j].r * weight;
            new_g += pixels[j].g * weight;
            new_b += pixels[j].b * weight;
            total_weight += weight;
        }
        if (total_weight > 0) {
            shifted_pixels[i].r = new_r / total_weight;
            shifted_pixels[i].g = new_g / total_weight;
            shifted_pixels[i].b = new_b / total_weight;
        }
    }
}

// --- ALGORITMO PARALLELO ---
void mean_shift_par(const std::vector<Pixel>& pixels, std::vector<Pixel>& shifted_pixels, double bandwidth) {
    int n = pixels.size();
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        double new_r = 0, new_g = 0, new_b = 0;
        double total_weight = 0;
        #pragma omp simd reduction(+:new_r, new_g, new_b, total_weight)
        for (int j = 0; j < n; ++j) {
            double dist_sq = get_color_dist_sq(pixels[i], pixels[j]);
            if (dist_sq > 9 * bandwidth * bandwidth) continue;
            double weight = gaussian_kernel(dist_sq, bandwidth);
            new_r += pixels[j].r * weight;
            new_g += pixels[j].g * weight;
            new_b += pixels[j].b * weight;
            total_weight += weight;
        }
        if (total_weight > 0) {
            shifted_pixels[i].r = new_r / total_weight;
            shifted_pixels[i].g = new_g / total_weight;
            shifted_pixels[i].b = new_b / total_weight;
        }
    }
}

int main() {
    // 1. PREPARAZIONE CARTELLE
    if (!fs::exists(DATASET_FOLDER)) {
        std::cerr << "ERRORE: Cartella '" << DATASET_FOLDER << "' non trovata!" << std::endl;
        std::cerr << "Crea la cartella e inserisci dentro le immagini COCO." << std::endl;
        return 1;
    }
    if (!fs::exists(OUTPUT_FOLDER)) {
        fs::create_directory(OUTPUT_FOLDER);
    }

    std::cout << ">> Scansione dataset COCO in '" << DATASET_FOLDER << "'..." << std::endl;
    std::vector<fs::path> image_files;
    for (const auto& entry : fs::directory_iterator(DATASET_FOLDER)) {
        auto ext = entry.path().extension().string();
        // Controllo estensione semplice (lowercase check sarebbe meglio ma richiede <cctype>)
        if (ext == ".jpg" || ext == ".png" || ext == ".jpeg" || ext == ".JPG") {
            image_files.push_back(entry.path());
        }
    }

    if (image_files.empty()) {
        std::cerr << "Nessuna immagine trovata." << std::endl;
        return 1;
    }

    std::cout << ">> Trovate " << image_files.size() << " immagini." << std::endl;

    // Variabili per statistiche globali
    double total_time_seq = 0.0;
    double total_time_par = 0.0;
    int processed_count = 0;

    // 2. CICLO SU OGNI IMMAGINE
    for (const auto& filepath : image_files) {
        std::string filename = filepath.filename().string();
        std::cout << "\n[" << processed_count + 1 << "/" << image_files.size() << "] Elaborazione " << filename << "..." << std::endl;

        int w, h, c;
        unsigned char* img_data = stbi_load(filepath.string().c_str(), &w, &h, &c, 3);
        if (!img_data) {
            std::cerr << "  Errore caricamento. Salto." << std::endl;
            continue;
        }

        // Conversione e Ridimensionamento (Obbligatorio per O(N^2))
        std::vector<Pixel> pixels;
        // Se l'immagine è troppo grande, prendiamo solo un subset o facciamo resize "brutale" (skip pixel)
        // Qui facciamo un semplice downsampling se necessario per stare nei TARGET_PIXELS
        int step = 1;
        int total_raw_pixels = w * h;
        if (total_raw_pixels > TARGET_PIXELS) {
            step = std::sqrt(total_raw_pixels / TARGET_PIXELS) + 1;
        }

        for (int y = 0; y < h; y += step) {
            for (int x = 0; x < w; x += step) {
                int idx = (y * w + x) * 3;
                pixels.push_back({(double)img_data[idx], (double)img_data[idx+1], (double)img_data[idx+2]});
            }
        }
        stbi_image_free(img_data);

        // Se dopo il resize abbiamo troppi pochi pixel (es. icona piccola), saltiamo
        if (pixels.size() < 1000) {
            std::cout << "  Troppo piccola (" << pixels.size() << " px). Salto." << std::endl;
            continue;
        }

        // Limitiamo esattamente al target per coerenza nei tempi
        if (pixels.size() > TARGET_PIXELS) pixels.resize(TARGET_PIXELS);

        std::vector<Pixel> buffer = pixels;
        int n = pixels.size();

        // --- BENCHMARK SEQUENZIALE ---
        auto t1 = std::chrono::high_resolution_clock::now();
        mean_shift_seq(pixels, buffer, BANDWIDTH);
        auto t2 = std::chrono::high_resolution_clock::now();
        double dt_seq = std::chrono::duration<double>(t2 - t1).count();
        total_time_seq += dt_seq;

        // --- BENCHMARK PARALLELO ---
        // Reset buffer
        std::vector<Pixel> input_par = pixels;
        std::vector<Pixel> output_par = pixels;

        auto t3 = std::chrono::high_resolution_clock::now();
        mean_shift_par(input_par, output_par, BANDWIDTH);
        auto t4 = std::chrono::high_resolution_clock::now();
        double dt_par = std::chrono::duration<double>(t4 - t3).count();
        total_time_par += dt_par;

        std::cout << "  Pixels: " << n << " | Seq: " << dt_seq << "s | Par: " << dt_par << "s | Speedup: " << std::fixed << std::setprecision(2) << dt_seq/dt_par << "x" << std::endl;

        // Salvataggio output (visualizzazione risultato)
        // Ricostruiamo un'immagine quadrata fittizia dai pixel segmentati
        int out_w = std::sqrt(n);
        int out_h = n / out_w;
        std::vector<unsigned char> out_data(out_w * out_h * 3);
        for(int i=0; i<out_w*out_h; ++i) {
            out_data[i*3] = (unsigned char)std::min(255.0, std::max(0.0, output_par[i].r));
            out_data[i*3+1] = (unsigned char)std::min(255.0, std::max(0.0, output_par[i].g));
            out_data[i*3+2] = (unsigned char)std::min(255.0, std::max(0.0, output_par[i].b));
        }
        std::string out_path = OUTPUT_FOLDER + "/seg_" + filename;
        stbi_write_png(out_path.c_str(), out_w, out_h, 3, out_data.data(), out_w*3);

        processed_count++;
    }

    // 3. REPORT FINALE
    std::cout << "\n==========================================" << std::endl;
    std::cout << "RISULTATI COMPLESSIVI SU " << processed_count << " IMMAGINI" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Tempo Totale Sequenziale: " << total_time_seq << " s" << std::endl;
    std::cout << "Tempo Totale Parallelo:   " << total_time_par << " s" << std::endl;

    if (total_time_par > 0) {
        double avg_speedup = total_time_seq / total_time_par;
        std::cout << "SPEEDUP MEDIO: " << avg_speedup << "x" << std::endl;
    }

    return 0;
}