#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <random>

// Struttura Punto (es. Pixel RGB o coordinate spaziali)
struct Point {
    double x, y, z;
};

// Distanza Euclidea al quadrato (evitiamo sqrt per performance)
inline double get_dist_sq(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    double dz = p1.z - p2.z;
    return dx*dx + dy*dy + dz*dz;
}

// Kernel Gaussiano
inline double gaussian_kernel(double dist_sq, double bandwidth) {
    return std::exp(-dist_sq / (2 * bandwidth * bandwidth));
}

// --- VERSIONE SEQUENZIALE ---
void mean_shift_seq(const std::vector<Point>& points, std::vector<Point>& shifted_points, double bandwidth) {
    int n = points.size();
    for (int i = 0; i < n; ++i) {
        double new_x = 0, new_y = 0, new_z = 0;
        double total_weight = 0;

        for (int j = 0; j < n; ++j) {
            double dist_sq = get_dist_sq(points[i], points[j]);
            if (dist_sq > 9 * bandwidth * bandwidth) continue; // Cutoff per velocità

            double weight = gaussian_kernel(dist_sq, bandwidth);
            new_x += points[j].x * weight;
            new_y += points[j].y * weight;
            new_z += points[j].z * weight;
            total_weight += weight;
        }

        if (total_weight > 0) {
            shifted_points[i].x = new_x / total_weight;
            shifted_points[i].y = new_y / total_weight;
            shifted_points[i].z = new_z / total_weight;
        }
    }
}

// --- VERSIONE PARALLELA (OpenMP) ---
void mean_shift_par(const std::vector<Point>& points, std::vector<Point>& shifted_points, double bandwidth) {
    int n = points.size();

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        double new_x = 0, new_y = 0, new_z = 0;
        double total_weight = 0;

        #pragma omp simd reduction(+:new_x, new_y, new_z, total_weight)
        for (int j = 0; j < n; ++j) {
            double dist_sq = get_dist_sq(points[i], points[j]);
            if (dist_sq > 9 * bandwidth * bandwidth) continue;

            double weight = gaussian_kernel(dist_sq, bandwidth);
            new_x += points[j].x * weight;
            new_y += points[j].y * weight;
            new_z += points[j].z * weight;
            total_weight += weight;
        }

        if (total_weight > 0) {
            shifted_points[i].x = new_x / total_weight;
            shifted_points[i].y = new_y / total_weight;
            shifted_points[i].z = new_z / total_weight;
        }
    }
}

std::vector<Point> generate_data(int n) {
    std::vector<Point> data(n);
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0.0, 100.0);
    for (auto& p : data) {
        p.x = dis(gen); p.y = dis(gen); p.z = dis(gen);
    }
    return data;
}

int main() {
    // PARAMETRI DI TEST
    // Aumenta 'num_points' per vedere speedup maggiori (es. 5000, 10000, 20000)
    int num_points = 5000;
    int iterations = 5;
    double bandwidth = 15.0;

    std::cout << "---- MEAN SHIFT CLUSTERING BENCHMARK ----\n";
    std::cout << "Punti: " << num_points << " | Iterazioni: " << iterations << "\n";
    std::cout << "Processori disponibili: " << omp_get_num_procs() << "\n";

    auto points = generate_data(num_points);
    auto result_seq = points;
    auto result_par = points;

    // 1. Test Sequenziale
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) {
        std::vector<Point> next = result_seq;
        mean_shift_seq(result_seq, next, bandwidth);
        result_seq = next;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> t_seq = end - start;
    std::cout << "Tempo Sequenziale: " << t_seq.count() << "s\n";

    // 2. Test Parallelo
    start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) {
        std::vector<Point> next = result_par;
        mean_shift_par(result_par, next, bandwidth);
        result_par = next;
    }
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> t_par = end - start;
    std::cout << "Tempo Parallelo:   " << t_par.count() << "s\n";

    // Risultati
    std::cout << "---------------------------------------\n";
    std::cout << "SPEEDUP: " << t_seq.count() / t_par.count() << "x\n";

    return 0;
}