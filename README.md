# Mean Shift Parallel

Progetto per il corso di **Parallel Programming** (Università degli Studi di Firenze).

Implementazione in C++17 del **mean shift clustering** applicato alla segmentazione di immagini
a colori, in versione sequenziale, vettorizzata (SIMD) e parallela (OpenMP), con un harness di
benchmark per l'analisi delle prestazioni.

Il mean shift non richiede di specificare il numero di cluster: ogni pixel, visto come punto
nello spazio colore RGB, viene spostato iterativamente verso la media pesata dei punti vicini
(gradient ascent sulla stima di densità a kernel, KDE) fino a convergere a un **modo** della
densità. I pixel che convergono allo stesso modo formano un cluster. L'unico parametro è il
**bandwidth** `h` del kernel gaussiano. Il costo è **O(N²)** per iterazione, con N numero di pixel.

## Algoritmo

1. **Shift** (O(N²·iterazioni)): per ogni punto `y`, finché lo spostamento è ≥ `eps·h`:
   `y ← Σⱼ K(‖y − xⱼ‖) xⱼ / Σⱼ K(‖y − xⱼ‖)`, con `K(d) = exp(−d² / 2h²)` e cutoff a `3h`.
2. **Rilevamento dei modi** (O(N·K)): i modi distanti meno di `h/2` vengono uniti in un unico centro.
3. **Etichettatura**: ogni pixel viene assegnato al centro più vicino al suo modo e, nell'immagine
   segmentata, ne prende il colore.

## Varianti implementate

| Variante  | Parallelismo                                         | Layout |
|-----------|------------------------------------------------------|--------|
| `seq`     | nessuno, scalare (baseline)                          | AoS    |
| `simd`    | 1 thread, loop interno con `#pragma omp simd`        | SoA    |
| `par`     | `#pragma omp parallel for` sui punti + loop SIMD     | SoA    |
| `par_aos` | come `par` (per il confronto AoS vs SoA)             | AoS    |

Ogni punto converge indipendentemente dagli altri: il dataset viene solo letto e ogni thread
scrive solo il modo dei propri punti, quindi non serve alcuna sincronizzazione. Il loop interno è
reso vettorizzabile con una select branchless al posto del cutoff e con un'esponenziale
polinomiale inline (errore relativo < 4·10⁻⁷), usata da tutte le varianti.

## Dataset

Immagini del [Berkeley Segmentation Dataset (BSDS500)](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)
(test set: 118015, 108004, 145059). Preprocessing (`scripts/prepare_images.py`): crop quadrato
centrale e ridimensionamento a 64², 96², 128² pixel con filtro Lanczos. Il
ridimensionamento è necessario per via della complessità O(N²). Le immagini preparate sono già in
`images/`.

## Requisiti

- Linux con GCC (testato su Debian 13: `sudo apt install g++ cmake`; OpenMP è incluso in GCC)
- CMake ≥ 3.20
- Python 3 con `numpy`, `matplotlib`, `pillow` (solo per preprocessing e grafici)

## Compilazione ed esecuzione

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Segmentare un'immagine
./build/MeanShiftParallel segment images/118015_128.png out.png --bandwidth 12 --threads 8

# Benchmark completo (~20 min): verifica di correttezza + tutti gli esperimenti
./build/MeanShiftParallel bench
# oppure un singolo esperimento: correctness | scaling | bandwidth | schedule | layout
./build/MeanShiftParallel bench scaling --reps 5 --warmup 1

# Grafici (figures/*.png) a partire da results/*.csv
python3 -m venv .venv && .venv/bin/pip install numpy matplotlib pillow
.venv/bin/python scripts/plot_results.py
```

Opzioni di `segment`: `--bandwidth H` (default 12), `--variant seq|simd|par|par_aos`
(default `par`), `--threads T` (default 8), `--schedule S` (es. `static`, `dynamic,16`, `guided`).

Report di vettorizzazione di GCC: `cmake -S . -B build -DVEC_REPORT=ON`.

## Esperimenti

Prima di ogni benchmark una **verifica di correttezza** confronta le varianti ottimizzate con la
baseline sequenziale (stessi cluster, stesse etichette per ogni pixel, modi entro l'1% di `h`) e
interrompe l'esecuzione se fallisce. Ogni misura usa 1 run di warm-up scartato e 5 run misurati
(media, deviazione standard, min, max, intervallo di confidenza al 95%, tempo CPU). Si misura solo
il clustering: lettura dell'immagine e conversione dei layout sono escluse.

| Esperimento | Cosa misura | Output |
|-------------|-------------|--------|
| `scaling`   | strong scaling: 1–16 thread su 3 dimensioni, seq e simd come riferimento | `results/scaling.csv` |
| `bandwidth` | cluster, iterazioni e tempi per h ∈ {6, 8, 12, 16, 24, 32} | `results/bandwidth.csv`, `results/seg_*.png` |
| `schedule`  | `static`, `dynamic`, `guided` con diversi chunk size | `results/schedule.csv` |
| `layout`    | AoS vs SoA, con 1 e 8 thread | `results/layout.csv` |

## Struttura

```
src/main.cpp           interfaccia a riga di comando (segment, bench)
src/mean_shift.{h,cpp} algoritmo e varianti simd / par / par_aos
src/mean_shift_seq.cpp baseline sequenziale scalare (compilata con -fno-tree-vectorize)
src/kernel.h           kernel gaussiano, fast_exp e passo di shift comuni
src/benchmark.{h,cpp}  misure, statistiche, esperimenti, CSV
src/image_io.{h,cpp}   lettura/scrittura immagini
third_party/           stb_image, stb_image_write (header-only)
scripts/               preprocessing del dataset e grafici
images/                immagini BSDS preprocessate
results/, figures/     risultati dei benchmark e grafici
report/                relazione (LaTeX, template CVPR): latexmk -pdf report.tex
```
