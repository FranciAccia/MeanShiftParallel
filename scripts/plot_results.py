"""Genera i grafici della relazione a partire dai CSV del benchmark (results/*.csv).

Uso:  .venv/bin/python scripts/plot_results.py
Output: figures/*.png
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Palette categorica (ordine fisso) e inchiostri per testo/assi
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.edgecolor": TEXT_SECONDARY,
    "axes.labelcolor": TEXT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "legend.frameon": False,
})


def load(name):
    with open(RESULTS / f"{name}.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key in ("size", "n_pixels", "threads", "reps", "clusters"):
            r[key] = int(r[key])
        for key in ("bandwidth", "mean_s", "std_s", "min_s", "max_s", "ci95_s", "cpu_s", "modes_s", "avg_iter"):
            r[key] = float(r[key])
    return rows


def save(fig, name):
    fig.savefig(FIGURES / name)
    plt.close(fig)
    print(f"figures/{name}")


def plot_scaling():
    """Strong scaling. Speedup dei soli thread (rispetto a par con 1 thread) con retta ideale ed
    efficienza; speedup totale rispetto alla baseline seq (thread + SIMD)."""
    rows = load("scaling")
    by_size = defaultdict(list)
    for r in rows:
        by_size[r["size"]].append(r)
    sizes = sorted(by_size)
    all_threads = sorted({r["threads"] for r in rows if r["variant"] == "par"})
    max_threads = all_threads[-1]

    fig, (ax_s, ax_e, ax_t) = plt.subplots(1, 3, figsize=(14, 4))
    for color, size in zip(SERIES, sizes):
        seq = next(r for r in by_size[size] if r["variant"] == "seq")
        par = sorted((r for r in by_size[size] if r["variant"] == "par"), key=lambda r: r["threads"])
        threads = [r["threads"] for r in par]
        thread_speedup = [par[0]["mean_s"] / r["mean_s"] for r in par]
        label = f"{size}×{size} ({size * size} px)"
        ax_s.plot(threads, thread_speedup, "o-", color=color, label=label)
        ax_e.plot(threads, [s / t for s, t in zip(thread_speedup, threads)], "o-", color=color, label=label)
        ax_t.plot(threads, [seq["mean_s"] / r["mean_s"] for r in par], "o-", color=color, label=label)

    ax_s.plot([1, max_threads], [1, max_threads], color=GRID, linewidth=1.5, linestyle="--", zorder=0)
    ax_s.text(9.5, 10.5, "ideal", color=TEXT_SECONDARY, fontsize=8, rotation=35)
    ax_s.set(xlabel="Threads", ylabel="Speedup over par with 1 thread", title="Thread speedup",
             ylim=(0, 12))
    ax_e.axhline(1, color=TEXT_SECONDARY, linewidth=1, linestyle=":")
    ax_e.set(xlabel="Threads", ylabel="Efficiency = speedup / threads", title="Efficiency", ylim=(0, 1.1))
    ax_t.set(xlabel="Threads", ylabel="Speedup over seq", title="Total speedup (threads + SIMD)",
             ylim=(0, None))
    for ax in (ax_s, ax_e, ax_t):
        ax.set_xticks(all_threads)
        ax.axvline(8, color=TEXT_SECONDARY, linewidth=1, linestyle=":")
    ax_s.text(8.3, 0.4, "8 cores\n(4P + 4E)", color=TEXT_SECONDARY, fontsize=8)
    ax_s.legend(loc="upper left")
    save(fig, "scaling.png")

    # Tempo wall-clock e tempo CPU (somma dei thread) per la dimensione più grande: overhead
    size = sizes[-1]
    par = sorted((r for r in by_size[size] if r["variant"] == "par"), key=lambda r: r["threads"])
    threads = [r["threads"] for r in par]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(threads, [r["mean_s"] for r in par], yerr=[r["ci95_s"] for r in par], fmt="o-",
                color=SERIES[0], capsize=3, label="Wall-clock")
    ax.plot(threads, [r["cpu_s"] for r in par], "s--", color=SERIES[1], label="CPU (sum over threads)")
    ax.set(xlabel="Threads", ylabel="Time (s)", title=f"Wall-clock and CPU time, {size}×{size}", ylim=(0, None))
    ax.set_xticks(threads)
    ax.legend()
    save(fig, "cpu_time.png")


def plot_vectorization():
    """Seq (scalare) vs simd (1 thread, vettorizzato) per ogni dimensione."""
    rows = load("scaling")
    sizes = sorted({r["size"] for r in rows})
    seq = [next(r for r in rows if r["size"] == s and r["variant"] == "seq") for s in sizes]
    simd = [next(r for r in rows if r["size"] == s and r["variant"] == "simd") for s in sizes]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(sizes))
    w = 0.38
    ax.bar([i - w / 2 - 0.01 for i in x], [r["mean_s"] for r in seq], w, yerr=[r["ci95_s"] for r in seq],
           color=SERIES[0], capsize=3, label="seq (scalar)")
    ax.bar([i + w / 2 + 0.01 for i in x], [r["mean_s"] for r in simd], w, yerr=[r["ci95_s"] for r in simd],
           color=SERIES[1], capsize=3, label="simd (NEON, 4 float)")
    for i, (a, b) in enumerate(zip(seq, simd)):
        ax.text(i + w / 2, b["mean_s"], f"{a['mean_s'] / b['mean_s']:.2f}×", ha="center", va="bottom",
                color=TEXT_SECONDARY, fontsize=9)
    ax.set_xticks(list(x), [f"{s}×{s}" for s in sizes])
    ax.set(xlabel="Image size", ylabel="Time (s), 1 thread", title="Effect of vectorization")
    ax.grid(axis="x", visible=False)
    ax.legend()
    save(fig, "vectorization.png")


def plot_bandwidth():
    """Cluster trovati, iterazioni medie e tempi al variare del bandwidth (grafici separati: un asse ciascuno)."""
    rows = load("bandwidth")
    seq = sorted((r for r in rows if r["variant"] == "seq"), key=lambda r: r["bandwidth"])
    par = sorted((r for r in rows if r["variant"] == "par"), key=lambda r: r["bandwidth"])
    h = [r["bandwidth"] for r in seq]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(h, [r["clusters"] for r in seq], "o-", color=SERIES[0])
    for x, r in zip(h, seq):
        axes[0].annotate(str(r["clusters"]), (x, r["clusters"]), textcoords="offset points", xytext=(0, 6),
                         ha="center", color=TEXT_SECONDARY, fontsize=9)
    axes[0].set(xlabel="Bandwidth h", ylabel="Number of clusters K", title="Clusters found", ylim=(0, None))

    axes[1].plot(h, [r["avg_iter"] for r in seq], "o-", color=SERIES[0])
    axes[1].set(xlabel="Bandwidth h", ylabel="Mean iterations per point", title="Iterations to convergence",
                ylim=(0, None))

    axes[2].errorbar(h, [r["mean_s"] for r in seq], yerr=[r["ci95_s"] for r in seq], fmt="o-",
                     color=SERIES[0], capsize=3, label="seq")
    axes[2].errorbar(h, [r["mean_s"] for r in par], yerr=[r["ci95_s"] for r in par], fmt="o-",
                     color=SERIES[1], capsize=3, label=f"par ({par[0]['threads']} threads)")
    for x, a, b in zip(h, seq, par):
        axes[2].annotate(f"{a['mean_s'] / b['mean_s']:.1f}×", (x, b["mean_s"]), textcoords="offset points",
                         xytext=(0, 8), ha="center", color=TEXT_SECONDARY, fontsize=8)
    axes[2].set(xlabel="Bandwidth h", ylabel="Time (s)", title="Time and speedup", ylim=(0, None))
    axes[2].legend()
    fig.suptitle(f"Effect of the bandwidth ({seq[0]['size']}×{seq[0]['size']})", color=TEXT, y=1.04)
    save(fig, "bandwidth.png")


def plot_schedule():
    rows = load("schedule")
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [r["schedule"] for r in rows]
    ax.bar(labels, [r["mean_s"] for r in rows], 0.6, yerr=[r["ci95_s"] for r in rows], color=SERIES[0], capsize=3)
    best = min(rows, key=lambda r: r["mean_s"])
    for i, r in enumerate(rows):
        ax.text(i, r["mean_s"] + r["ci95_s"], f"{r['mean_s']:.2f}", ha="center", va="bottom",
                color=TEXT if r is best else TEXT_SECONDARY, fontsize=9,
                fontweight="bold" if r is best else "normal")
    ax.set(xlabel="schedule(kind, chunk)", ylabel="Time (s)",
           title=f"OpenMP scheduling ({rows[0]['size']}×{rows[0]['size']}, {rows[0]['threads']} threads)")
    ax.grid(axis="x", visible=False)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    save(fig, "schedule.png")


def plot_layout():
    rows = load("layout")
    threads = sorted({r["threads"] for r in rows})
    fig, ax = plt.subplots(figsize=(6, 4))
    w = 0.38
    for k, (variant, label) in enumerate([("par", "SoA"), ("par_aos", "AoS")]):
        sel = [next(r for r in rows if r["variant"] == variant and r["threads"] == t) for t in threads]
        xs = [i + (k - 0.5) * (w + 0.02) for i in range(len(threads))]
        ax.bar(xs, [r["mean_s"] for r in sel], w, yerr=[r["ci95_s"] for r in sel], color=SERIES[k],
               capsize=3, label=label)
        for x, r in zip(xs, sel):
            ax.text(x, r["mean_s"] + r["ci95_s"], f"{r['mean_s']:.2f}", ha="center", va="bottom",
                    color=TEXT_SECONDARY, fontsize=9)
    ax.set_xticks(range(len(threads)), [f"{t} thread{'s' if t > 1 else ''}" for t in threads])
    ax.set(ylabel="Time (s)", title=f"Memory layout ({rows[0]['size']}×{rows[0]['size']})")
    ax.grid(axis="x", visible=False)
    ax.legend()
    save(fig, "layout.png")


def plot_segmentations():
    """Griglia: immagine originale + segmentazioni per alcuni bandwidth, per ogni immagine."""
    ids = ["118015", "108004", "145059"]
    bandwidths = [6, 12, 16, 24]
    fig, axes = plt.subplots(len(ids), len(bandwidths) + 1, figsize=(2.2 * (len(bandwidths) + 1), 2.3 * len(ids)))
    for row, image_id in enumerate(ids):
        panels = [(ROOT / "images" / f"{image_id}_128.png", "Original")]
        panels += [(RESULTS / f"seg_{image_id}_h{h:02d}.png", f"h = {h}") for h in bandwidths]
        for col, (path, title) in enumerate(panels):
            ax = axes[row][col]
            ax.imshow(Image.open(path), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(title, fontsize=10)
    save(fig, "segmentations.png")


def main():
    FIGURES.mkdir(exist_ok=True)
    plot_scaling()
    plot_vectorization()
    plot_bandwidth()
    plot_schedule()
    plot_layout()
    plot_segmentations()


if __name__ == "__main__":
    main()
