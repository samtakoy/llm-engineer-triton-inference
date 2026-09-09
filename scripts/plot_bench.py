"""Графики к отчёту: строятся из report/summary.csv."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
REPORT = REPO / "report"

BLUE, ORANGE, AQUA, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e34948"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": MUTED,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.size": 10,
})

table = pd.read_csv(REPORT / "summary.csv")


def dress(axis, title, ylabel):
    axis.set_yscale("log")
    axis.set_title(title, fontsize = 11, color = INK, loc = "left")
    axis.set_xlabel("одновременных запросов")
    axis.set_ylabel(ylabel)
    axis.grid(True, which = "major", color = GRID, linewidth = 0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def line(axis, scenario, column, color, label, dashed = False):
    group = table[table["scenario"] == scenario].sort_values("Concurrency")
    axis.plot(
        group["Concurrency"], group[column],
        marker = "o", markersize = 6, linewidth = 2,
        color = color, label = label,
        linestyle = "--" if dashed else "-",
    )


# 1. Общая картина: цепочка и модели по отдельности
SERIES = [
    ("bls_questions", RED,    "bls_questions (вся цепочка)"),
    ("bls_rude",      ORANGE, "bls_rude (цепочка без генерации)"),
    ("gen",           VIOLET, "gen (генератор)"),
    ("e5",            AQUA,   "e5 (эмбеддер)"),
    ("tox",           BLUE,   "tox (классификатор)"),
]

figure, axes = plt.subplots(1, 2, figsize = (12, 4.6))
for scenario, color, label in SERIES:
    line(axes[0], scenario, "RPS", color, label)
    line(axes[1], scenario, "avg latency", color, label)
dress(axes[0], "Пропускная способность", "запросов в секунду")
dress(axes[1], "Средняя задержка", "миллисекунд")
axes[0].legend(fontsize = 8.5, frameon = False, loc = "center left")
figure.tight_layout()
figure.savefig(REPORT / "bench.png", dpi = 150)

# 2. Батчинг: две модели, с ним и без
figure, axes = plt.subplots(1, 2, figsize = (12, 4.6))
line(axes[0], "tox", "RPS", BLUE, "с батчингом")
line(axes[0], "tox_nobatch", "RPS", BLUE, "без батчинга", dashed = True)
line(axes[1], "e5", "RPS", AQUA, "с батчингом")
line(axes[1], "e5_nobatch", "RPS", AQUA, "без батчинга", dashed = True)
for axis, title in [(axes[0], "toxicity_clf"), (axes[1], "e5_embedder")]:
    dress(axis, title, "запросов в секунду")
    axis.set_yscale("linear")
    axis.set_ylim(0, 100)
    axis.legend(fontsize = 9, frameon = False, loc = "lower right", handlelength = 3.5)
figure.tight_layout()
figure.savefig(REPORT / "batching.png", dpi = 150)

# 3. Копии эмбеддера при concurrency 16
CONFIGS = [
    ("e5_count2",    "2 копии\n× 2 потока"),
    ("e5_count4x2",  "4 копии\n× 2 потока"),
    ("e5_count1",    "1 копия\nбез ограничения"),
    ("e5_count4x4",  "4 копии\n× 4 потока"),
]
names = [name for _, name in CONFIGS]
rows = [table[table["scenario"] == scenario].iloc[0] for scenario, _ in CONFIGS]

figure, axes = plt.subplots(1, 2, figsize = (12, 4.6))
for axis, column, title, ylabel, color in [
    (axes[0], "RPS", "Пропускная способность", "запросов в секунду", AQUA),
    (axes[1], "Queue", "Ожидание в очереди", "миллисекунд", ORANGE),
]:
    values = [row[column] for row in rows]
    bars = axis.bar(names, values, color = color, width = 0.6)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2, value,
            f"{value:.1f}", ha = "center", va = "bottom", fontsize = 9.5, color = INK,
        )
    axis.set_title(title, fontsize = 11, color = INK, loc = "left")
    axis.set_ylabel(ylabel)
    axis.set_ylim(0, max(values) * 1.18)
    axis.grid(True, axis = "y", color = GRID, linewidth = 0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
figure.suptitle(
    "e5_embedder, 16 одновременных запросов", fontsize = 11, color = MUTED, x = 0.01, ha = "left",
)
figure.tight_layout()
figure.savefig(REPORT / "e5_instances.png", dpi = 150)
print("готово")
