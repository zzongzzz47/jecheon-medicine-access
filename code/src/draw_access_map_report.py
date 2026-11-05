"""A report-ready city overview and urban close-up using saved walking distances."""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from walking_network import TO_XY

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/residents_connector80"
OUTPUT = ROOT / "results/resident_analysis/figures_report"
INK = "#243b4c"
COLORS = np.array(["#248990", "#9bc5b8", "#e5a64a", "#b84e50"])
LABELS = ["500m 이하", "500m 초과–1km", "1km 초과–3km", "3km 초과"]
ZOOM = (1057500, 1900800, 1068200, 1910050)


def read(name):
    return json.loads((SOURCE / name).read_text())


def xy(rows):
    return np.array(TO_XY.transform([r["lon"] for r in rows], [r["lat"] for r in rows])).T


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": FontProperties(fname="/System/Library/Fonts/Supplemental/AppleGothic.ttf").get_name(),
        "axes.unicode_minus": False, "font.size": 11, "text.color": INK,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    result = read("optimization_medoid_screened.json")
    plan = next(p for p in result["results"] if p["max_sites"] == 10
                and p["candidate_group"] == "mixed" and p["mode"] == "efficiency")
    origins, existing, candidates = read("origins.json"), read("existing.json"), read("candidates.json")
    distance = np.load(SOURCE / "distances.npz")["walking"]
    lookup = {r["id"]: i for i, r in enumerate(candidates)}
    before = distance[:, :len(existing)].min(axis=1)
    selected_columns = [len(existing) + lookup[r["id"]] for r in plan["selected"]]
    after = np.minimum(before, distance[:, selected_columns].min(axis=1))
    population = np.array([(r.get("population", 0) or 0) if r["group"] == "grid100m" else 0 for r in origins])
    valid = (population > 0) & np.isfinite(before)
    assert population[valid].sum() == 132709
    for values, expected in [(before, result["baseline"]["residents"]["mean_m"]),
                             (after, plan["residents"]["mean_m"])]:
        assert np.isclose(np.average(values[valid], weights=population[valid]), expected)
    points, facilities, additions = xy(origins), xy(existing), xy(plan["selected"])
    admin = gpd.read_file(ROOT / "data/processed/population/population.gpkg", layer="admin").to_crs(5179)
    x0, y0, x1, y1 = admin.total_bounds
    city_bounds = (x0 - 1500, y0 - 1000, x1 + 1500, y1 + 1000)
    fig = plt.figure(figsize=(11.5, 12.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], left=.035, right=.985,
                           top=.92, bottom=.105, hspace=.14, wspace=.06)
    axes = np.array([[fig.add_subplot(grid[i, j]) for j in range(2)] for i in range(2)])

    def base(ax, bounds, closeup):
        admin.plot(ax=ax, facecolor="#f6f8fa", edgecolor="#748994", linewidth=.85 if closeup else .7, zorder=0)
        ax.set_xlim(bounds[0], bounds[2]); ax.set_ylim(bounds[1], bounds[3])
        ax.set_aspect("equal"); ax.axis("off")
        ax.text(.98, .97, "N ↑", transform=ax.transAxes, ha="right", va="top", fontsize=10)
        length = 2000 if closeup else 5000
        sx = bounds[0] + .045 * (bounds[2] - bounds[0])
        sy = bounds[1] + .045 * (bounds[3] - bounds[1])
        ax.plot([sx, sx + length], [sy, sy], color=INK, lw=2.5, zorder=10)
        ax.text(sx + length / 2, sy + .012 * (bounds[3] - bounds[1]),
                f"{length // 1000} km", ha="center", fontsize=9, zorder=10)

    def label(ax, x, y, text, size=11):
        ax.text(x, y, text, ha="center", va="center", fontsize=size, weight="bold",
                color=INK, zorder=12, path_effects=[pe.withStroke(linewidth=3.5, foreground="white")])

    # Fixed display classes, identical on every panel. Sizes indicate points,
    # not population, service catchments, or areas covered by a facility.
    for col, values in enumerate([before, after]):
        cls = np.digitize(values[valid], [500, 1000, 3000], right=True)
        for row in range(2):
            ax = axes[row, col]; closeup = row == 1
            bounds = ZOOM if closeup else city_bounds
            base(ax, bounds, closeup)
            ax.scatter(points[valid, 0], points[valid, 1], c=COLORS[cls],
                       s=11 if closeup else 5.5, marker="o", edgecolors="none", alpha=.95, zorder=2)
            ax.scatter(facilities[:, 0], facilities[:, 1], s=25 if closeup else 14,
                       marker="s", color=INK, edgecolor="white", linewidth=.45, zorder=5)
            if col == 1:
                ax.scatter(additions[:, 0], additions[:, 1], s=190 if closeup else 120,
                           marker="*", color="#1260ad", edgecolor="white", linewidth=.85, zorder=8)
            if not closeup:
                for _, r in admin.iterrows():
                    if r.ADM_NM.endswith(("읍", "면")):
                        rp = r.geometry.representative_point()
                        label(ax, rp.x, rp.y, r.ADM_NM, 10)
                ax.add_patch(Rectangle((ZOOM[0], ZOOM[1]), ZOOM[2] - ZOOM[0], ZOOM[3] - ZOOM[1],
                                      fill=False, ec=INK, lw=1.4, ls=(0, (4, 2)), zorder=11))
                ax.annotate("아래 지도에서 확대", xy=((ZOOM[0] + ZOOM[2]) / 2, ZOOM[1]),
                            xytext=(ZOOM[0] - 13000, ZOOM[1] - 4300), fontsize=10,
                            arrowprops={"arrowstyle": "-", "color": INK, "lw": .9},
                            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2}, zorder=15)
            else:
                positions = {
                    "용두동": (1059750, 1907650), "의림지동": (1064900, 1908850),
                    "신백동": (1066900, 1902200), "교동": (1065000, 1905700),
                    "청전동": (1062350, 1906200), "중앙동": (1062500, 1904650),
                    "남현동": (1064100, 1903550), "영서동": (1059800, 1902550),
                    "화산동": (1062800, 1901650),
                }
                for name, (x, y) in positions.items():
                    label(ax, x, y, name, 10.5)
                ax.set_title("도심 확대", fontsize=12, pad=8, color=INK)

    # Panel identifiers only; the overall figure caption belongs in the report.
    fig.text(.275, .972, "기존 수거망", ha="center", va="top", fontsize=17, weight="bold")
    fig.text(.755, .972, "10곳 추가 후", ha="center", va="top", fontsize=17, weight="bold")
    fig.text(.275, .943, "전체 평균거리 1,438m", ha="center", va="top", fontsize=12)
    fig.text(.755, .943, "편의점 5곳 + 경로당 5곳 · 전체 평균거리 945m", ha="center", va="top", fontsize=11)
    color_handles = [Line2D([], [], marker="o", linestyle="none", markerfacecolor=c,
                            markeredgecolor="none", markersize=8, label=t) for c, t in zip(COLORS, LABELS)]
    fig.legend(handles=color_handles, loc="lower center", bbox_to_anchor=(.5, .052),
               ncol=4, frameon=False, columnspacing=2.1, title="가장 가까운 수거지까지의 보행경로 거리", fontsize=11, title_fontsize=11)
    site_handles = [Line2D([], [], marker="s", linestyle="none", color=INK, markersize=6, label="기존 수거지"),
                    Line2D([], [], marker="*", linestyle="none", color="#1260ad", markersize=12, label="추가 거점"),
                    Line2D([], [], linestyle="-", color="#748994", linewidth=1, label="읍면동 경계")]
    fig.legend(handles=site_handles, loc="lower center", bbox_to_anchor=(.5, .025), ncol=3, frameon=False, fontsize=10)
    target = OUTPUT / "그림2-1_접근거리_경계와확대지도.png"
    fig.savefig(target, dpi=300, facecolor="white")
    plt.close(fig)
    print(target)


if __name__ == "__main__":
    main()
