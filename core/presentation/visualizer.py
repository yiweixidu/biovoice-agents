"""
core/presentation/visualizer.py
Matplotlib chart generators that return in-memory PNG bytes for embedding
in python-pptx slides.

All public functions return io.BytesIO, passable directly to
slide.shapes.add_picture(img_bytes, ...).

Design palette matches the PPT generator:
  Navy #1A3557 | Teal #2E86AB | Light gray #F5F6FA
"""

from __future__ import annotations

import io
from typing import Dict, List, Optional

NAVY  = "#1A3557"
TEAL  = "#2E86AB"
LGRAY = "#F5F6FA"
MGRAY = "#D0D3DA"

_PALETTE = [
    "#2E86AB",  # teal
    "#1A3557",  # navy
    "#E84855",  # red (H5, pandemic risk)
    "#3BB273",  # green
    "#F4A259",  # amber
    "#9B5DE5",  # purple
    "#00BBF9",  # sky blue
]


def _mpl():
    """Lazy import matplotlib to avoid failure at module import time."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        return plt, ticker
    except ImportError:
        raise ImportError(
            "matplotlib is required for chart generation. "
            "Install: pip install matplotlib"
        )


def _fig_to_bytes(fig) -> io.BytesIO:
    plt, _ = _mpl()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=LGRAY)
    buf.seek(0)
    plt.close(fig)
    return buf


# ── Chart 1: Influenza strain surveillance bar (FluNet data) ─────────────────

def create_strain_bar_chart(
    subtypes: Dict[str, int],
    title: str = "Influenza Strain Distribution",
    subtitle: str = "WHO FluNet Global Surveillance",
) -> io.BytesIO:
    """
    Horizontal bar chart of influenza subtype case counts.
    H5 strains are highlighted red (pandemic risk).

    Parameters
    ----------
    subtypes : { "H1N1pdm09": 12450, "H3N2": 8300, "H5": 23, "B/Victoria": 4100 }
    """
    plt, ticker = _mpl()

    if not subtypes:
        subtypes = {"No data available": 0}

    labels = list(subtypes.keys())
    values = list(subtypes.values())
    max_val = max(values) if values else 1

    colors = ["#E84855" if ("H5" in lbl or "H5N1" in lbl) else TEAL
              for lbl in labels]

    fig, ax = plt.subplots(figsize=(10, max(3, len(labels) * 0.65)))
    fig.patch.set_facecolor(LGRAY)
    ax.set_facecolor(LGRAY)

    bars = ax.barh(labels, values, color=colors, height=0.55, zorder=3)

    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(
                bar.get_width() + max_val * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}",
                va="center", ha="left",
                fontsize=10, color=NAVY, fontweight="bold",
            )

    ax.set_xlabel("Confirmed Cases", fontsize=11, color=NAVY)
    ax.set_title(f"{title}\n{subtitle}", fontsize=13, color=NAVY,
                 fontweight="bold", loc="left", pad=12)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.tick_params(axis="both", length=0, labelsize=10, labelcolor=NAVY)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    ax.grid(axis="x", color=MGRAY, linewidth=0.8, zorder=0)
    ax.invert_yaxis()

    plt.tight_layout()
    return _fig_to_bytes(fig)


# ── Chart 2: Neutralization heatmap (antibody × strain matrix) ───────────────

def create_neutralization_heatmap(
    data: Dict[str, List[float]],
    title: str = "Neutralization Breadth",
    strain_labels: Optional[List[str]] = None,
) -> io.BytesIO:
    """
    Heatmap of IC50 values (log₁₀ scale).
    Rows = antibodies, columns = virus strains.

    Parameters
    ----------
    data : { antibody_name: [ic50_strain1, ic50_strain2, ...] }
           Use None / NaN for untested combinations.
    strain_labels : column labels (virus strain names).
    """
    plt, _ = _mpl()
    import numpy as np  # noqa: PLC0415

    try:
        import seaborn as sns  # noqa: PLC0415
    except ImportError:
        raise ImportError("seaborn required for heatmaps: pip install seaborn")

    antibodies = list(data.keys())
    n_strains  = max((len(v) for v in data.values()), default=0)
    if not antibodies or n_strains == 0:
        return _empty_chart(plt, "No neutralization data available")

    matrix = []
    for ab in antibodies:
        row = [(float(v) if v is not None else float("nan")) for v in data[ab]]
        row += [float("nan")] * (n_strains - len(row))
        matrix.append(row)

    arr = np.array(matrix, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        arr_log = np.where(arr > 0, np.log10(arr), np.nan)

    fig, ax = plt.subplots(
        figsize=(min(14, max(7, n_strains * 1.2)),
                 min(10, max(4, len(antibodies) * 0.55)))
    )
    fig.patch.set_facecolor(LGRAY)

    sns.heatmap(
        arr_log, ax=ax,
        cmap="RdYlGn_r",
        annot=True, fmt=".1f",
        linewidths=0.5, linecolor=LGRAY,
        xticklabels=strain_labels or [f"Strain {i+1}" for i in range(n_strains)],
        yticklabels=antibodies,
        cbar_kws={"label": "log₁₀ IC50 (µg/mL)"},
    )
    ax.set_title(title, fontsize=13, fontweight="bold", color=NAVY,
                 loc="left", pad=12)
    ax.tick_params(axis="both", length=0, labelsize=9)
    plt.tight_layout()
    return _fig_to_bytes(fig)


# ── Chart 3: Publication trend (year → count) ─────────────────────────────────

def create_publication_trend(
    year_counts: Dict[int, int],
    title: str = "Publications per Year",
    highlight_years: Optional[List[int]] = None,
) -> io.BytesIO:
    """
    Line + area chart. Vertical dashed lines mark key events
    (e.g. pandemic years 2009, 2020).
    """
    plt, _ = _mpl()

    if not year_counts:
        return _empty_chart(plt, "No publication data")

    years  = sorted(year_counts.keys())
    counts = [year_counts[y] for y in years]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(LGRAY)
    ax.set_facecolor(LGRAY)

    ax.fill_between(years, counts, alpha=0.18, color=TEAL)
    ax.plot(years, counts, color=TEAL, linewidth=2.5, zorder=3)
    ax.scatter(years, counts, color=NAVY, s=50, zorder=4)

    if highlight_years:
        for hy in highlight_years:
            if min(years) <= hy <= max(years):
                ax.axvline(hy, color="#E84855", linewidth=1.2,
                           linestyle="--", alpha=0.7, zorder=2)
                ax.text(hy + 0.1, max(counts) * 0.92, str(hy),
                        color="#E84855", fontsize=9)

    ax.set_xlabel("Year", fontsize=11, color=NAVY)
    ax.set_ylabel("Publications", fontsize=11, color=NAVY)
    ax.set_title(title, fontsize=13, fontweight="bold", color=NAVY,
                 loc="left", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MGRAY)
    ax.tick_params(axis="both", length=0, labelsize=10, labelcolor=NAVY)
    ax.grid(axis="y", color=MGRAY, linewidth=0.8, zorder=0)
    ax.set_xticks(years[::max(1, len(years) // 10)])

    plt.tight_layout()
    return _fig_to_bytes(fig)


# ── Chart 4: Coverage donut ───────────────────────────────────────────────────

def create_coverage_donut(
    segments: Dict[str, float],
    title: str = "Antibody Coverage",
    center_text: str = "",
) -> io.BytesIO:
    """
    Donut chart. segments = { label: percentage }.
    e.g. {"Group 1 A": 72, "Group 2 A": 18, "B lineages": 10}
    """
    plt, _ = _mpl()

    if not segments:
        return _empty_chart(plt, "No coverage data")

    labels = list(segments.keys())
    sizes  = list(segments.values())
    colors = _PALETTE[:len(labels)]

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(LGRAY)

    wedges, _, autotexts = ax.pie(
        sizes, labels=None, colors=colors,
        autopct="%1.0f%%", startangle=90, pctdistance=0.78,
        wedgeprops={"width": 0.52, "edgecolor": LGRAY, "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_color("white")
        at.set_fontweight("bold")

    if center_text:
        ax.text(0, 0, center_text, ha="center", va="center",
                fontsize=13, fontweight="bold", color=NAVY)

    ax.legend(wedges, labels, loc="lower center",
              bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9,
              framealpha=0, labelcolor=NAVY)
    ax.set_title(title, fontsize=13, fontweight="bold", color=NAVY, pad=10)

    plt.tight_layout()
    return _fig_to_bytes(fig)


# ── Utility ───────────────────────────────────────────────────────────────────

def _empty_chart(plt, message: str) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(LGRAY)
    ax.text(0.5, 0.5, message, ha="center", va="center",
            fontsize=14, color=NAVY)
    ax.set_axis_off()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=LGRAY)
    buf.seek(0)
    plt.close(fig)
    return buf