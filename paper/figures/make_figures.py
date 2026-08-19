#!/usr/bin/env python
"""Bikin semua figur Paper 1 dari CSV hasil notebook (yang sudah di-download).

Jalankan dari root repo:  ./venv/Scripts/python.exe paper/figures/make_figures.py

Sumber angka (jangan hardcode angka di sini kecuali ambang permutasi, yang
memang dihitung terpisah di analisis lokal finding 09):
  fig_headmap      <- notebooks/output/09_.../peta_kesetiaan_full.csv
  fig_layercurve   <- notebooks/output/09_.../peta_kesetiaan_full.csv
  fig_causal_sweep <- notebooks/output/13_.../sweep_rows.csv (t dihitung ulang)

Palet: Okabe-Ito subset, lolos validator CVD (deutan dE 11.0, normal 25.8).
Diverging heatmap: dua kutub + titik tengah netral abu (bukan pelangi).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MAP_CSV = ROOT / "notebooks/output/09_tahap2_peta_kesetiaan_kaggle/peta_kesetiaan_full.csv"
SWEEP_CSV = ROOT / "notebooks/output/13_tahap2_sweep_dan_probe_kaggle/sweep_rows.csv"
NB14 = ROOT / "notebooks/output/14_tahap2_probefix_sweep6_kaggle"
SWEEP6_CSV = NB14 / "sweep6_rows.csv"
MAXSTAT_CSV = NB14 / "sweep_all6_maxstat.csv"      # ambang noise 6 tipe (1 run, 1 seed)
FIXED_LAYER_CSV = NB14 / "fixed_layer_L11_L1.csv"  # uji lokasi-tetap L11 & L1 (level item)
PAIRLEVEL_CSV = NB14 / "causal_pairlevel_sweep.csv"  # inferensi level-PASANGAN (utama)
SPLITHALF_CSV = ROOT / ("notebooks/output/09_tahap2_peta_kesetiaan_kaggle/"
                        "koreksi_seleksi/cek3_splithalf.csv")
PROBE_CSV = NB14 / "probe_v2_summary.csv"
GROUPRANK_CSV = NB14 / "probe_v2_grouprank.csv"     # uji belah-dua (bebas artefak LOO)
CEILING_CSV = NB14 / "grouprank_noise_ceiling.csv"  # langit-langit reliabilitas metrik
HELDOUT_CSV = ROOT / ("notebooks/output/09_tahap2_peta_kesetiaan_kaggle/"
                      "koreksi_seleksi/cek1_heldout_template.csv")

# --- parameter desain -------------------------------------------------------
CAT = {"resid": "#0072B2", "head": "#D55E00", "mlp": "#009E73"}
INK, INK_MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d8d8d5"
DIVERGING = LinearSegmentedColormap.from_list(
    "cool_neutral_warm", ["#0072B2", "#7fb8d8", "#eeeeec", "#e8a06a", "#D55E00"]
)
STAR_LAYER, STAR_HEAD = 11, 16

TYPES = ["AGExPOLPARTY", "EDUCATIONxINCOME", "RELIGxPOLPARTY",
         "RACExPOLPARTY", "RACExPOLIDEOLOGY", "RACExRELIG"]
# ambang noise (persentil-95 max-t, sign-flip 2000x) dibaca dari CSV, bukan
# di-hardcode -- semua 6 tipe dihitung dalam satu run/seed yang sama.

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5,
    "axes.edgecolor": GRID, "axes.linewidth": 0.6,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "text.color": INK, "axes.labelcolor": INK,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def pretty(t):
    return t.replace("x", r" $\times$ ")


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / (stem + "." + ext))
    plt.close(fig)
    print("  -> " + stem + ".pdf / .png")


# --- Fig 1: peta head 32x32 per tipe ---------------------------------------
def fig_headmap(df):
    h = df[(df.component == "head") & (df.template == "Tmean")]
    grids = {t: h[h.attr_type == t].pivot(index="layer", columns="head", values="rho").to_numpy()
             for t in TYPES}
    vmax = max(np.nanmax(np.abs(g)) for g in grids.values())
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.6), constrained_layout=True)
    im = None
    for ax, t in zip(axes.ravel(), TYPES):
        g = grids[t]
        im = ax.imshow(g, cmap=DIVERGING, norm=norm, aspect="auto", origin="lower")
        li, hi = np.unravel_index(np.nanargmax(g), g.shape)
        ax.add_patch(plt.Rectangle((STAR_HEAD - .5, STAR_LAYER - .5), 1, 1,
                                   fill=False, ec=INK, lw=1.1))
        ax.plot(hi, li, marker="o", ms=3.2, mfc="none", mec=INK, mew=0.9)
        ax.set_title(pretty(t) + "\nmax $\\rho$=" + ("%.2f" % np.nanmax(g))
                     + " at L%d H%d" % (li, hi), pad=3)
        ax.set_xticks([0, 8, 16, 24, 31])
        ax.set_yticks([0, 8, 16, 24, 31])
        ax.tick_params(length=2)
    for ax in axes[-1]:
        ax.set_xlabel("head")
    for ax in axes[:, 0]:
        ax.set_ylabel("layer")
    cb = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.015)
    cb.set_label(r"fidelity $\rho$ (RSA vs. survey RDM)")
    cb.outline.set_edgecolor(GRID)
    fig.suptitle("Fidelity of every attention head (Mistral-7B, multi-cue read-out)",
                 fontsize=9, y=1.06)
    fig.text(0.5, 1.015,
             "square = L11 H16 (fixed cross-type location); circle = per-type maximum",
             ha="center", fontsize=7, color=INK_MUTED)
    save(fig, "fig_headmap")


# --- Fig 2: kurva per-layer, residual vs head terbaik vs FFN ---------------
def fig_layercurve(df):
    d = df[df.template == "Tmean"]
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 3.9), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, t in zip(axes.ravel(), TYPES):
        s = d[d.attr_type == t]
        resid = s[s.component == "resid"].sort_values("layer")
        mlp = s[s.component == "mlp"].sort_values("layer")
        best = s[s.component == "head"].groupby("layer")["rho"].max().sort_index()
        ax.axhline(0, color=GRID, lw=0.6, zorder=0)
        ax.plot(best.index, best.to_numpy(), color=CAT["head"], lw=1.6,
                label="best head in layer")
        ax.plot(resid.layer, resid.rho, color=CAT["resid"], lw=1.6,
                label="residual stream")
        ax.plot(mlp.layer, mlp.rho, color=CAT["mlp"], lw=1.2, ls=(0, (4, 2)),
                label="FFN output")
        ax.axvline(STAR_LAYER, color=INK_MUTED, lw=0.7, ls=(0, (1, 2)), zorder=0)
        ax.set_title(pretty(t), pad=3)
        ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(length=2)
    for ax in axes[-1]:
        ax.set_xlabel("layer")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"fidelity $\rho$")
    axes[0, 0].legend(frameon=False, loc="lower left", handlelength=1.6, borderpad=0.2)
    fig.suptitle("Attention heads dominate the residual stream at every depth",
                 fontsize=9, y=1.04)
    save(fig, "fig_layercurve")


# --- Fig 3: sweep kausal per-layer, 6 tipe, t vs ambang permutasi ---------
def _sweep_t():
    """t per layer per tipe, dari dua notebook sweep digabung."""
    d = pd.concat([pd.read_csv(SWEEP_CSV), pd.read_csv(SWEEP6_CSV)])
    d["diff"] = d.shift_ke_realB - d.shift_ctrl_ke_realB
    out = {}
    for ty, g in d.groupby("attr_type"):
        piv = g.pivot_table(index=["pair", "qkey"], columns="layer", values="diff").dropna()
        M = piv.to_numpy()
        out[ty] = (np.array(piv.columns),
                   M.mean(0) / (M.std(0, ddof=1) / np.sqrt(len(M))))
    return out


def fig_causal_sweep():
    tvals_by_type = _sweep_t()
    thr = pd.read_csv(MAXSTAT_CSV).set_index("attr_type")["null_p95"]
    pcor = pd.read_csv(MAXSTAT_CSV).set_index("attr_type")["p_corrected"]
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.0), sharey=True,
                             constrained_layout=True)
    for ax, ty in zip(axes.ravel(), TYPES):
        layers, tv = tvals_by_type[ty]
        th = thr[ty]
        win = int(layers[int(np.argmax(tv))])
        colors = [CAT["head"] if v >= th else "#c9c9c4" for v in tv]
        ax.axhline(0, color=GRID, lw=0.6)
        ax.bar(layers, tv, color=colors, width=0.8)
        ax.axhline(th, color=INK, lw=0.9, ls=(0, (3, 2)))
        sig = pcor[ty] < 0.05
        ax.annotate("L%d%s" % (win, "" if sig else " (n.s.)"),
                    xy=(win, float(np.max(tv))), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=7,
                    color=INK if sig else INK_MUTED,
                    bbox=dict(fc="white", ec="none", pad=0.6))
        ax.set_title(pretty(ty) + "   $p_{\\mathrm{item}}$=%.4f" % pcor[ty], pad=3)
        ax.set_xticks([0, 8, 16, 24, 31])
        ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(length=2)
    for ax in axes[-1]:
        ax.set_xlabel("layer patched (all 32 heads)")
    for ax in axes[:, 0]:
        ax.set_ylabel("$t$ vs. random-layer ctrl")
    axes[0, 2].annotate("selection-corrected\nnoise ceiling (p95)",
                        xy=(31, thr[TYPES[2]]), xytext=(0, -14),
                        textcoords="offset points", ha="right", fontsize=6.5,
                        color=INK_MUTED)
    fig.suptitle("Where identity causally enters differs by type "
                 "— and not in proportion to fidelity", fontsize=9, y=1.05)
    save(fig, "fig_causal_sweep")


# --- Fig 5: kesetiaan vs kekuatan kausal (disosiasi, level PASANGAN) ------
def fig_fidelity_vs_causal():
    # estimator kesetiaan KONSISTEN (review C2): median split-half (cek3)
    fid = pd.read_csv(SPLITHALF_CSV).set_index("attr_type")["heldout_median"]
    pl = pd.read_csv(PAIRLEVEL_CSV).set_index("tipe")
    from scipy.stats import spearmanr
    x = fid[TYPES].to_numpy()
    panels = [("$t$ at the a-priori locus L11", pl.loc[TYPES, "t_L11_pair"].to_numpy(),
               pl.loc[TYPES, "p_L11_pair"].to_numpy()),
              ("$t$ at each type's best layer", pl.loc[TYPES, "t_win_pair"].to_numpy(),
               pl.loc[TYPES, "p_maxstat_pair"].to_numpy())]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), constrained_layout=True)
    for ax, (lab, yy, pp) in zip(axes, panels):
        rr = spearmanr(x, yy).statistic
        sig = pp < 0.05
        ax.scatter(x[sig], yy[sig], s=40, color=CAT["head"], zorder=3,
                   label="$p<0.05$ (pair-level)")
        ax.scatter(x[~sig], yy[~sig], s=40, facecolors="none",
                   edgecolors=CAT["head"], linewidths=1.4, zorder=3,
                   label="n.s. (pair-level)")
        for xi, yi, ty in zip(x, yy, TYPES):
            ax.annotate(pretty(ty).replace(" $\times$ ", "×"), (xi, yi),
                        xytext=(4, 3), textcoords="offset points", fontsize=6,
                        color=INK_MUTED)
        ax.set_xlabel(r"fidelity $\rho$ (split-half median)")
        ax.set_ylabel(lab)
        ax.set_title(r"Spearman $\rho$ = %+.2f ($n$=6)" % rr, pad=4)
        ax.grid(color=GRID, lw=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(length=2)
    axes[0].legend(frameon=False, fontsize=6.2, loc="upper left")
    fig.suptitle("A faithful map does not predict causal use "
                 "(cluster-robust, $n$=12 pairs/type)", fontsize=9, y=1.06)
    save(fig, "fig_fidelity_vs_causal")


# --- Fig 4: lokasi TETAP L11 H16 vs baseline residual ---------------------
def fig_starhead(df):
    # dua seri dari SUMBER YANG SAMA (peta Tmean) biar apple-to-apple;
    # uji signifikansinya sendiri ada di cek4_fixed_heads.csv (subset sel
    # bersih-NaN, angkanya beda tipis di tipe berbau ras: .59/.33 vs .56/.31)
    d = df[df.template == "Tmean"]
    fixed = (d[(d.component == "head") & (d.layer == STAR_LAYER) & (d["head"] == STAR_HEAD)]
             .set_index("attr_type")["rho"])
    resid = d[d.component == "resid"].groupby("attr_type")["rho"].max()

    order = fixed.sort_values().index.tolist()
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(4.6, 2.7), constrained_layout=True)
    ax.barh(y, [fixed[t] for t in order], color=CAT["head"], height=0.55,
            label="L11 H16 (location fixed in advance)")
    ax.plot([resid[t] for t in order], y, "o", ms=5, mfc="none",
            mec=CAT["resid"], mew=1.4, ls="none", label="best residual-stream read-out")
    ax.set_yticks(y, [pretty(t) for t in order])
    ax.set_xlabel(r"fidelity $\rho$ vs. survey ground truth")
    ax.set_xlim(0, 0.75)
    ax.grid(axis="x", color=GRID, lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=2)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=2, handlelength=1.4, borderpad=0.2, fontsize=6.8)
    ax.set_title("One head is faithful in all six types — without being re-selected",
                 fontsize=9, pad=6)
    save(fig, "fig_starhead")


# --- Fig 6: probe (baca peta) vs mulut model -----------------------------
def fig_probe():
    s = pd.read_csv(PROBE_CSV).set_index("tipe")
    g = pd.read_csv(GROUPRANK_CSV).set_index("tipe")
    ceil = pd.read_csv(CEILING_CSV).set_index("tipe")["langit_reliabilitas"]
    order = TYPES
    x = np.arange(len(order))
    w = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), constrained_layout=True)

    ax = axes[0]
    series = [("model's own answer", "wd_mulut", "#8a8a85"),
              ("probe: L11 (4{,}096 d)", "wd_L11", CAT["resid"]),
              ("probe: L11 H16 (128 d)", "wd_L11H16", CAT["head"]),
              ("other groups' real answers", "wd_qmean", CAT["mlp"])]
    for k, (lab, col, c) in enumerate(series):
        ax.bar(x + (k - 1.5) * w, s.loc[order, col], width=w, color=c,
               label=lab.replace("{,}", ","))
    ax.set_ylabel("distance to survey truth (WD)\n$\\leftarrow$ lower is better")
    ax.set_xticks(x, [pretty(t).replace(" $\\times$ ", "×") for t in order],
                  rotation=30, ha="right", fontsize=6.5)
    ax.legend(frameon=False, fontsize=6.3, ncol=1, loc="upper left")
    ax.set_title("Absolute accuracy: the map beats the mouth", pad=4)
    ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=2)

    ax = axes[1]
    for k, (lab, col, c) in enumerate([("model's own answer", "gr_mulut", "#8a8a85"),
                                       ("probe: L11", "gr_L11", CAT["resid"]),
                                       ("probe: L11 H16", "gr_L11H16", CAT["head"])]):
        ax.bar(x + (k - 1) * w, g.loc[order, col], width=w, color=c, label=lab,
               yerr=g.loc[order, col + "_se"], error_kw=dict(lw=0.7, ecolor=INK_MUTED))
    ax.axhline(float(ceil.mean()), color=INK, lw=0.9, ls=(0, (3, 2)))
    ax.annotate("reliability ceiling of this metric (%.2f)" % ceil.mean(),
                xy=(len(order) - 0.5, ceil.mean()), xytext=(0, -11),
                textcoords="offset points", ha="right", fontsize=6.5, color=INK_MUTED)
    ax.axhline(0, color=GRID, lw=0.6)
    ax.set_ylim(-0.08, 0.95)
    ax.set_ylabel("group ordering recovered\n(mean per-question Spearman)")
    ax.set_xticks(x, [pretty(t).replace(" $\\times$ ", "×") for t in order],
                  rotation=30, ha="right", fontsize=6.5)
    ax.legend(frameon=False, fontsize=6.3, loc="center left")
    ax.set_title("Group discrimination: neither one has it", pad=4)
    ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=2)

    fig.suptitle("Reading the map beats listening to the mouth — but not by "
                 "knowing the group", fontsize=9, y=1.05)
    save(fig, "fig_probe")


if __name__ == "__main__":
    print("membaca peta kesetiaan ...")
    df = pd.read_csv(MAP_CSV)
    fig_headmap(df)
    fig_layercurve(df)
    fig_starhead(df)
    print("membaca sweep kausal ...")
    fig_causal_sweep()
    fig_fidelity_vs_causal()
    print("membaca hasil probe ...")
    fig_probe()
    print("selesai. figur di", OUT)
