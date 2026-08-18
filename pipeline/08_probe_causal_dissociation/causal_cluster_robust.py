"""Review A4 + A3 + B5: inferensi kausal yang menghormati CLUSTERING pasangan.

Masalah (review A4): n=240 item = 12 pasangan x 20 soal. Item dalam satu
pasangan berbagi persona/arah/idiosinkrasi -> sangat berkorelasi. Sign-flip
level item memperlakukan 240 sebagai independen -> p terlalu kecil.

Perbaikan di sini, SEMUA di level pasangan (12 cluster):
  1. tab:sweep   : max-stat permutation dgn flip level-PASANGAN (semua 4096
                   kombinasi flip dihitung EKSAK, bukan sampel).
  2. tab:fixedlayer : p satu-sisi L11 & L1, flip level pasangan (eksak).
  3. tab:patchv2 : finding 08, per kondisi vs kontrol random-head DAN vs nol,
                   flip level pasangan (eksak).
  4. B5 sensitivitas: efek diuji juga TANPA pengurangan kontrol (vs nol) --
     shift_ke_realB sendiri sudah relatif ke prediksi TANPA patch.
  5. A3: uji beda ANTAR tipe (bukan significant-vs-nonsignificant):
     bootstrap 10.000x selisih t level-pasangan antara dua tipe.

Output: notebooks/output/14_.../causal_pairlevel.csv (+ print).
"""
import itertools

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _common import data_path, results_dir

OUT = results_dir("08_probe_causal_dissociation")
NB07 = results_dir("07_sweep_and_probe")
NB06 = results_dir("06_patching_v2")
SEED = 42


def tstat(x):
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


def exact_flips(n):
    """Semua 2^n kombinasi tanda utk n cluster (n<=14)."""
    return np.array(list(itertools.product([-1.0, 1.0], repeat=n)))


def p_onesided_exact(pairmeans):
    """p satu-sisi eksak: dari semua kombinasi flip, berapa fraksi t >= t_obs."""
    F = exact_flips(len(pairmeans))
    M = F * pairmeans[None, :]
    ts = M.mean(1) / (M.std(1, ddof=1) / np.sqrt(M.shape[1]))
    return float((ts >= tstat(pairmeans) - 1e-12).mean())


# ================= 1+2: sweep & lokasi tetap, level pasangan =================
d = pd.concat([pd.read_csv(f"{NB07}/sweep_rows.csv"),
               pd.read_csv(f"{OUT}/sweep6_rows.csv")])
d["diff"] = d.shift_ke_realB - d.shift_ctrl_ke_realB

rows = []
for ty, g in d.groupby("attr_type"):
    # per-pair mean per layer -> (n_pair, n_layer)
    pm = g.pivot_table(index="pair", columns="layer", values="diff", aggfunc="mean")
    pm_raw = g.pivot_table(index="pair", columns="layer", values="shift_ke_realB",
                           aggfunc="mean")
    M = pm.to_numpy()
    n_pair, n_layer = M.shape
    t_obs = M.mean(0) / (M.std(0, ddof=1) / np.sqrt(n_pair))
    layers = list(pm.columns)

    # max-stat EKSAK di level pasangan: flip yang sama utk semua layer
    F = exact_flips(n_pair)                       # (4096, n_pair)
    TS = np.empty((len(F), n_layer))
    for j in range(n_layer):
        Mf = F * M[None, :, j]
        TS[:, j] = Mf.mean(1) / (Mf.std(1, ddof=1) / np.sqrt(n_pair))
    null_max = TS.max(1)
    win = int(np.argmax(t_obs))
    p_max = float((null_max >= t_obs[win] - 1e-12).mean())

    i11, i1 = layers.index(11), layers.index(1)
    rows.append(dict(
        tipe=ty, n_pair=n_pair,
        win_layer=layers[win], t_win_pair=t_obs[win],
        null_p95_pair=float(np.percentile(null_max, 95)), p_maxstat_pair=p_max,
        t_L11_pair=t_obs[i11], p_L11_pair=p_onesided_exact(M[:, i11]),
        t_L1_pair=t_obs[i1], p_L1_pair=p_onesided_exact(M[:, i1]),
        # B5: tanpa pengurangan kontrol (vs nol)
        t_L11_vs0=tstat(pm_raw.to_numpy()[:, i11]),
        p_L11_vs0=p_onesided_exact(pm_raw.to_numpy()[:, i11]),
        t_win_vs0=tstat(pm_raw.to_numpy()[:, win]),
        p_win_vs0=p_onesided_exact(pm_raw.to_numpy()[:, win]),
    ))

sweep_pair = pd.DataFrame(rows).sort_values("p_maxstat_pair")
pd.set_option("display.width", 250)
print("=== SWEEP, level PASANGAN (n=12, flip eksak 4096) ===")
print(sweep_pair.round(4).to_string(index=False))

# ================= 3: finding 08 (patching v2), level pasangan ==============
pv = pd.read_csv(f"{NB06}/patching_v2_rows.csv")
CONDS = ["patch_L11_all32", "patch_L11H16", "patch_L11L18_all", "patch_L11_all32_x5"]
rows2 = []
for ty, g in pv.groupby("attr_type"):
    ctrl = g[g.condition == "patch_randhead"].set_index(["pair", "qkey"])["shift_ke_realB"]
    for cond in CONDS:
        gc = g[g.condition == cond].set_index(["pair", "qkey"])["shift_ke_realB"]
        joined = pd.concat([gc.rename("x"), ctrl.rename("c")], axis=1).dropna()
        joined = joined.reset_index()
        pm_diff = joined.groupby("pair").apply(
            lambda r: (r.x - r.c).mean(), include_groups=False).to_numpy()
        pm_raw = joined.groupby("pair")["x"].mean().to_numpy()
        rows2.append(dict(
            tipe=ty, kondisi=cond, n_pair=len(pm_diff),
            mean_shift=float(joined.x.mean()),
            t_vs_ctrl=tstat(pm_diff), p_vs_ctrl_pair=p_onesided_exact(pm_diff),
            t_vs_0=tstat(pm_raw), p_vs_0_pair=p_onesided_exact(pm_raw),
        ))
pv_pair = pd.DataFrame(rows2)
print("\n=== PATCHING V2 (finding 08), level PASANGAN ===")
print(pv_pair.round(4).to_string(index=False))

# ================= 4 (A3): uji beda ANTAR tipe ==============================
# Selisih t level-pasangan, bootstrap 10.000x + permutasi dua-sampel.
rng = np.random.default_rng(SEED)


def pairmeans_at(ty, layer):
    g = d[d.attr_type == ty]
    pm = g.pivot_table(index="pair", columns="layer", values="diff", aggfunc="mean")
    return pm[layer].to_numpy()


def diff_test(tyA, layA, tyB, layB, n_boot=10000):
    a, b = pairmeans_at(tyA, layA), pairmeans_at(tyB, layB)
    t_a, t_b = tstat(a), tstat(b)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = (tstat(rng.choice(a, len(a), replace=True))
                    - tstat(rng.choice(b, len(b), replace=True)))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # permutasi dua-sampel (tukar label tipe antar pasangan)
    pool = np.concatenate([a, b])
    obs = t_a - t_b
    cnt = 0
    for _ in range(10000):
        perm = rng.permutation(len(pool))
        pa, pb = pool[perm[:len(a)]], pool[perm[len(a):]]
        if abs(tstat(pa) - tstat(pb)) >= abs(obs) - 1e-12:
            cnt += 1
    return dict(A=f"{tyA}@L{layA}", B=f"{tyB}@L{layB}", t_A=t_a, t_B=t_b,
                dt=obs, ci_lo=lo, ci_hi=hi, p_perm2s=cnt / 10000)


print("\n=== A3: uji beda antar tipe (bootstrap CI selisih t + permutasi 2-sampel) ===")
tests = [
    # counterexample #1: EDU (peta terbaik) vs tipe yang lolos, di layer juara masing2
    ("EDUCATIONxINCOME", 9, "RACExPOLIDEOLOGY", 11),
    ("EDUCATIONxINCOME", 9, "AGExPOLPARTY", 15),
    ("EDUCATIONxINCOME", 9, "RELIGxPOLPARTY", 8),
    # di lokasi TETAP L11
    ("EDUCATIONxINCOME", 11, "RACExPOLIDEOLOGY", 11),
    ("EDUCATIONxINCOME", 11, "AGExPOLPARTY", 11),
]
res3 = pd.DataFrame([diff_test(*t) for t in tests])
print(res3.round(4).to_string(index=False))

sweep_pair.to_csv(f"{OUT}/causal_pairlevel_sweep.csv", index=False)
pv_pair.to_csv(f"{OUT}/causal_pairlevel_patchv2.csv", index=False)
res3.to_csv(f"{OUT}/causal_typediff_tests.csv", index=False)
print("\ndisimpan: causal_pairlevel_sweep.csv, causal_pairlevel_patchv2.csv, causal_typediff_tests.csv")
