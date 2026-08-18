"""Analisis robustness lintas-checkpoint (notebook 15): peta + mulut.

Per checkpoint (base / mid / osim), per tipe:
  - kesetiaan residual terbaik & head terbaik (Tmean)
  - koreksi seleksi: max-statistic permutation (juara-vs-juara, 2000x)
    + held-out template (angka jujur)
  - kesetiaan OUTPUT: RSA antara RDM prediksi-mulut dan RDM survei
  - akurasi mulut absolut (WD rata-rata ke distribusi asli)
Lalu tabel perbandingan base -> mid -> osim.

Validasi: `--selftest` menjalankan pipeline yang sama di data Mistral
(notebook 09) dan membandingkan dengan angka finding 05/06 yang sudah
dipublikasikan di catatan.

Jalankan dari root repo:
  ./venv/Scripts/python.exe analisis_lokal/robustness_map.py            # data notebook 15
  ./venv/Scripts/python.exe analisis_lokal/robustness_map.py --selftest # validasi di data lama
"""
import ast
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wasserstein_distance

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _common import data_path, results_dir

NB09 = results_dir("04_fidelity_map")
NB15 = results_dir("09_robustness_cross_model")
N_PERM = 2000
SEED = 42


# ---------- penggaris survei (dipakai semua model) ----------
def load_survey():
    real = np.load(f"{NB09}/group_real_dist.npy")
    keys = list(np.load(f"{NB09}/emb_resid.npz", allow_pickle=True)["group_keys"])
    types = np.array([k.split(" :: ")[0] for k in keys])
    return real, keys, types


def clean_subset(real, idx):
    """Buang sel yang bikin pasangan NaN (per tipe), seperti notebook 10."""
    idx = list(idx)
    while True:
        sub = real[np.ix_(idx, idx)]
        iu = np.triu_indices(len(idx), 1)
        bad = np.isnan(sub[iu])
        if not bad.any():
            return np.array(idx)
        nan_count = np.isnan(sub).sum(0)
        idx.pop(int(np.argmax(nan_count)))


def spearman_matrix(D_rows, target):
    """Spearman tiap baris D_rows (n_loc, n_pair) lawan target (n_pair,)."""
    rt = rankdata(target)
    rt = (rt - rt.mean()) / rt.std()
    rr = np.apply_along_axis(rankdata, 1, D_rows)
    with np.errstate(invalid="ignore"):   # baris konstan (mis. resid layer 0) -> NaN
        rr = (rr - rr.mean(1, keepdims=True)) / rr.std(1, keepdims=True)
    return rr @ rt / len(target)


def cosine_rows(X):
    """X (n_loc, n_cell, d) -> jarak cosine per lokasi (n_loc, n_pair)."""
    Xn = X / (np.linalg.norm(X, axis=2, keepdims=True) + 1e-8)
    sims = np.einsum("lcd,lkd->lck", Xn, Xn)
    n = X.shape[1]
    iu = np.triu_indices(n, 1)
    return 1.0 - sims[:, iu[0], iu[1]]


def analyse_map(emb_heads, emb_resid, real, types, tag, n_perm=N_PERM):
    """emb_heads (4,169,L,H,dh); emb_resid (4,169,L+1,hidden)."""
    Tm_h = emb_heads.astype(np.float32).mean(0)   # (169,L,H,dh)
    Tm_r = emb_resid.astype(np.float32).mean(0)   # (169,L+1,hidden)
    n_cell, L, H, dh = Tm_h.shape
    rng = np.random.default_rng(SEED)
    rows = []
    for ty in sorted(set(types)):
        idx = clean_subset(real, np.where(types == ty)[0])
        sub_real = real[np.ix_(idx, idx)]
        iu = np.triu_indices(len(idx), 1)
        target = sub_real[iu]

        # residual per layer
        Xr = Tm_r[idx].transpose(1, 0, 2)             # (L+1,cell,hidden)
        rho_r = spearman_matrix(cosine_rows(Xr), target)
        resid_best = np.nanmax(rho_r[1:])             # layer 0 degenerate

        # heads
        Xh = Tm_h[idx].reshape(len(idx), L * H, dh).transpose(1, 0, 2)
        D = cosine_rows(Xh)                           # (L*H, n_pair)
        rho_h = spearman_matrix(D, target)
        best = int(np.nanargmax(rho_h))
        bl, bh = divmod(best, H)

        # max-stat permutation (acak label sel, juara lawan juara)
        null_max = np.empty(n_perm)
        m = len(idx)
        ia, ib = iu  # indeks pasangan (di dalam subset)
        rr = np.apply_along_axis(rankdata, 1, D)
        rr = (rr - rr.mean(1, keepdims=True)) / rr.std(1, keepdims=True)
        for p in range(n_perm):
            perm = rng.permutation(m)
            tgt_p = sub_real[perm[ia], perm[ib]]
            rt = rankdata(tgt_p)
            rt = (rt - rt.mean()) / rt.std()
            null_max[p] = (rr @ rt / len(target)).max()
        p_corr = float((null_max >= rho_h[best]).mean())

        # held-out template (pilih di 3, ukur di 1, rata-rata 4 fold).
        # CATATAN: varian ini memilih head lewat rata-rata EMBEDDING 3
        # template (bukan prosedur cek1 notebook 10 persis), jadi angkanya
        # sedikit lebih konservatif. Pakai HANYA utk perbandingan
        # antar-checkpoint (pipeline sama) -- jangan disandingkan langsung
        # dgn kolom heldout finding 06.
        ho = []
        for t_out in range(4):
            t_in = [t for t in range(4) if t != t_out]
            Xi = emb_heads[t_in].astype(np.float32).mean(0)[idx]
            Di = cosine_rows(Xi.reshape(len(idx), L * H, dh).transpose(1, 0, 2))
            sel = int(np.nanargmax(spearman_matrix(Di, target)))
            Xo = emb_heads[t_out].astype(np.float32)[idx]
            Do = cosine_rows(Xo.reshape(len(idx), L * H, dh).transpose(1, 0, 2)[sel:sel + 1])
            ho.append(float(spearman_matrix(Do, target)[0]))
        rows.append(dict(model=tag, tipe=ty, n_sel=len(idx),
                         resid_best=float(resid_best),
                         head_best=float(rho_h[best]), head_loc=f"L{bl} H{bh}",
                         null_p95=float(np.percentile(null_max, 95)),
                         p_maxstat=p_corr, heldout_mean=float(np.mean(ho))))
        print(f"  [{tag}|{ty}] resid {resid_best:+.3f} | head {rho_h[best]:+.3f} "
              f"@L{bl}H{bh} | p={p_corr:.4f} | held-out {np.mean(ho):+.3f}")
    return pd.DataFrame(rows)


# ---------- kesetiaan output (mulut) ----------
def analyse_mouth(csv_path, real, keys, types, tag):
    raw = pd.read_csv(data_path("opinionqa_intersectional.csv"))
    raw["ordinal"] = raw["ordinal"].apply(ast.literal_eval)
    raw["responses"] = raw["responses"].apply(ast.literal_eval)
    raw["gk"] = raw["attribute"] + " :: " + raw["group"]
    REAL = {(r.gk, r.qkey): (np.array(r.responses, float), np.array(r.ordinal, float))
            for r in raw.itertuples()}

    d = pd.read_csv(csv_path)
    d["p"] = d["pred"].apply(lambda s: np.fromstring(s, sep=","))
    key_index = {k: i for i, k in enumerate(keys)}
    rows = []
    for ty, g in d.groupby("ty"):
        # akurasi absolut mulut
        wds = []
        for r in g.itertuples():
            pr, od = REAL[(r.gk, r.qk)]
            wds.append(wasserstein_distance(od, od, u_weights=r.p, v_weights=pr))
        # RDM output antar sel
        cells = sorted(g.gk.unique())
        pred_by = {(r.gk, r.qk): r.p for r in g.itertuples()}
        q_by = g.groupby("gk")["qk"].apply(set).to_dict()
        m = len(cells)
        out_rdm = np.full((m, m), np.nan)
        for a in range(m):
            for b in range(a + 1, m):
                shared = q_by[cells[a]] & q_by[cells[b]]
                if not shared:
                    continue
                ws = []
                for qk in shared:
                    od = REAL[(cells[a], qk)][1]
                    ws.append(wasserstein_distance(od, od,
                                                   u_weights=pred_by[(cells[a], qk)],
                                                   v_weights=pred_by[(cells[b], qk)]))
                out_rdm[a, b] = out_rdm[b, a] = np.mean(ws)
        idx = [key_index[c] for c in cells]
        sub_real = real[np.ix_(idx, idx)]
        iu = np.triu_indices(m, 1)
        ok = ~np.isnan(sub_real[iu]) & ~np.isnan(out_rdm[iu])
        from scipy.stats import spearmanr
        rho = float(spearmanr(sub_real[iu][ok], out_rdm[iu][ok]).statistic)
        rows.append(dict(model=tag, tipe=ty, wd_mulut=float(np.mean(wds)),
                         rho_output=rho, n_pair=int(ok.sum())))
        print(f"  [{tag}|{ty}] WD mulut {np.mean(wds):.4f} | rho output {rho:+.3f}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    real, keys, types = load_survey()

    if "--selftest" in sys.argv:
        print("=== SELFTEST di data Mistral (notebook 09) ===")
        eh = np.load(f"{NB09}/emb_heads.npz")["emb"]
        er = np.load(f"{NB09}/emb_resid.npz")["emb"]
        res = analyse_map(eh, er, real, types, "mistral", n_perm=500)
        print(res.round(3).to_string(index=False))
        print("\nBandingkan dgn finding 05/06: AGE head ~0.74 (L11 H19), "
              "resid ~0.51; EDU 0.68/0.61; RELIGxPP 0.60 (L11 H16)/0.26.")
        sys.exit(0)

    pass  # results_dir() sudah bikin folder
    maps, mouths = [], []
    for tag in ["base", "mid", "osim"]:
        print(f"=== {tag} ===")
        eh = np.load(f"{NB15}/emb_heads_{tag}.npz")["emb"]
        er = np.load(f"{NB15}/emb_resid_{tag}.npz")["emb"]
        maps.append(analyse_map(eh, er, real, types, tag))
        mouths.append(analyse_mouth(f"{NB15}/mouth_preds_{tag}.csv", real, keys, types, tag))
    M = pd.concat(maps)
    U = pd.concat(mouths)
    M.to_csv(f"{NB15}/map_summary.csv", index=False)
    U.to_csv(f"{NB15}/mouth_summary.csv", index=False)
    print("\n=== PETA (per checkpoint) ===")
    print(M.round(3).to_string(index=False))
    print("\n=== MULUT (per checkpoint) ===")
    print(U.round(3).to_string(index=False))
