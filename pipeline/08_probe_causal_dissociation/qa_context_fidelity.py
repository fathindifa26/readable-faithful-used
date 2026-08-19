"""Review ronde 3 (Fable 5), S1: apakah peta kesetiaan L11H16 bertahan
kalau diukur di KONTEKS QA (posisi token jawaban-opini), bukan di prompt
identitas-saja seperti §5?

Kekhawatiran reviewer: disosiasi fidelity-kausal (finding 11) bisa jadi
bukan "model tidak mengonsultasi peta", tapi "peta itu sendiri luntur
begitu pertanyaan opini hadir dalam konteks" -- karena §5 diukur di prompt
identitas TANPA pertanyaan (token terakhir, rata-rata 4 template),
sedangkan §6 (causal + probe) berjalan di konteks QA penuh, posisi token
jawaban-opini yang beda.

Data QA-context sudah ada, tidak perlu run GPU baru: probeA_<tipe>.npz
(notebook 14) menyimpan vecs_l11 (4096-dim, residual layer-11 UTUH,
posisi jawaban-opini) per (sel, soal) -- persis input yang dipakai
probe/causal di §6. Di sini kita slice 128 dim milik head 16 (urutan
head sama dgn patching: head h = dim [h*128:(h+1)*128], lihat notebook
14 hook `_oproj_prehook`), rata-ratakan per sel lintas soal (analog
"rata-rata 4 template" di peta asli), lalu RSA lawan jarak-survei-asli
PERSIS yang dipakai peta asli (group_real_dist.npy, notebook 09),
disubset ke sel yang sama lewat `gk`. Sisi "real" identik dgn peta asli;
cuma sisi model yang beda konteks -- jadi kalau rho berubah, itu murni
efek konteks, bukan definisi jarak yang beda.

Angka identitas-saja dihitung ulang di sini dari emb_heads.npz (Tmean,
L11H16) memakai sel yang SAMA (bukan diambil dari peta_kesetiaan_full.csv
yang per-template) -- supaya perbandingan identitas-saja vs QA-context
benar-benar apple-to-apple (sel sama, definisi RDM sama, cuma sumber
vektor beda).

Output: notebooks/output/14_.../qa_context_fidelity.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _common import results_dir

NB09 = results_dir("04_fidelity_map")
OUT = results_dir("08_probe_causal_dissociation")
STAR_LAYER, STAR_HEAD, HEAD_DIM = 11, 16, 128

TYPES = ["AGExPOLPARTY", "EDUCATIONxINCOME", "RELIGxPOLPARTY",
         "RACExPOLPARTY", "RACExPOLIDEOLOGY", "RACExRELIG"]

real_all = np.load(f"{NB09}/group_real_dist.npy")
Z09 = np.load(f"{NB09}/emb_heads.npz", allow_pickle=True)
map_keys = list(Z09["group_keys"])
key_pos = {k: i for i, k in enumerate(map_keys)}
Tm = Z09["emb"].astype(np.float32).mean(0)   # (n_g_total, 32, 32, 128), rata2 4 template


def rdm_cos(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    S = Xn @ Xn.T
    iu = np.triu_indices(len(X), 1)
    return 1.0 - S[iu]


def clean_and_align(cells):
    """Buang sel yg nggak ada di peta asli, lalu buang yg bikin pasangan
    NaN di jarak-survei (spt notebook 10)."""
    cells = [c for c in cells if c in key_pos]
    keep = list(range(len(cells)))
    while True:
        pos = [key_pos[cells[i]] for i in keep]
        sub = real_all[np.ix_(pos, pos)]
        iu = np.triu_indices(len(keep), 1)
        if len(keep) < 3 or not np.isnan(sub[iu]).any():
            break
        keep.pop(int(np.argmax(np.isnan(sub).sum(0))))
    return [cells[i] for i in keep]


rows = []
for ty in TYPES:
    df = pd.read_csv(f"{OUT}/probeA_{ty}.csv").reset_index(drop=True)
    v11 = np.load(f"{OUT}/probeA_{ty}.npz")["vecs_l11"]
    assert len(df) == len(v11), f"{ty}: csv/npz baris beda ({len(df)} vs {len(v11)})"
    h16 = v11[:, STAR_HEAD * HEAD_DIM:(STAR_HEAD + 1) * HEAD_DIM].astype(np.float32)

    cells = clean_and_align(sorted(df.gk.unique()))
    if len(cells) < 3:
        print(f"[{ty}] dilewati, sel tersisa < 3 setelah cleaning")
        continue
    pos = [key_pos[c] for c in cells]
    real_sub = real_all[np.ix_(pos, pos)]
    iu = np.triu_indices(len(cells), 1)
    d_real = real_sub[iu]

    # ---- QA-context (v1): rata2 head16 lintas SEMUA soal sel itu ----
    gk_to_rows = df.groupby("gk").indices
    n_q_mean = float(np.mean([len(gk_to_rows[c]) for c in cells]))
    X_qa = np.stack([h16[gk_to_rows[c]].mean(0) for c in cells])
    rho_qa = spearmanr(rdm_cos(X_qa), d_real).statistic

    # ---- QA-context (v2, kontrol komposisi soal): per PASANGAN, cuma soal
    # yang dijawab KEDUA sel. Cakupan soal cuma >=60% per sel, jadi di v1
    # tiap sel dirata-rata atas subset soal yang beda-beda -- beda komposisi
    # subset itu sendiri bisa nyumbang jarak antar-sel, lepas dari identitas.
    # Ini persis konstruksi yang dipakai sisi survei (lih. rsa_reliability_
    # ceiling.py): jarak pasangan dihitung cuma atas soal bersama.
    qsets = {c: set(df.qk.values[gk_to_rows[c]]) for c in cells}
    vec_by = {}                       # (sel, qk) -> vektor head16
    for c in cells:
        for r in gk_to_rows[c]:
            vec_by[(c, df.qk.values[r])] = h16[r]
    d_shared, n_shared = [], []
    for a, b in zip(*iu):
        ca, cb = cells[a], cells[b]
        qs = sorted(qsets[ca] & qsets[cb])
        n_shared.append(len(qs))
        va = np.mean([vec_by[(ca, q)] for q in qs], axis=0)
        vb = np.mean([vec_by[(cb, q)] for q in qs], axis=0)
        va = va / (np.linalg.norm(va) + 1e-8)
        vb = vb / (np.linalg.norm(vb) + 1e-8)
        d_shared.append(1.0 - float(va @ vb))
    rho_qa_shared = spearmanr(np.array(d_shared), d_real).statistic

    # ---- QA-context (v3, kontrol terbersih): SATU set soal yang dijawab
    # SEMUA sel tipe ini. Tiap sel tetap dapat satu vektor (analog persis
    # identitas-saja), komposisi soal identik antar-sel, dan tidak seperti
    # v2 skala jaraknya seragam antar pasangan (v2 tiap pasangan pakai set
    # soal beda -> jarak antar-pasangan tak sebanding + atenuasi krn soal
    # lebih sedikit). Ini pembanding yang paling adil buat angka §5.
    q_common = sorted(set.intersection(*[qsets[c] for c in cells]))
    if len(q_common) >= 10:
        X_common = np.stack([np.mean([vec_by[(c, q)] for q in q_common], axis=0)
                             for c in cells])
        rho_qa_common = float(spearmanr(rdm_cos(X_common), d_real).statistic)
    else:
        rho_qa_common = float("nan")

    # ---- v4, pemisah: JUMLAH soal dicocokkan ke v3 tapi KOMPOSISI dibiarkan
    # beda (tiap sel ambil acak sebanyak len(q_common) dari soalnya sendiri).
    # Kalau v4 ~ v1 -> jatuhnya v3 murni krn komposisi (confound N2 nyata).
    # Kalau v4 ~ v3 -> jatuhnya cuma atenuasi krn soal lebih sedikit.
    rng4 = np.random.default_rng(0)
    if len(q_common) >= 10:
        r4 = []
        for _ in range(20):
            Xs = []
            for c in cells:
                qs_c = sorted(qsets[c])
                pick = rng4.choice(len(qs_c), size=len(q_common), replace=False)
                Xs.append(np.mean([vec_by[(c, qs_c[k])] for k in pick], axis=0))
            r4.append(spearmanr(rdm_cos(np.stack(Xs)), d_real).statistic)
        rho_qa_nmatched = float(np.mean(r4))
    else:
        rho_qa_nmatched = float("nan")

    # ---- reliabilitas RDM v3 (belah-dua soal umum): seberapa banyak
    # penurunan v3 itu cuma noise pengukuran (soal jauh lebih sedikit
    # daripada v1) vs degradasi asli. Bagi q_common jadi 2 separuh, bangun
    # 2 RDM, korelasikan. Ini penyebut yang benar buat baca rho v3 --
    # sejajar dgn r_model_star di rsa_reliability_ceiling.py utk sisi §5.
    rng5 = np.random.default_rng(1)
    if len(q_common) >= 20:
        rs = []
        for _ in range(20):
            perm = rng5.permutation(len(q_common))
            hA = [q_common[k] for k in perm[:len(perm) // 2]]
            hB = [q_common[k] for k in perm[len(perm) // 2:]]
            XA = np.stack([np.mean([vec_by[(c, q)] for q in hA], axis=0) for c in cells])
            XB = np.stack([np.mean([vec_by[(c, q)] for q in hB], axis=0) for c in cells])
            rs.append(spearmanr(rdm_cos(XA), rdm_cos(XB)).statistic)
        rel_v3 = float(np.mean(rs))
    else:
        rel_v3 = float("nan")

    # ---- identitas-saja: Tmean L11H16, sel yang SAMA ----
    X_id = Tm[pos, STAR_LAYER, STAR_HEAD, :]
    rho_identity = spearmanr(rdm_cos(X_id), d_real).statistic

    rows.append(dict(tipe=ty, n_sel=len(cells), n_soal_rata2_per_sel=n_q_mean,
                      n_soal_bersama_min=int(np.min(n_shared)),
                      n_soal_bersama_rata2=float(np.mean(n_shared)),
                      n_soal_umum_semua_sel=len(q_common),
                      rho_identity_only=float(rho_identity),
                      rho_qa_context=float(rho_qa),
                      rho_qa_shared_q=float(rho_qa_shared),
                      rho_qa_common_q=rho_qa_common,
                      rho_qa_nmatched=rho_qa_nmatched,
                      reliabilitas_rdm_v3=rel_v3))
    print(f"[{ty}] n_sel={len(cells):>2} | ident {rho_identity:+.3f} "
          f"| v1 semua-soal {rho_qa:+.3f} "
          f"| v2 bersama/pasangan {rho_qa_shared:+.3f} "
          f"| v3 umum-semua-sel (n={len(q_common)}) {rho_qa_common:+.3f} "
          f"| v4 n-cocok {rho_qa_nmatched:+.3f} "
          f"| reliabilitas-RDM-v3 {rel_v3:+.3f}")

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/qa_context_fidelity.csv", index=False)
print("\ndisimpan -> qa_context_fidelity.csv")
