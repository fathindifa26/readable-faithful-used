"""Review round 3 (Fable 5), S1: does the L11H16 fidelity map survive when it
is measured in QA CONTEXT (at the opinion-answer token position) instead of on
the identity-only prompt as in §5?

The reviewer's worry: the fidelity-causal dissociation (finding 11) might not
mean "the model does not consult the map", but rather "the map itself fades
once an opinion question is present in the context" -- because §5 is measured
on an identity prompt WITHOUT a question (last token, averaged over 4
templates), whereas §6 (causal + probe) runs in full QA context, at a
different opinion-answer token position.

The QA-context data already exists, no new GPU run needed: probeA_<type>.npz
(notebook 14) stores vecs_l11 (4096-dim, the WHOLE layer-11 residual, at the
opinion-answer position) per (cell, question) -- exactly the input the
probe/causal analysis uses in §6. Here we slice out the 128 dims belonging to
head 16 (head ordering is the same as for patching: head h = dims
[h*128:(h+1)*128], see the notebook 14 hook `_oproj_prehook`), average per
cell across questions (analogous to the "mean of 4 templates" in the original
map), then run RSA against EXACTLY the real-survey distances used by the
original map (group_real_dist.npy, notebook 09), subset to the same cells via
`gk`. The "real" side is identical to the original map; only the model side
differs in context -- so if rho changes, that is purely a context effect, not
a different distance definition.

The identity-only number is recomputed here from emb_heads.npz (Tmean, L11H16)
using the SAME cells (rather than taken from fidelity_map_full.csv, which is
per-template) -- so that the identity-only vs QA-context comparison is really
apple-to-apple (same cells, same RDM definition, only the source of the
vectors differs).

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
Tm = Z09["emb"].astype(np.float32).mean(0)   # (n_g_total, 32, 32, 128), mean of 4 templates


def rdm_cos(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    S = Xn @ Xn.T
    iu = np.triu_indices(len(X), 1)
    return 1.0 - S[iu]


def clean_and_align(cells):
    """Drop cells missing from the original map, then drop the ones that make
    a NaN pair in the survey distances (as in notebook 10)."""
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
    assert len(df) == len(v11), f"{ty}: csv/npz row counts differ ({len(df)} vs {len(v11)})"
    h16 = v11[:, STAR_HEAD * HEAD_DIM:(STAR_HEAD + 1) * HEAD_DIM].astype(np.float32)

    cells = clean_and_align(sorted(df.gk.unique()))
    if len(cells) < 3:
        print(f"[{ty}] skipped, fewer than 3 cells left after cleaning")
        continue
    pos = [key_pos[c] for c in cells]
    real_sub = real_all[np.ix_(pos, pos)]
    iu = np.triu_indices(len(cells), 1)
    d_real = real_sub[iu]

    # ---- QA-context (v1): mean head16 across ALL questions of that cell ----
    gk_to_rows = df.groupby("gk").indices
    n_q_mean = float(np.mean([len(gk_to_rows[c]) for c in cells]))
    X_qa = np.stack([h16[gk_to_rows[c]].mean(0) for c in cells])
    rho_qa = spearmanr(rdm_cos(X_qa), d_real).statistic

    # ---- QA-context (v2, question-composition control): per PAIR, only the
    # questions answered by BOTH cells. Question coverage is only >=60% per
    # cell, so in v1 each cell is averaged over a different subset of questions
    # -- that difference in subset composition can itself contribute to the
    # between-cell distance, quite apart from identity.
    # This is exactly the construction used on the survey side (see
    # rsa_reliability_ceiling.py): a pair distance is computed over the shared
    # questions only.
    qsets = {c: set(df.qk.values[gk_to_rows[c]]) for c in cells}
    vec_by = {}                       # (cell, qk) -> head16 vector
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

    # ---- QA-context (v3, the cleanest control): ONE set of questions answered
    # by ALL cells of this type. Each cell still gets a single vector (exactly
    # analogous to identity-only), the question composition is identical across
    # cells, and unlike v2 the distance scale is uniform across pairs (in v2
    # every pair uses a different question set -> distances are not comparable
    # across pairs + attenuation from having fewer questions). This is the
    # fairest comparison for the §5 number.
    q_common = sorted(set.intersection(*[qsets[c] for c in cells]))
    if len(q_common) >= 10:
        X_common = np.stack([np.mean([vec_by[(c, q)] for q in q_common], axis=0)
                             for c in cells])
        rho_qa_common = float(spearmanr(rdm_cos(X_common), d_real).statistic)
    else:
        rho_qa_common = float("nan")

    # ---- v4, the separator: the NUMBER of questions is matched to v3 but the
    # COMPOSITION is left free to differ (each cell draws len(q_common) of its
    # own questions at random).
    # If v4 ~ v1 -> the v3 drop is purely composition (confound N2 is real).
    # If v4 ~ v3 -> the drop is only attenuation from having fewer questions.
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

    # ---- v3 RDM reliability (split-half over the common questions): how much
    # of the v3 drop is merely measurement noise (far fewer questions than in
    # v1) vs a real degradation. Split q_common into 2 halves, build 2 RDMs,
    # correlate them. This is the right denominator for reading rho v3 --
    # parallel to r_model_star in rsa_reliability_ceiling.py for the §5 side.
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

    # ---- identity-only: Tmean L11H16, the SAME cells ----
    X_id = Tm[pos, STAR_LAYER, STAR_HEAD, :]
    rho_identity = spearmanr(rdm_cos(X_id), d_real).statistic

    rows.append(dict(type=ty, n_cells=len(cells), n_questions_mean_per_cell=n_q_mean,
                      n_shared_questions_min=int(np.min(n_shared)),
                      n_shared_questions_mean=float(np.mean(n_shared)),
                      n_questions_common_to_all_cells=len(q_common),
                      rho_identity_only=float(rho_identity),
                      rho_qa_context=float(rho_qa),
                      rho_qa_shared_q=float(rho_qa_shared),
                      rho_qa_common_q=rho_qa_common,
                      rho_qa_nmatched=rho_qa_nmatched,
                      rdm_reliability_v3=rel_v3))
    print(f"[{ty}] n_cells={len(cells):>2} | ident {rho_identity:+.3f} "
          f"| v1 all-questions {rho_qa:+.3f} "
          f"| v2 shared/pair {rho_qa_shared:+.3f} "
          f"| v3 common-across-cells (n={len(q_common)}) {rho_qa_common:+.3f} "
          f"| v4 n-matched {rho_qa_nmatched:+.3f} "
          f"| RDM-v3-reliability {rel_v3:+.3f}")

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/qa_context_fidelity.csv", index=False)
print("\nsaved -> qa_context_fidelity.csv")
