"""Bangun tabel sel demografi IRISAN (2 atribut) dari data respons individual
Pew ATP mentah, format sama seperti opinionqa.csv (qkey, attribute, group,
responses, ordinal, question, options) tapi 'group' sekarang kombinasi 2
atribut, mis. 'White | Protestant'.

Dibuat buat follow-up Tahap 0 (lihat notes/research_question/03_pivot2_group_consistency.md §12.9/§12.10):
tes RSA sebelumnya (notebooks/06_tahap0_rsa_kaggle.ipynb) cuma pakai
kelompok 1-atribut sebagai proxy; ini bikin data buat tes ulang pakai sel
irisan asli (mis. "Black Hindu"), target sebenarnya dari `L_group`.

Sumber: data/opinionqa_original/data/human_resp/<wave>/{responses.csv,info.csv}
  (unduh dulu -- lihat data/README.md, tidak ikut repo karena lisensi Pew ATP)
Output: data/opinionqa_intersectional.csv

Jalankan dari root repo: python data/build_intersectional_cells.py
"""
import ast
import glob
import os
import pandas as pd

# BASE = folder data/ ini sendiri (bukan root repo) -- lihat data/README.md
BASE = os.path.dirname(os.path.abspath(__file__))
WAVE_DIRS = sorted(glob.glob(os.path.join(BASE, "opinionqa_original/data/human_resp/American_Trends_Panel_W*")))
PEW_WAVES = [int(d.split("_W")[-1]) for d in WAVE_DIRS]

ATTR_PAIRS = [
    ("RACE", "RELIG"),
    ("RACE", "POLPARTY"),
    ("RACE", "POLIDEOLOGY"),
    ("RELIG", "POLPARTY"),
    ("EDUCATION", "INCOME"),
    ("AGE", "POLPARTY"),
]
N_THRESHOLD = 30  # ambang jumlah responden minimal per sel irisan (dalam 1 wave)

OUT_PATH = os.path.join(BASE, "opinionqa_intersectional.csv")


def parse_cell(x):
    if isinstance(x, str):
        return ast.literal_eval(x)
    return x


def process_wave(wave_dir, wave_num):
    responses = pd.read_csv(os.path.join(wave_dir, "responses.csv"), low_memory=False)
    info = pd.read_csv(os.path.join(wave_dir, "info.csv"))
    info["option_mapping"] = info["option_mapping"].apply(parse_cell)
    info["references"] = info["references"].apply(parse_cell)
    info["option_ordinal"] = info["option_ordinal"].apply(parse_cell)

    weight_col = f"WEIGHT_W{wave_num}"
    if weight_col not in responses.columns:
        print(f"  [skip wave {wave_num}] tidak ada kolom {weight_col}")
        return []

    rows = []
    for attr1, attr2 in ATTR_PAIRS:
        if attr1 not in responses.columns or attr2 not in responses.columns:
            continue
        combo_key = f"{attr1}x{attr2}"
        sub = responses[[attr1, attr2, weight_col]].copy()
        sub["group_label"] = sub[attr1].astype(str) + " | " + sub[attr2].astype(str)

        # buang baris dengan atribut kosong/NaN
        valid = sub[attr1].notna() & sub[attr2].notna()
        sub = sub[valid]

        cell_sizes = sub.groupby("group_label").size()
        valid_groups = set(cell_sizes[cell_sizes >= N_THRESHOLD].index)
        if not valid_groups:
            continue

        for _, qrow in info.iterrows():
            qkey = qrow["key"]
            if qkey not in responses.columns:
                continue
            ordinal = qrow["option_ordinal"]
            references = qrow["references"]
            if not isinstance(ordinal, list) or len(ordinal) == 0:
                continue
            ordinal_refs = references[: len(ordinal)]

            qdf = responses[[attr1, attr2, weight_col, qkey]].copy()
            qdf["group_label"] = qdf[attr1].astype(str) + " | " + qdf[attr2].astype(str)
            qdf = qdf[qdf["group_label"].isin(valid_groups)]
            qdf = qdf[qdf[qkey].apply(lambda v: isinstance(v, str))]
            if len(qdf) == 0:
                continue

            for group_label, gdf in qdf.groupby("group_label"):
                weighted_counts = {}
                for ref in ordinal_refs:
                    w = gdf.loc[gdf[qkey] == ref, weight_col].sum()
                    weighted_counts[ref] = w
                total = sum(weighted_counts.values())
                if total <= 0:
                    continue
                resp_vec = [float(weighted_counts[ref] / total) for ref in ordinal_refs]
                rows.append({
                    "qkey": qkey,
                    "attribute": combo_key,
                    "group": group_label,
                    "responses": resp_vec,
                    "ordinal": ordinal,
                    "question": qrow["question"],
                    "options": references,
                    "wave": wave_num,
                    "n_unweighted": int(cell_sizes[group_label]),
                })
    return rows


def main():
    all_rows = []
    for d, w in zip(WAVE_DIRS, PEW_WAVES):
        print(f"Proses wave {w} ...")
        rows = process_wave(d, w)
        print(f"  -> {len(rows)} baris (qkey x sel irisan)")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nTotal baris: {len(df)}")
    print(f"Kombinasi atribut: {sorted(df['attribute'].unique().tolist())}")
    print(f"Jumlah sel irisan unik: {df['group'].nunique()}")
    print(f"Jumlah qkey unik: {df['qkey'].nunique()}")
    print(f"\nDisimpan ke: {OUT_PATH}")

    print("\nJumlah sel per kombinasi atribut:")
    print(df.groupby("attribute")["group"].nunique())


if __name__ == "__main__":
    main()
