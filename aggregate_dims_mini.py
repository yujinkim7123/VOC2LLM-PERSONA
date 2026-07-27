# -*- coding: utf-8 -*-
"""
dimension 집계 전용 — API 호출 없음(무료), 기존 mini 채점 결과만 읽어서
5차원(keyword/value/pain/behavioral/sentiment) 평균/표준편차를 집계합니다.
실행: python aggregate_dims_mini.py
출력: statistics/dim_table2_mini.json
"""
import os, json
import numpy as np

BASE_DIR = r""
EVAL_SUBDIR = "evaluation_results_mini"
N_RUNS = 20

FILE_KEYS = ["voc_en", "vad_en"]
EXP_IDS = [
    "BASELINE",
    "EXP1_SNA", "EXP2_LDA", "EXP3_KMeans",
    "EXP4_SNA_LDA", "EXP5_LDA_KMeans", "EXP6_SNA_KMeans", "EXP7_Full",
    "EXP8_SNA_LDA_COMBO", "EXP9_LDA_KMeans_COMBO",
    "EXP10_SNA_KMeans_COMBO", "EXP11_Full_COMBO",
    "EXP_RAW",
]
DIMS = ["keyword_alignment", "value_alignment", "pain_alignment",
        "behavioral_alignment", "sentiment_alignment"]

rows = []
for fk in FILE_KEYS:
    for exp in EXP_IDS:
        per_dim = {d: [] for d in DIMS}
        for run_id in range(1, N_RUNS + 1):
            p = os.path.join(BASE_DIR, f"run_{run_id:02d}", EVAL_SUBDIR, fk, f"{exp}_eval.json")
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8") as f:
                r = json.load(f)
            da = r.get("judge", {}).get("dim_averages", {})
            for d in DIMS:
                if d in da:
                    per_dim[d].append(da[d])
        if not per_dim[DIMS[0]]:
            print(f"[SKIP] {fk} {exp}: 데이터 없음")
            continue
        row = {"file_key": fk, "exp_id": exp, "judge_model": "gpt-4o-mini",
               "n": len(per_dim[DIMS[0]])}
        for d in DIMS:
            row[f"{d}_mean"] = round(float(np.mean(per_dim[d])), 4)
            row[f"{d}_std"] = round(float(np.std(per_dim[d], ddof=1)), 4) if len(per_dim[d]) > 1 else 0.0
        rows.append(row)
        print(f"[OK] {fk} {exp} (n={row['n']}): " +
              " ".join(f"{d.split('_')[0]}={row[f'{d}_mean']:.3f}" for d in DIMS))

out_dir = os.path.join(BASE_DIR, "statistics")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "dim_table2_mini.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print(f"\n저장 완료: {out}")
print("이 파일을 업로드해주세요.")
