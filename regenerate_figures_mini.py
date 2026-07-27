# -*- coding: utf-8 -*-
"""
regenerate_figures_mini.py
=================================================================
GPT-4o-mini 재채점 결과 기준으로 논문 그림을 전부 재생성합니다.
API 호출 없음 (이미 저장된 통계 JSON만 읽음) - 완전 무료.

생성 파일:
  research_pipeline.png       - Figure 1 (파이프라인 다이어그램)
                                 ※ 데이터 무관, 재생성 불필요하지만
                                   요청 시 그대로 재실행 가능
  radar_chart_combined.png    - Figure 2 (5차원 레이더, VOC/VAD 나란히)
                                 ※ dim_table2_mini.json 필요
  fig_grouped_bar.png         - 부가: EXP별 3지표 그룹 막대
  fig_heatmap_cohens_d.png    - 부가: Cohen's d 효과크기 히트맵
  fig_complexity_paradox.png  - 부가: 그룹평균 비교 (single/pipeline/indep)

필요 입력 파일 (경로는 아래 CONFIG에서 수정):
  final_table3_mini.json      - 필수 (matching/validity, Table 4 기준)
  dim_table2_mini.json        - 필수 (5차원, Figure 2 기준)
                                 aggregate_dims_mini.py 로 먼저 생성

사용법:
  python regenerate_figures_mini.py
"""

import json
import os
from math import pi

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# =================================================================
# 0. 경로 설정 — 본인 환경에 맞게 수정
# =================================================================
FINAL_TABLE_MINI = r"C:\Users\User\OneDrive\바탕 화면\대학원\캡스톤\runs\statistics\final_table3_mini.json"
DIM_TABLE_MINI   = r"C:\Users\User\OneDrive\바탕 화면\대학원\캡스톤\runs\statistics\dim_table2_mini.json"
OUT_DIR          = r"C:\Users\User\OneDrive\바탕 화면\대학원\캡스톤\figures_mini"

os.makedirs(OUT_DIR, exist_ok=True)

# =================================================================
# 1. 공통 스타일 (기존 paper_visualizations2.py와 톤 통일)
# =================================================================
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#cccccc",
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "legend.framealpha": 0.9,
    "savefig.dpi":       300,
    "savefig.facecolor": "white",
    "savefig.bbox":      "tight",
})

DATASETS = ["voc_en", "vad_en"]
DS_NAMES = {"voc_en": "VOC (Synthetic, Top-Down)",
            "vad_en": "VAD (Real-World, Bottom-Up)"}

EXP_ORDER = [
    "BASELINE", "EXP1_SNA", "EXP2_LDA", "EXP3_KMeans",
    "EXP4_SNA_LDA", "EXP5_LDA_KMeans", "EXP6_SNA_KMeans", "EXP7_Full",
    "EXP8_SNA_LDA_COMBO", "EXP9_LDA_KMeans_COMBO",
    "EXP10_SNA_KMeans_COMBO", "EXP11_Full_COMBO", "EXP_RAW",
]
EXP_SHORT = {
    "BASELINE": "BASE", "EXP1_SNA": "EXP1\nSNA", "EXP2_LDA": "EXP2\nLDA",
    "EXP3_KMeans": "EXP3\nKMeans", "EXP4_SNA_LDA": "EXP4\nSNA+LDA",
    "EXP5_LDA_KMeans": "EXP5\nLDA+KM", "EXP6_SNA_KMeans": "EXP6\nSNA+KM",
    "EXP7_Full": "EXP7\nFull", "EXP8_SNA_LDA_COMBO": "EXP8\nSNA||LDA",
    "EXP9_LDA_KMeans_COMBO": "EXP9\nLDA||KM",
    "EXP10_SNA_KMeans_COMBO": "EXP10\nSNA||KM",
    "EXP11_Full_COMBO": "EXP11\nFull||", "EXP_RAW": "EXP\nRAW",
}

C_MATCH, C_VALID, C_TOPIC = "#2A7DC9", "#1A9E6E", "#7060CC"
C_BASE, C_EXP3, C_EXP7, C_EXP10, C_RAWREV = (
    "#AAAAAA", "#E85D24", "#6633AA", "#1A9E6E", "#E6A817")

# =================================================================
# 2. 데이터 로드
# =================================================================
def load_json(path, required=True):
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(
                f"\n[필요 파일 없음] {path}\n"
                "먼저 rejudge_mini.py / aggregate_dims_mini.py 를 실행하세요.\n")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

main_raw = load_json(FINAL_TABLE_MINI)
dim_raw  = load_json(DIM_TABLE_MINI, required=False)

table = {ds: {} for ds in DATASETS}
for row in main_raw:
    ds = row["file_key"]
    if ds not in table:
        continue
    table[ds][row["exp_id"]] = row

dim_table = {ds: {} for ds in DATASETS}
if dim_raw:
    for row in dim_raw:
        ds = row["file_key"]
        if ds not in dim_table:
            continue
        dim_table[ds][row["exp_id"]] = row


# =================================================================
# Figure 2 — 5차원 레이더 차트 (논문 Figure 2 재현)
#   BASELINE / EXP_RAW / EXP3(KMeans) / EXP7(Full Pipeline) /
#   EXP10(SNA||KMeans) 5개 조건, VOC/VAD 나란히
# =================================================================
def fig_radar_combined():
    if not dim_raw:
        print("[SKIP] radar_chart_combined.png : dim_table2_mini.json 없음")
        return

    DIMS = ["keyword_alignment", "value_alignment", "pain_alignment",
            "behavioral_alignment", "sentiment_alignment"]
    DIM_LABELS = ["Keyword\nAlignment", "Value\nAlignment", "Pain\nAlignment",
                  "Behavioral\nAlignment", "Sentiment\nAlignment"]
    N = len(DIMS)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    TARGET = {
        "BASELINE":              (C_BASE,   "--", 1.6, "BASELINE"),
        "EXP_RAW":               (C_RAWREV, ":",  1.8, "EXP_RAW"),
        "EXP3_KMeans":           (C_EXP3,   "-",  2.4, "EXP3 (KMeans)"),
        "EXP7_Full":             (C_EXP7,   "-",  2.0, "EXP7 (Full PL)"),
        "EXP10_SNA_KMeans_COMBO": (C_EXP10, "-",  2.0, "EXP10 (SNA||KM)"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 6),
                              subplot_kw=dict(polar=True))

    for ax, ds in zip(axes, DATASETS):
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(DIM_LABELS, fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"],
                            fontsize=7, color="#aaaaaa")
        ax.grid(color="#dddddd", linewidth=0.6)
        ax.set_title(DS_NAMES[ds], pad=20, fontsize=12, fontweight="bold")

        handles = []
        for exp_id, (color, ls, lw, label) in TARGET.items():
            row = dim_table[ds].get(exp_id)
            if row is None:
                print(f"[WARN] {ds} {exp_id} dimension 데이터 없음, 건너뜀")
                continue
            vals = [row[f"{d}_mean"] for d in DIMS]
            vals += vals[:1]
            ax.plot(angles, vals, color=color, linestyle=ls, linewidth=lw)
            ax.fill(angles, vals, color=color, alpha=0.07)
            handles.append(mpatches.Patch(color=color, label=label))

        ax.legend(handles=handles, loc="upper right",
                  bbox_to_anchor=(1.35, 1.15), fontsize=8, frameon=True)

    fig.suptitle("Dimension-level Matching Score: BASELINE vs Key Conditions "
                  "(GPT-4o-mini judge)", fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "radar_chart_combined.png")
    plt.savefig(out)
    plt.close()
    print(f"[OK] {out}")


# =================================================================
# 부가 Figure A — EXP별 3지표(Matching/Validity/TopicSim) 그룹 막대
# =================================================================
def fig_grouped_bar():
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    for ax, ds in zip(axes, DATASETS):
        exps = [e for e in EXP_ORDER if e in table[ds]]
        x = np.arange(len(exps))
        w = 0.25
        m_vals = [table[ds][e]["matching_mean"] for e in exps]
        m_err  = [table[ds][e]["matching_std"] for e in exps]
        v_vals = [table[ds][e]["validity_mean"] for e in exps]
        v_err  = [table[ds][e]["validity_std"] for e in exps]
        t_vals = [table[ds][e].get("topic_sim_mean", 0) for e in exps]
        t_err  = [table[ds][e].get("topic_sim_std", 0) for e in exps]

        ax.bar(x - w, m_vals, w, yerr=m_err, label="Matching",
               color=C_MATCH, capsize=2)
        ax.bar(x,     v_vals, w, yerr=v_err, label="Validity",
               color=C_VALID, capsize=2)
        ax.bar(x + w, t_vals, w, yerr=t_err, label="Topic Sim",
               color=C_TOPIC, capsize=2)

        ax.set_xticks(x)
        ax.set_xticklabels([EXP_SHORT.get(e, e) for e in exps],
                            fontsize=8, rotation=0)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{DS_NAMES[ds]} — GPT-4o-mini judge")
        ax.legend(loc="upper left", ncol=3)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig_grouped_bar.png")
    plt.savefig(out)
    plt.close()
    print(f"[OK] {out}")


# =================================================================
# 부가 Figure B — Cohen's d 효과크기 히트맵 (vs BASELINE, matching)
# =================================================================
def fig_heatmap_cohens_d():
    import math
    from scipy import stats as sstats

    def welch_d(m1, s1, n1, m2, s2, n2):
        sp = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
        return (m1 - m2) / sp if sp > 0 else 0.0

    exps = [e for e in EXP_ORDER if e != "BASELINE"]
    mat = np.zeros((len(exps), len(DATASETS)))
    for j, ds in enumerate(DATASETS):
        b = table[ds]["BASELINE"]
        for i, e in enumerate(exps):
            r = table[ds][e]
            mat[i, j] = welch_d(r["matching_mean"], r["matching_std"], r["matching_n"],
                                 b["matching_mean"], b["matching_std"], b["matching_n"])

    fig, ax = plt.subplots(figsize=(5, 8))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=mat.max())
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([DS_NAMES[d].split(" ")[0] for d in DATASETS])
    ax.set_yticks(range(len(exps)))
    ax.set_yticklabels([EXP_SHORT.get(e, e).replace("\n", " ") for e in exps], fontsize=8)
    for i in range(len(exps)):
        for j in range(len(DATASETS)):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if mat[i, j] < mat.max() * 0.6 else "white")
    ax.set_title("Cohen's $d$ vs.\\ BASELINE (Matching)\nGPT-4o-mini judge")
    fig.colorbar(im, ax=ax, label="Cohen's d")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig_heatmap_cohens_d.png")
    plt.savefig(out)
    plt.close()
    print(f"[OK] {out}")


# =================================================================
# 부가 Figure C — Complexity Paradox: 그룹평균 (single/pipeline/indep)
# =================================================================
def fig_complexity_paradox():
    GROUPS = {
        "Single":   ["EXP1_SNA", "EXP2_LDA", "EXP3_KMeans"],
        "Pipeline": ["EXP4_SNA_LDA", "EXP5_LDA_KMeans", "EXP6_SNA_KMeans", "EXP7_Full"],
        "Indep.":   ["EXP8_SNA_LDA_COMBO", "EXP9_LDA_KMeans_COMBO",
                      "EXP10_SNA_KMeans_COMBO", "EXP11_Full_COMBO"],
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, ds in zip(axes, DATASETS):
        means = [np.mean([table[ds][e]["matching_mean"] for e in ids])
                 for ids in GROUPS.values()]
        colors = [C_MATCH, C_EXP7, C_EXP10]
        bars = ax.bar(GROUPS.keys(), means, color=colors, width=0.5)
        for b, v in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", fontsize=10, fontweight="bold")
        ax.set_ylim(0, max(means) * 1.25)
        ax.set_title(f"{DS_NAMES[ds]}")
        ax.set_ylabel("Group-average Matching Score")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Complexity Paradox: Group-level Matching Score by Strategy\n"
                  "(GPT-4o-mini judge)", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig_complexity_paradox.png")
    plt.savefig(out)
    plt.close()
    print(f"[OK] {out}")


# =================================================================
# main
# =================================================================
if __name__ == "__main__":
    print(f"출력 폴더: {OUT_DIR}\n")
    fig_radar_combined()
    fig_grouped_bar()
    fig_heatmap_cohens_d()
    fig_complexity_paradox()
    print("\n완료. 논문에 삽입할 파일: radar_chart_combined.png (Figure 2 대체)")
    print("나머지는 부가 시각화(선택 사용)입니다.")
