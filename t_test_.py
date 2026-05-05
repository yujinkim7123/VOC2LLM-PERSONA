"""
=============================================================
ARR 논문용 t-test 유의성 검정 스크립트
=============================================================
[사용법]
  1. 아래 경로 설정을 본인 환경에 맞게 수정
  2. pip install scipy numpy (없으면 설치)
  3. python ttest_significance.py

[입력 파일]
  - final_table_3.json   : EXP1~7 + BASELINE 결과
  - EXP_RAW_stats.json   : EXP_RAW 결과

[출력 파일]
  - ttest_results.json   : 모든 t-test 결과 (JSON)
  - ttest_results.txt    : 콘솔 출력 텍스트 저장

[검정 내용]
  H1: BASELINE 대비 각 EXP 유의성 (Welch's t-test)
  H2: EXP3(KMeans) vs EXP_RAW 구조화 우월성
  H3: EXP3(KMeans) vs EXP7(Full) 복잡도 역설
=============================================================
"""

import json
import numpy as np
from scipy import stats
import sys
import os

# =============================================================
# ★ 경로 설정 — 본인 환경에 맞게 수정하세요
# =============================================================
FINAL_TABLE_PATH   = r""       # EXP1~7 + BASELINE
EXP_RAW_PATH       = r""       # EXP_RAW 결과
OUTPUT_JSON_PATH   = r""        # 결과 JSON 저장
OUTPUT_TXT_PATH    = r""         # 결과 텍스트 저장


# =============================================================
# 데이터 로드
# =============================================================
def load_data():
    with open(FINAL_TABLE_PATH, encoding="utf-8") as f:
        table3 = json.load(f)

    # EXP_RAW 파일이 있으면 로드, 없으면 수동 입력값 사용
    if os.path.exists(EXP_RAW_PATH):
        with open(EXP_RAW_PATH, encoding="utf-8") as f:
            raw_data = json.load(f)
        print("  ✅ EXP_RAW_stats.json 로드 성공")
    else:
        print("  ⚠️  EXP_RAW_stats.json 없음 → 수동 입력값 사용")
        # EXP_RAW_stats.json이 없을 경우 프로젝트에서 확인한 값 사용
        raw_data = [
            {
                "file_key": "voc_en", "exp_id": "EXP_RAW",
                "matching_mean": 0.5245, "matching_std": 0.052,  "matching_n": 10,
                "validity_mean": 0.775,  "validity_std": 0.0548, "validity_n": 10,
                "topic_sim_mean": 0.4272,"topic_sim_std": 0.0169,"topic_sim_n": 10,
            },
            {
                "file_key": "vad_en", "exp_id": "EXP_RAW",
                "matching_mean": 0.4205, "matching_std": 0.059,  "matching_n": 10,
                "validity_mean": 0.68,   "validity_std": 0.0589, "validity_n": 10,
                "topic_sim_mean": 0.4869,"topic_sim_std": 0.0211,"topic_sim_n": 10,
            },
        ]

    return table3 + raw_data


# =============================================================
# 데이터 구조화
# =============================================================
def build_dict(data):
    """
    {file_key: {exp_id: {metric: {mean, std, n}}}} 형태로 변환
    """
    d = {}
    for row in data:
        fk  = row["file_key"]
        eid = row["exp_id"]
        d.setdefault(fk, {})[eid] = {
            "matching":  {
                "mean": row["matching_mean"],
                "std":  row["matching_std"],
                "n":    row["matching_n"]
            },
            "validity":  {
                "mean": row["validity_mean"],
                "std":  row["validity_std"],
                "n":    row["validity_n"]
            },
            "topic_sim": {
                "mean": row["topic_sim_mean"],
                "std":  row["topic_sim_std"],
                "n":    row["topic_sim_n"]
            },
        }
    return d


# =============================================================
# Welch's t-test (평균/표준편차/n만으로 계산)
# =============================================================
def welch_ttest(m1, s1, n1, m2, s2, n2):
    """
    m1,s1,n1 = 비교 대상 EXP
    m2,s2,n2 = 기준 (BASELINE 또는 비교군)

    반환:
        t      : t 통계량
        p      : p-value (양측)
        cohens_d : 효과 크기
    """
    se = np.sqrt(s1**2 / n1 + s2**2 / n2)
    if se == 0:
        return 0.0, 1.0, 0.0

    t = (m1 - m2) / se

    # Welch-Satterthwaite 자유도
    numerator   = (s1**2/n1 + s2**2/n2) ** 2
    denominator = (s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1)
    df = numerator / denominator

    p = 2 * stats.t.sf(abs(t), df)  # 양측 검정

    # Cohen's d (pooled std)
    pooled_std = np.sqrt((s1**2 + s2**2) / 2)
    cohens_d   = (m1 - m2) / pooled_std if pooled_std > 0 else 0.0

    return round(t, 4), round(p, 6), round(cohens_d, 4)


def sig_marker(p):
    """p-value → 유의성 표시"""
    if   p < 0.001: return "***"
    elif p < 0.01:  return "**"
    elif p < 0.05:  return "*"
    else:           return "ns"


def effect_label(d):
    """Cohen's d → 효과 크기 레이블 (Cohen 1988)"""
    ad = abs(d)
    if   ad >= 0.8: return "large"
    elif ad >= 0.5: return "medium"
    elif ad >= 0.2: return "small"
    else:           return "negligible"


# =============================================================
# 메인 실행
# =============================================================
def main():
    all_data  = load_data()
    db        = build_dict(all_data)

    metrics   = ["matching", "validity", "topic_sim"]
    file_keys = ["voc_en", "vad_en"]
    EXP_ORDER = [
        "EXP1_SNA", "EXP2_LDA", "EXP3_KMeans",
        "EXP4_SNA_LDA", "EXP5_LDA_KMeans", "EXP6_SNA_KMeans",
        "EXP7_Full", "EXP_RAW"
    ]

    SEP   = "=" * 105
    lines = []   # 텍스트 저장용

    def pr(s=""):
        print(s)
        lines.append(s)

    # ─────────────────────────────────────────────
    # H1: BASELINE 대비 각 EXP
    # ─────────────────────────────────────────────
    h1_results = []
    pr(f"\n{SEP}")
    pr("  [H1] BASELINE 대비 각 EXP — Welch's t-test (양측검정, α=0.05)")
    pr("  * p<0.05  ** p<0.01  *** p<0.001  ns=not significant")
    pr(SEP)

    for fk in file_keys:
        if fk not in db:
            pr(f"  ⚠️  {fk} 데이터 없음, 스킵")
            continue

        baseline = db[fk].get("BASELINE")
        if baseline is None:
            pr(f"  ⚠️  {fk} BASELINE 없음, 스킵")
            continue

        pr(f"\n  Dataset: {fk}")
        pr(f"  {'EXP':<22} {'Metric':<12} {'EXP_mean':>9} {'BASE_mean':>9} "
           f"{'Δ':>7} {'t':>7} {'p':>10} {'sig':>4} {'Cohen_d':>8} {'Effect'}")
        pr(f"  {'─'*100}")

        for eid in EXP_ORDER:
            if eid not in db[fk]:
                continue
            for metric in metrics:
                em = db[fk][eid][metric]["mean"]
                es = db[fk][eid][metric]["std"]
                en = db[fk][eid][metric]["n"]
                bm = baseline[metric]["mean"]
                bs = baseline[metric]["std"]
                bn = baseline[metric]["n"]

                t, p, d = welch_ttest(em, es, en, bm, bs, bn)
                sig  = sig_marker(p)
                eff  = effect_label(d)
                delta = round(em - bm, 4)

                pr(f"  {eid:<22} {metric:<12} {em:>9.4f} {bm:>9.4f} "
                   f"{delta:>+7.4f} {t:>7.3f} {p:>10.5f} {sig:>4} {d:>8.4f} {eff}")

                h1_results.append({
                    "file_key": fk, "exp_id": eid, "metric": metric,
                    "exp_mean": em, "baseline_mean": bm, "delta": delta,
                    "t_stat": t, "p_value": p, "significance": sig,
                    "cohens_d": d, "effect_size": eff
                })

    # ─────────────────────────────────────────────
    # H2: EXP3 vs EXP_RAW
    # ─────────────────────────────────────────────
    h2_results = []
    pr(f"\n{SEP}")
    pr("  [H2] EXP3(KMeans) vs EXP_RAW — 구조화 우월성 검정")
    pr("  해석: EXP3 > EXP_RAW이면 '분석 구조화가 원문 주입보다 우수'")
    pr(SEP)

    for fk in file_keys:
        if "EXP3_KMeans" not in db.get(fk, {}) or "EXP_RAW" not in db.get(fk, {}):
            pr(f"  ⚠️  {fk} EXP3 또는 EXP_RAW 없음, 스킵")
            continue

        pr(f"\n  Dataset: {fk}")
        pr(f"  {'Metric':<12} {'EXP3':>9} {'EXP_RAW':>9} {'Δ(3-R)':>8} "
           f"{'t':>7} {'p':>10} {'sig':>4} {'Cohen_d':>8} {'Effect'}")
        pr(f"  {'─'*75}")

        for metric in metrics:
            m3 = db[fk]["EXP3_KMeans"][metric]["mean"]
            s3 = db[fk]["EXP3_KMeans"][metric]["std"]
            n3 = db[fk]["EXP3_KMeans"][metric]["n"]
            mr = db[fk]["EXP_RAW"][metric]["mean"]
            sr = db[fk]["EXP_RAW"][metric]["std"]
            nr = db[fk]["EXP_RAW"][metric]["n"]

            t, p, d = welch_ttest(m3, s3, n3, mr, sr, nr)
            sig   = sig_marker(p)
            eff   = effect_label(d)
            delta = round(m3 - mr, 4)

            pr(f"  {metric:<12} {m3:>9.4f} {mr:>9.4f} {delta:>+8.4f} "
               f"{t:>7.3f} {p:>10.5f} {sig:>4} {d:>8.4f} {eff}")

            h2_results.append({
                "file_key": fk, "metric": metric,
                "exp3_mean": m3, "raw_mean": mr, "delta": delta,
                "t_stat": t, "p_value": p, "significance": sig,
                "cohens_d": d, "effect_size": eff
            })

    # ─────────────────────────────────────────────
    # H3: EXP3 vs EXP7
    # ─────────────────────────────────────────────
    h3_results = []
    pr(f"\n{SEP}")
    pr("  [H3] EXP3(KMeans) vs EXP7(SNA+LDA+KMeans) — 복잡도 역설 검정")
    pr("  해석: EXP3 > EXP7이면 '단순 방법이 복합 방법보다 우수' (역설 지지)")
    pr(SEP)

    for fk in file_keys:
        if "EXP3_KMeans" not in db.get(fk, {}) or "EXP7_Full" not in db.get(fk, {}):
            pr(f"  ⚠️  {fk} EXP3 또는 EXP7 없음, 스킵")
            continue

        pr(f"\n  Dataset: {fk}")
        pr(f"  {'Metric':<12} {'EXP3':>9} {'EXP7':>9} {'Δ(3-7)':>8} "
           f"{'t':>7} {'p':>10} {'sig':>4} {'Cohen_d':>8} {'Effect'}")
        pr(f"  {'─'*75}")

        for metric in metrics:
            m3 = db[fk]["EXP3_KMeans"][metric]["mean"]
            s3 = db[fk]["EXP3_KMeans"][metric]["std"]
            n3 = db[fk]["EXP3_KMeans"][metric]["n"]
            m7 = db[fk]["EXP7_Full"][metric]["mean"]
            s7 = db[fk]["EXP7_Full"][metric]["std"]
            n7 = db[fk]["EXP7_Full"][metric]["n"]

            t, p, d = welch_ttest(m3, s3, n3, m7, s7, n7)
            sig   = sig_marker(p)
            eff   = effect_label(d)
            delta = round(m3 - m7, 4)

            pr(f"  {metric:<12} {m3:>9.4f} {m7:>9.4f} {delta:>+8.4f} "
               f"{t:>7.3f} {p:>10.5f} {sig:>4} {d:>8.4f} {eff}")

            h3_results.append({
                "file_key": fk, "metric": metric,
                "exp3_mean": m3, "exp7_mean": m7, "delta": delta,
                "t_stat": t, "p_value": p, "significance": sig,
                "cohens_d": d, "effect_size": eff
            })

    pr(f"\n{SEP}\n")

    # ─────────────────────────────────────────────
    # 결과 저장
    # ─────────────────────────────────────────────
    output = {
        "note": "Welch's two-sided t-test. Significance: * p<0.05, ** p<0.01, *** p<0.001, ns=not significant. Effect size (Cohen's d): negligible<0.2, small<0.5, medium<0.8, large>=0.8",
        "h1_baseline_comparison": h1_results,
        "h2_exp3_vs_raw":         h2_results,
        "h3_exp3_vs_exp7":        h3_results,
    }

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    pr(f"  ✅ JSON 저장: {OUTPUT_JSON_PATH}")

    with open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    pr(f"  ✅ TXT  저장: {OUTPUT_TXT_PATH}")


if __name__ == "__main__":
    main()