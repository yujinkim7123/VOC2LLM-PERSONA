"""
=================================================================
Re-Judge Only Pipeline — gpt-4o-mini 재채점 전용
=================================================================
[목적]
  이미 생성된 run_01~run_20의 페르소나를 건드리지 않고,
  채점(judge)만 gpt-4o-mini로 다시 수행합니다.
  생성(generation) 단계는 전혀 실행하지 않으므로 비용이 저렴합니다.

[원본과의 관계]
  unified_pipeline.py 의 judge 로직(_call_judge, evaluate_llm_judge,
  JUDGE_SYSTEM_PROMPT, 헝가리안 매칭)을 그대로 사용.
  변경점은 딱 두 가지:
    1) JUDGE_MODEL = "gpt-4o-mini"
    2) 결과를 기존 evaluation_results 를 덮어쓰지 않고
       evaluation_results_mini/ 별도 폴더에 저장

[사용법]
  1) 아래 OPENAI_API_KEY 환경변수를 새로 발급받은 키로 설정
     (코드에 직접 쓰지 말고 환경변수 권장!)
       Windows PowerShell:  $env:OPENAI_API_KEY="sk-..."
       cmd:                 set OPENAI_API_KEY=sk-...
  2) BASE_DIR / GT_DIR 경로 확인
  3) python rejudge_mini.py                    # 전체 run 재채점
     python rejudge_mini.py --start 1 --end 3  # run 1~3만 (파일럿)
     python rejudge_mini.py --stats-only       # 통계만 재계산

[비용 참고]
  gpt-4o-mini는 gpt-4o 대비 입력 ~33배, 출력 ~25배 저렴.
  13조건 x 2데이터셋 x 16쌍 x 20run = 8,320 호출이지만
  호출당 토큰이 작아 (400 max) 전체 수 달러 수준 예상.
  파일럿으로 --start 1 --end 3 (3 run) 먼저 돌려 확인 권장.
=================================================================
"""

import os, json, re, time, math, warnings
import argparse
warnings.filterwarnings("ignore")

# =================================================================
# ★★★ 여기에 API 키를 붙여넣으세요 ★★★
#   platform.openai.com → API keys 에서 새로 발급받은 키
#   주의: 이 파일은 절대 GitHub/공개 저장소에 올리지 마세요!
# =================================================================
OPENAI_API_KEY = ""

if OPENAI_API_KEY.startswith("sk-"):
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
elif not os.environ.get("OPENAI_API_KEY"):
    raise SystemExit(
        "\n[설정 필요] 스크립트 상단의 OPENAI_API_KEY 변수에\n"
        "발급받은 키(sk-로 시작)를 붙여넣고 다시 실행하세요.\n")

# =================================================================
# 0. 경로 설정 — 기존 unified_pipeline.py와 동일하게 맞춰주세요
# =================================================================
BASE_DIR = r"C:\Users\User\OneDrive\바탕 화면\대학원\캡스톤\runs"
GT_DIR   = r"C:\Users\User\OneDrive\바탕 화면\대학원\캡스톤\data_precess"

# =================================================================
# ★ 재채점 설정
# =================================================================
START_RUN   = 1
END_RUN     = 20
N_RUNS      = 20

JUDGE_MODEL = "gpt-4o-mini"     # ★ 유일한 모델 변경점
API_SLEEP   = 0.5               # mini는 rate limit 여유가 커서 단축
JUDGE_THRESHOLD = 0.5

EVAL_SUBDIR = "evaluation_results_mini"   # 기존 결과 보호: 별도 폴더
STATS_NAME  = "final_table3_mini.json"    # 기존 통계 보호: 별도 파일

# =================================================================
# 1. 설정 (원본과 동일)
# =================================================================
FILE_CONFIG = {
    "voc_en": {
        "lang":            "en",
        "domain":          "Home Appliance Products (English reviews, synthetic VOC)",
        "gt_persona_path": os.path.join(GT_DIR, "kr_appliance_personas_en.json"),
    },
    "vad_en": {
        "lang":            "en",
        "domain":          "Virtual Assistant Devices — smart speakers and displays (English reviews)",
        "gt_persona_path": os.path.join(GT_DIR, "vad_consumer_personas.json"),
    },
}
DEFAULT_FILE_KEYS = ["voc_en", "vad_en"]

EXP_IDS = [
    "BASELINE",
    "EXP1_SNA", "EXP2_LDA", "EXP3_KMeans",
    "EXP4_SNA_LDA", "EXP5_LDA_KMeans", "EXP6_SNA_KMeans", "EXP7_Full",
    "EXP8_SNA_LDA_COMBO", "EXP9_LDA_KMeans_COMBO",
    "EXP10_SNA_KMeans_COMBO", "EXP11_Full_COMBO",
    "EXP_RAW",
]

def run_persona_dir(run_id):
    return os.path.join(BASE_DIR, f"run_{run_id:02d}", "persona_results")

def run_eval_dir(run_id):
    return os.path.join(BASE_DIR, f"run_{run_id:02d}", EVAL_SUBDIR)

def stats_dir():
    return os.path.join(BASE_DIR, "statistics")

def load_gt_persona_from_path(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("personas", data.get("generated_personas", []))
    return data

# =================================================================
# 2. Judge (원본 unified_pipeline.py에서 그대로 가져옴)
# =================================================================
JUDGE_SYSTEM_PROMPT = """You are an expert consumer persona evaluator.

Your task:
  Compare a GENERATED consumer persona against a REFERENCE (ground truth) persona.
  Score how well the generated persona matches the reference across 5 dimensions.
  Also judge validity — how well the generated persona is grounded in the DOMAIN TOPICS provided.

Important context about sentiment_ratio:
  The generated personas were produced from analysis data that contains NO explicit
  sentiment labels. The sentiment_ratio was inferred by the LLM from the semantic
  valence of topic keywords (e.g., '만족/satisfied' → positive signal).
  When scoring sentiment_alignment, evaluate whether the DIRECTION and ROUGH PROPORTION
  of inferred sentiment is reasonable given the keywords — do not penalize for small
  numeric differences if the overall emotional tone is correctly captured.

Scoring guide (0.0 ~ 1.0):
  1.0 = almost identical in meaning or direction
  0.7 = mostly aligned, minor differences
  0.5 = partially aligned, some overlap
  0.3 = weak alignment, mostly different
  0.0 = completely different, no meaningful overlap

5 matching dimensions:
  keyword_alignment    : Do generated keywords cover the same topics as reference keywords?
  value_alignment      : Does the generated core_value align with the reference core_value?
  pain_alignment       : Do generated pain_points reflect the same frustrations as reference?
  behavioral_alignment : Does generated description+motivation match reference consumer behavior?
  sentiment_alignment  : Does the inferred sentiment direction and rough proportion match reference?
                         (allow tolerance for inference error — focus on Pos/Neg direction, not exact decimals)

matching_score : average of the 5 dimensions above

validity_score : How well is the GENERATED persona grounded in the DOMAIN TOPICS listed?
  Judge whether the persona's keywords, aspects, and concerns are traceable
  to the provided domain topic list — NOT general consumer knowledge.
  1.0 = all content clearly grounded in the domain topics provided
  0.7 = most content grounded, minor unsupported details
  0.5 = partially grounded, some content not traceable to domain topics
  0.3 = weakly grounded, most content is generic or off-domain
  0.0 = completely fabricated, no connection to the domain topics provided

Return ONLY valid JSON. No text outside the JSON.

Output format:
{
  "keyword_alignment":    <float 0.0-1.0>,
  "value_alignment":      <float 0.0-1.0>,
  "pain_alignment":       <float 0.0-1.0>,
  "behavioral_alignment": <float 0.0-1.0>,
  "sentiment_alignment":  <float 0.0-1.0>,
  "matching_score":       <average of 5 dimensions, float 0.0-1.0>,
  "validity_score":       <float 0.0-1.0>,
  "reasoning":            "<max 80 chars>"
}"""


def _fmt_persona(p, label):
    sr  = p.get("sentiment_ratio", {})
    pos = sr.get("Positive", sr.get("긍정", 0))
    neu = sr.get("Neutral",  sr.get("중립", 0))
    neg = sr.get("Negative", sr.get("부정", 0))
    lines = [
        "## " + label,
        "- name: "        + p.get("persona_name", ""),
        "- keywords: "    + ", ".join(p.get("keywords", [])),
        "- core_value: "  + p.get("core_value", ""),
        "- description: " + p.get("description", ""),
        "- motivation: "  + p.get("motivation", ""),
        "- pain_points: " + p.get("pain_points", ""),
        "- sentiment_ratio: Pos=" + str(round(pos, 2)) +
        " Neu=" + str(round(neu, 2)) + " Neg=" + str(round(neg, 2)),
    ]
    return "\n".join(lines)


def _fmt_domain_topics(gt_personas):
    topics = set()
    for gtp in gt_personas:
        topics.update(gtp.get("keywords", []))
        topics.update(gtp.get("top_aspects", gtp.get("top_topics", [])))
    return "## DOMAIN TOPICS (validity grounding criteria)\n- " + "\n- ".join(sorted(topics))


def _call_judge(gen_p, gt_p, all_gt_personas=None):
    try:
        from openai import OpenAI
        client = OpenAI()
    except Exception as e:
        print("    [ERROR] OpenAI init failed:", e)
        return None
    domain_block = ("\n\n" + _fmt_domain_topics(all_gt_personas)) if all_gt_personas else ""
    user_prompt  = (_fmt_persona(gt_p, "REFERENCE (Ground Truth)")
                    + "\n\n" + _fmt_persona(gen_p, "GENERATED") + domain_block)
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL, max_tokens=400,          # ★ gpt-4o-mini
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                      {"role": "user",   "content": user_prompt}],
        )
        raw    = resp.choices[0].message.content.strip()
        raw    = re.sub(r"```json\s*", "", raw)
        raw    = re.sub(r"```\s*", "", raw)
        parsed = json.loads(raw)
        DIMS   = ["keyword_alignment", "value_alignment", "pain_alignment",
                  "behavioral_alignment", "sentiment_alignment"]
        for d in DIMS:
            parsed.setdefault(d, 0.0)
            parsed[d] = round(max(0.0, min(1.0, float(parsed[d]))), 4)
        parsed["matching_score"] = round(sum(parsed[d] for d in DIMS) / len(DIMS), 4)
        parsed["validity_score"] = round(max(0.0, min(1.0, float(parsed.get("validity_score", 0.5)))), 4)
        return parsed
    except Exception as e:
        print("    [WARN] Judge failed:", e)
        return None


def evaluate_llm_judge(gen_personas, gt_personas):
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    DIMS = ["keyword_alignment", "value_alignment", "pain_alignment",
            "behavioral_alignment", "sentiment_alignment"]
    judge_matrix, detail_matrix = [], []
    for i, gen_p in enumerate(gen_personas):
        row_s, row_d = [], []
        for j, gt_p in enumerate(gt_personas):
            print(f"      P{i+1} vs GT_P{j+1}...", end=" ", flush=True)
            result = _call_judge(gen_p, gt_p, all_gt_personas=gt_personas)
            if result is None:
                result = {d: 0.0 for d in DIMS}
                result.update({"matching_score": 0.0, "validity_score": 0.0, "reasoning": "failed"})
            print(f"match={result['matching_score']:.2f} valid={result['validity_score']:.2f}")
            row_s.append(result["matching_score"])
            row_d.append(result)
            time.sleep(API_SLEEP)
        judge_matrix.append(row_s)
        detail_matrix.append(row_d)
    mat = np.array(judge_matrix)
    row_ind, col_ind = linear_sum_assignment(-mat)
    best_mapping = []
    for r, c in zip(row_ind, col_ind):
        d = detail_matrix[r][c]
        best_mapping.append({
            "gen_name": gen_personas[r].get("persona_name", ""),
            "gt_name":  gt_personas[c].get("persona_name", ""),
            **{k: d.get(k, 0.0) for k in DIMS + ["matching_score", "validity_score", "reasoning"]},
            "success":  d.get("matching_score", 0.0) >= JUDGE_THRESHOLD,
        })
    matching_vals = [detail_matrix[r][c]["matching_score"] for r, c in zip(row_ind, col_ind)]
    validity_vals = [detail_matrix[r][c]["validity_score"] for r, c in zip(row_ind, col_ind)]
    dim_avgs = {dim: round(float(np.mean([detail_matrix[r][c].get(dim, 0.0)
                for r, c in zip(row_ind, col_ind)])), 4) for dim in DIMS}
    n_success = sum(1 for m in best_mapping if m["success"])
    return {
        "judge_model":          JUDGE_MODEL,
        "judge_matrix":         judge_matrix,
        "best_mapping":         best_mapping,
        "matching_score":       round(float(np.mean(matching_vals)), 4),
        "validity_score":       round(float(np.mean(validity_vals)), 4),
        "mean_all_matching":    round(float(mat.mean()), 4),
        "dim_averages":         dim_avgs,
        "n_success":            n_success,
        "mapping_success_rate": round(n_success / max(len(gt_personas), 1), 4),
    }

# =================================================================
# 3. 채점 실행 (생성 단계 없음 — 기존 persona_results만 읽음)
# =================================================================
def rejudge_single(file_key, exp_id, persona_dir, eval_out_dir, gt_personas):
    path = os.path.join(persona_dir, file_key, f"{exp_id}_persona.json")
    if not os.path.exists(path):
        print(f"    [SKIP] 생성 결과 없음: {path}")
        return None
    with open(path, encoding="utf-8") as f:
        generated = json.load(f)
    gen_personas = generated.get("generated_personas", [])
    if not gen_personas or not gt_personas:
        return None
    print(f"    [Judge-mini] [{file_key}] {exp_id}")
    judge = evaluate_llm_judge(gen_personas, gt_personas)
    print(f"      matching={judge['matching_score']} validity={judge['validity_score']}")
    result = {
        "file_key": file_key, "exp_id": exp_id,
        "judge_model": JUDGE_MODEL,
        "judge": judge,
        "summary": {
            "matching_score": judge["matching_score"],
            "validity_score": judge["validity_score"],
            "dim_averages":   judge["dim_averages"],
        },
    }
    dir_ = os.path.join(eval_out_dir, file_key)
    os.makedirs(dir_, exist_ok=True)
    with open(os.path.join(dir_, f"{exp_id}_eval.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def run_one(run_id, file_keys, exp_ids):
    persona_dir = run_persona_dir(run_id)
    eval_dir    = run_eval_dir(run_id)
    if not os.path.isdir(persona_dir):
        print(f"[SKIP] run_{run_id:02d}: persona_results 폴더 없음")
        return
    os.makedirs(eval_dir, exist_ok=True)
    print(f"\n===== run_{run_id:02d} 재채점 (judge={JUDGE_MODEL}) =====")
    for file_key in file_keys:
        cfg = FILE_CONFIG[file_key]
        gt_personas = load_gt_persona_from_path(cfg["gt_persona_path"])
        for exp_id in exp_ids:
            rejudge_single(file_key, exp_id, persona_dir, eval_dir, gt_personas)

# =================================================================
# 4. 통계 (matching / validity만 — topic_sim은 judge 무관이라 재계산 불필요)
# =================================================================
def compute_statistics(file_keys, exp_ids):
    import numpy as np
    rows = []
    for file_key in file_keys:
        for exp_id in exp_ids:
            m_vals, v_vals = [], []
            for run_id in range(1, N_RUNS + 1):
                p = os.path.join(run_eval_dir(run_id), file_key, f"{exp_id}_eval.json")
                if not os.path.exists(p):
                    continue
                with open(p, encoding="utf-8") as f:
                    r = json.load(f)
                s = r.get("summary", {})
                if "matching_score" in s:
                    m_vals.append(s["matching_score"])
                if "validity_score" in s:
                    v_vals.append(s["validity_score"])
            if not m_vals:
                continue
            rows.append({
                "file_key": file_key, "exp_id": exp_id,
                "judge_model": JUDGE_MODEL,
                "matching_mean": round(float(np.mean(m_vals)), 4),
                "matching_std":  round(float(np.std(m_vals, ddof=1)), 4) if len(m_vals) > 1 else 0.0,
                "matching_n":    len(m_vals),
                "validity_mean": round(float(np.mean(v_vals)), 4),
                "validity_std":  round(float(np.std(v_vals, ddof=1)), 4) if len(v_vals) > 1 else 0.0,
                "validity_n":    len(v_vals),
            })
    os.makedirs(stats_dir(), exist_ok=True)
    out = os.path.join(stats_dir(), STATS_NAME)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\n[통계 저장] {out}")

    # 기존 gpt-4o(?) 채점 통계와 즉석 비교 (있으면)
    orig = os.path.join(stats_dir(), "final_table3.json")
    if os.path.exists(orig):
        with open(orig, encoding="utf-8") as f:
            orig_rows = json.load(f)
        om = {(r["file_key"], r["exp_id"]): r for r in orig_rows}
        print("\n===== 기존 채점 vs mini 재채점 비교 (matching_mean) =====")
        print(f"{'dataset':<8}{'exp':<26}{'orig':>8}{'mini':>8}{'diff':>8}")
        diffs = []
        for r in rows:
            o = om.get((r["file_key"], r["exp_id"]))
            if o:
                d = r["matching_mean"] - o["matching_mean"]
                diffs.append(d)
                print(f"{r['file_key']:<8}{r['exp_id']:<26}"
                      f"{o['matching_mean']:>8.3f}{r['matching_mean']:>8.3f}{d:>+8.3f}")
        if diffs:
            import numpy as np
            print(f"\n평균 차이: {np.mean(diffs):+.4f}, 절대평균: {np.mean(np.abs(diffs)):.4f}")
            # 순위 상관 (조건별 순위가 유지되는지)
            try:
                from scipy.stats import spearmanr
                for fk in file_keys:
                    o_list = [om[(fk, e)]["matching_mean"] for e in exp_ids
                              if (fk, e) in om and any(r["file_key"] == fk and r["exp_id"] == e for r in rows)]
                    n_list = [next(r["matching_mean"] for r in rows
                              if r["file_key"] == fk and r["exp_id"] == e)
                              for e in exp_ids
                              if (fk, e) in om and any(r["file_key"] == fk and r["exp_id"] == e for r in rows)]
                    if len(o_list) > 2:
                        rho, p = spearmanr(o_list, n_list)
                        print(f"Spearman rank corr [{fk}]: rho={rho:.3f} (p={p:.4f})")
            except ImportError:
                pass

# =================================================================
# 5. main
# =================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=START_RUN)
    parser.add_argument("--end",   type=int, default=END_RUN)
    parser.add_argument("--files", nargs="+", choices=DEFAULT_FILE_KEYS, default=DEFAULT_FILE_KEYS)
    parser.add_argument("--exps",  nargs="+", choices=EXP_IDS, default=EXP_IDS)
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()

    if not args.stats_only:
        for run_id in range(args.start, args.end + 1):
            run_one(run_id, args.files, args.exps)

    compute_statistics(args.files, args.exps)
    print("\n완료. 기존 evaluation_results는 건드리지 않았습니다.")