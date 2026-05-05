"""
=================================================================
EXP_RAW 단독 실험 파이프라인
=================================================================
[실험 목적]
  분석(SNA/LDA/KMeans) 없이 리뷰 원문을 그대로 GPT에 주입해서
  페르소나를 생성하는 실험.

  비교 조건:
    BASELINE  : 아무 데이터도 안 줌 (GPT 혼자 생성)
    EXP_RAW   : 리뷰 원문 전체를 그대로 주입  ← 이 스크립트
    EXP1~EXP7 : SNA/LDA/KMeans 분석 결과 주입

[데이터]
  voc_en : merged_voc.csv  → review 컬럼 (영어 리뷰, 965행)
  vad_en : Virtual_Assistant_Devices_Dataset__Final_Version.xlsx
           → Unnamed: 4 컬럼 (영어 리뷰, 2370행)

[실행 방법]
  python exp_raw_pipeline.py                  # voc_en + vad_en 전체 실험
  python exp_raw_pipeline.py --files voc_en   # voc_en만
  python exp_raw_pipeline.py --files vad_en   # vad_en만
  python exp_raw_pipeline.py --runs 3         # 3회 반복

[출력]
  {BASE_DIR}/run_01/persona_results/{file_key}/EXP_RAW_persona.json
  {BASE_DIR}/run_01/evaluation_results/{file_key}/EXP_RAW_eval.json
  {BASE_DIR}/statistics/EXP_RAW_stats.json
=================================================================
"""

import os, json, re, time, math, random, warnings
import pandas as pd
warnings.filterwarnings("ignore")

# ★ OpenAI API 키 — 본인 키로 교체하세요
os.environ["OPENAI_API_KEY"] = ""

# =================================================================
# 0. 경로 설정 ← 본인 환경에 맞게 수정하세요
# =================================================================
BASE_DIR = r""
GT_DIR   = r""

# 두 데이터 파일 경로
VOC_CSV_PATH  = r""
VAD_XLSX_PATH = r""

os.makedirs(BASE_DIR, exist_ok=True)

# =================================================================
# 1. 실험 설정
# =================================================================
N_RUNS         = 10      # 반복 횟수 (통계 안정성용)
GPT_MODEL      = "gpt-4o"
API_SLEEP      = 1.0
JUDGE_THRESHOLD = 0.5

# 한 번에 GPT에 넣을 최대 리뷰 수
# (너무 많으면 토큰 초과 → 4o 기준 200개 정도가 안전)
MAX_REVIEWS_VOC = 200
MAX_REVIEWS_VAD = 200

RANDOM_SEED = 42

# =================================================================
# 2. 파일별 설정
# =================================================================
FILE_CONFIG = {
    "voc_en": {
        "lang":            "en",
        "domain":          "Home Appliance Products (English reviews, synthetic VOC)",
        "gt_persona_path": os.path.join(GT_DIR, "kr_appliance_personas_en.json"),
        "review_loader":   "voc",   # load_reviews() 에서 분기용
    },
    "vad_en": {
        "lang":            "en",
        "domain":          "Virtual Assistant Devices — smart speakers and displays (English reviews)",
        "gt_persona_path": os.path.join(GT_DIR, "vad_consumer_personas.json"),
        "review_loader":   "vad",
    },
}

DEFAULT_FILE_KEYS = ["voc_en", "vad_en"]

# =================================================================
# 3. 리뷰 로더
# =================================================================
def load_reviews(file_key, max_reviews=200, seed=RANDOM_SEED):
    """
    파일에서 리뷰 원문을 읽어서 텍스트 블록으로 반환.

    [왜 샘플링하나?]
    GPT-4o 의 컨텍스트 창은 128K 토큰이지만,
    리뷰 965개를 전부 넣으면 약 15,000~20,000 토큰 → 비용 폭발 + 처리 지연.
    200개 샘플링으로도 충분히 다양한 소비자 패턴이 커버됨.
    논문에는 "stratified random sampling (n=200, seed=42)" 으로 명시 가능.

    [VOC] merged_voc.csv → review 컬럼 (영어 리뷰)
    [VAD] xlsx → Unnamed: 4 컬럼 (실제 리뷰 텍스트)
    """
    cfg = FILE_CONFIG[file_key]
    loader_type = cfg["review_loader"]

    if loader_type == "voc":
        # ----- VOC -----
        df = pd.read_csv(VOC_CSV_PATH, encoding="utf-8-sig")
        reviews = df["review"].dropna().tolist()

        # segment별 균등 샘플링 (층화 샘플링) — 각 S1~S4에서 고르게 추출
        # 이렇게 해야 특정 페르소나에 편향되지 않음
        df_clean = df[df["review"].notna()].copy()
        segments = df_clean["segment"].unique().tolist()
        per_seg  = max(1, max_reviews // len(segments))

        sampled = []
        rng = random.Random(seed)
        for seg in segments:
            seg_reviews = df_clean[df_clean["segment"] == seg]["review"].tolist()
            sample_size = min(per_seg, len(seg_reviews))
            sampled.extend(rng.sample(seg_reviews, sample_size))

        # 총 max_reviews 초과 시 다시 샘플링
        if len(sampled) > max_reviews:
            rng2 = random.Random(seed)
            sampled = rng2.sample(sampled, max_reviews)

        print(f"    [VOC] 층화 샘플링: {len(sampled)}개 (전체 {len(reviews)}개에서)")
        return sampled

    elif loader_type == "vad":
        # ----- VAD -----
        df = pd.read_excel(VAD_XLSX_PATH)
        # 실제 리뷰 컬럼은 'Unnamed: 4' (파일 구조 확인 완료)
        reviews = df["Unnamed: 4"].dropna().tolist()
        reviews = [str(r).strip() for r in reviews if str(r).strip()]

        # VAD는 segment 없으므로 단순 랜덤 샘플링
        rng = random.Random(seed)
        if len(reviews) > max_reviews:
            reviews = rng.sample(reviews, max_reviews)

        print(f"    [VAD] 랜덤 샘플링: {len(reviews)}개 (전체 2370개에서)")
        return reviews

    return []


def reviews_to_block(reviews):
    """
    리뷰 리스트 → GPT 프롬프트용 텍스트 블록 변환.
    번호를 붙여서 GPT가 각 리뷰를 구분할 수 있게 함.
    """
    return "\n".join([f"[{i+1}] {r}" for i, r in enumerate(reviews)])


# =================================================================
# 4. GPT 페르소나 생성
# =================================================================

# 출력 스키마 (기존 파이프라인과 동일 → 평가 코드 재사용 가능)
PERSONA_OUTPUT_SCHEMA = """{
  "personas": [
    {
      "persona_id": "P1",
      "persona_name": "<소비자 유형 이름 — 리뷰에서 관찰된 패턴을 반영>",
      "category": "<제품 카테고리>",
      "data_source": "<file_key>_EXP_RAW",
      "keywords": ["<리뷰에서 추출한 5-7개 핵심 키워드>"],
      "core_value": "<이 소비자 세그먼트가 가장 중시하는 가치, 1문장>",
      "description": "<누구이며 어떻게 행동하는지, 2-3문장>",
      "motivation": "<구매 결정을 이끄는 동기, 1-2문장>",
      "pain_points": "<불만족 요소, 1-2문장>",
      "quote": "<이 페르소나가 실제로 쓸 법한 리뷰 인용문, 1문장>",
      "sentiment_ratio": {
        "Positive": 0.0,
        "Neutral": 0.0,
        "Negative": 0.0,
        "_inference_note": "<각 감성 비율을 어떤 리뷰 패턴에서 추론했는지 간단히 설명>"
      },
      "top_aspects": ["<리뷰에서 자주 등장한 5개 주요 측면>"],
      "co_aspects": {"<측면>": ["<함께 언급된 2-3개 측면>"]}
    }
  ]
}"""


SYSTEM_PROMPT = """\
You are a senior consumer insights researcher with 15+ years of experience
synthesizing consumer reviews into vivid, evidence-grounded consumer personas.

## Your Role
You will receive a set of raw consumer reviews (no pre-processing applied).
Your job is to:
  1. Read through all reviews carefully
  2. Identify 4 distinct consumer segments from the patterns you observe
  3. Synthesize each segment into a detailed, realistic consumer persona

## How to Identify Consumer Segments from Raw Reviews

### Step 1 — Cluster by behavioral pattern
  Look for reviews that share similar:
  - Topics (what they talk about: price, performance, design, reliability...)
  - Emotional tone (satisfied, frustrated, neutral)
  - Purchase motivation (deal-seeking, quality-seeking, design-seeking...)
  - Pain points (what they complain about)

### Step 2 — Infer sentiment_ratio from review language
  For each persona's review cluster, estimate:
  - Positive signals: satisfaction words (love, great, amazing, recommend...)
  - Negative signals: complaint words (broken, disappointed, return, issue...)
  - Neutral signals: factual description (used, purchased, okay, decent...)
  Record your reasoning in "_inference_note".
  Values must sum to exactly 1.0.

### Step 3 — Ground every field in actual review content
  keywords      → words/phrases that ACTUALLY appear in the reviews
  top_aspects   → topics that are ACTUALLY mentioned in the reviews
  co_aspects    → aspects that ACTUALLY co-occur in the same reviews
  persona_name  → describe the observed pattern (e.g. "Value Hunter", "Reliability Seeker")
  pain_points   → infer from negative/complaint language in the reviews
  motivation    → infer from what they praise or seek in their reviews
  quote         → select or lightly adapt a representative phrase from the reviews

### Step 4 — Ensure 4 clearly distinct segments
  Each persona must represent a meaningfully different consumer type.
  Avoid near-duplicates. Look for:
  - Different dominant concerns (price vs. quality vs. design vs. service)
  - Different emotional tones
  - Different product usage contexts

## Output Rules
  - Return ONLY valid JSON matching the schema below
  - sentiment_ratio values must sum to exactly 1.0
  - Include _inference_note for every persona
  - Do NOT fabricate details not supported by the reviews

Output format:
""" + PERSONA_OUTPUT_SCHEMA


def build_raw_user_prompt(file_key, reviews):
    """
    리뷰 원문을 그대로 GPT 프롬프트에 주입하는 유저 프롬프트 생성.

    [기존 BASELINE과의 차이]
    BASELINE: "이 도메인 아는 거 다 써서 페르소나 만들어"
    EXP_RAW:  "이 리뷰들 읽고 페르소나 만들어"  ← 실제 데이터 주입

    [기존 EXP1~7과의 차이]
    EXP1~7:   SNA/LDA/KMeans로 이미 분석된 결과(토픽 키워드, 군집 구조)를 주입
    EXP_RAW:  분석 없이 리뷰 원문 그대로 주입
    """
    cfg    = FILE_CONFIG[file_key]
    domain = cfg["domain"]
    review_block = reviews_to_block(reviews)
    n = len(reviews)

    return f"""\
Domain: {domain}

## Raw Consumer Reviews (n={n})
Below are actual consumer reviews collected for products in this domain.
These are unprocessed, raw review texts — no topic modeling or clustering has been applied.

{review_block}

## Your Task
1. Read all {n} reviews above carefully
2. Identify 4 distinct consumer segments based on behavioral patterns you observe
3. Synthesize each segment into a detailed consumer persona

Important notes:
  - keywords and top_aspects MUST come from actual words/phrases in the reviews above
  - co_aspects MUST reflect aspects that actually co-occur in the same reviews
  - sentiment_ratio MUST be inferred from the emotional tone of the reviews
  - persona_name MUST describe the observed pattern, not a generic label
  - Set data_source to "{file_key}_EXP_RAW" for all personas

Return ONLY valid JSON. No explanation text outside the JSON."""


def call_gpt_gen(user_prompt, max_retries=2):
    """GPT API 호출 (기존 파이프라인과 동일한 방식)"""
    try:
        from openai import OpenAI
        client = OpenAI()
    except Exception as e:
        print(f"    [ERROR] OpenAI 초기화 실패: {e}")
        return None

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=GPT_MODEL,
                max_tokens=2000,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"```json\s*", "", raw)
            raw = re.sub(r"```\s*",     "", raw)
            parsed = json.loads(raw)
            if "personas" not in parsed or not parsed["personas"]:
                print(f"    [WARN] personas 키 없음 (attempt {attempt+1})")
                continue
            return parsed
        except Exception as e:
            print(f"    [WARN] 생성 실패 (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(API_SLEEP * 2)
    return None


def validate_persona(persona, idx, file_key):
    """페르소나 필드 검증 및 기본값 채우기 (기존 파이프라인과 동일)"""
    exp_id = "EXP_RAW"
    pid    = f"P{idx+1}"
    persona.setdefault("persona_id",   pid)
    persona.setdefault("persona_name", f"Persona {pid}")
    persona.setdefault("category",     file_key)
    persona.setdefault("data_source",  f"{file_key}_{exp_id}")

    if not isinstance(persona.get("keywords"), list):
        persona["keywords"] = []

    for field in ["core_value", "description", "motivation", "pain_points", "quote"]:
        persona.setdefault(field, "")

    # sentiment_ratio 정규화
    sr_raw = persona.get("sentiment_ratio", {})
    inference_note = sr_raw.pop("_inference_note", "")
    key_map = {"긍정": "Positive", "중립": "Neutral", "부정": "Negative"}
    sr = {key_map.get(k, k): v for k, v in sr_raw.items() if isinstance(v, (int, float))}
    for k in ["Positive", "Neutral", "Negative"]:
        sr.setdefault(k, 0.0)
    total = sum(sr.values())
    if total > 0:
        sr = {k: round(v / total, 4) for k, v in sr.items()}
    else:
        sr = {"Positive": 0.40, "Neutral": 0.40, "Negative": 0.20}
        inference_note = inference_note or "fallback — no numeric values returned by LLM"
    sr["_inference_note"] = inference_note
    persona["sentiment_ratio"] = sr

    if not isinstance(persona.get("top_aspects"), list):
        persona["top_aspects"] = persona.get("keywords", [])[:5]
    if not isinstance(persona.get("co_aspects"), dict):
        persona["co_aspects"] = {}

    return persona


def generate_raw_persona(file_key, persona_out_dir, run_seed):
    """
    리뷰 원문 주입 → GPT 페르소나 생성 → JSON 저장

    run_seed: 매 run마다 다른 시드로 샘플링 → 샘플링 분산도 실험에 반영
    """
    cfg    = FILE_CONFIG[file_key]
    domain = cfg["domain"]

    # 리뷰 로드 (run마다 다른 seed로 다양한 샘플)
    reviews = load_reviews(file_key, max_reviews=MAX_REVIEWS_VOC, seed=run_seed)
    if not reviews:
        print(f"    [ERROR] 리뷰 로드 실패: {file_key}")
        return None

    user_prompt = build_raw_user_prompt(file_key, reviews)

    print(f"    GPT 호출 중... [{file_key}] EXP_RAW ({len(reviews)}개 리뷰 주입)")
    response = call_gpt_gen(user_prompt)
    if response is None:
        print(f"    [SKIP] 생성 실패")
        return None

    personas = [
        validate_persona(p, i, file_key)
        for i, p in enumerate(response.get("personas", []))
    ]

    result = {
        "file_key":          file_key,
        "exp_id":            "EXP_RAW",
        "method":            "RAW_REVIEW_INJECTION",
        "lang":              cfg["lang"],
        "domain":            domain,
        "n_reviews_injected": len(reviews),
        "run_seed":          run_seed,
        "generated_personas": personas,
    }

    out_dir = os.path.join(persona_out_dir, file_key)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "EXP_RAW_persona.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"    저장: {out_path}")
    return result


# =================================================================
# 5. 평가 (기존 파이프라인 evaluate_single 과 동일 로직)
# =================================================================

_embed_cache = {}

def get_embedding(text):
    if not text or not text.strip():
        return None
    key = text.strip()
    if key in _embed_cache:
        return _embed_cache[key]
    try:
        from openai import OpenAI
        client = OpenAI()
        resp   = client.embeddings.create(model="text-embedding-3-small", input=key)
        vec    = resp.data[0].embedding
        _embed_cache[key] = vec
        return vec
    except Exception as e:
        print(f"  [WARN] 임베딩 실패: {e}")
        return None


def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


JUDGE_SYSTEM_PROMPT = """\
You are an expert consumer persona evaluator.

Compare a GENERATED consumer persona against a REFERENCE (ground truth) persona.
Score alignment across 5 dimensions (0.0~1.0 each).

Scoring guide:
  1.0 = almost identical in meaning or direction
  0.7 = mostly aligned, minor differences
  0.5 = partially aligned, some overlap
  0.3 = weak alignment, mostly different
  0.0 = completely different

5 matching dimensions:
  keyword_alignment    : Do generated keywords cover the same topics as reference keywords?
  value_alignment      : Does the generated core_value align with the reference core_value?
  pain_alignment       : Do generated pain_points reflect the same frustrations as reference?
  behavioral_alignment : Does generated description+motivation match reference consumer behavior?
  sentiment_alignment  : Does the inferred sentiment direction and rough proportion match reference?

matching_score : average of the 5 dimensions above

validity_score : How well is the GENERATED persona grounded in the DOMAIN TOPICS listed?
  1.0 = all content clearly grounded in the domain topics provided
  0.7 = most content grounded, minor unsupported details
  0.5 = partially grounded
  0.3 = weakly grounded
  0.0 = completely fabricated

Return ONLY valid JSON:
{
  "keyword_alignment":    <float 0.0-1.0>,
  "value_alignment":      <float 0.0-1.0>,
  "pain_alignment":       <float 0.0-1.0>,
  "behavioral_alignment": <float 0.0-1.0>,
  "sentiment_alignment":  <float 0.0-1.0>,
  "matching_score":       <average of 5 dimensions>,
  "validity_score":       <float 0.0-1.0>,
  "reasoning":            "<max 80 chars>"
}"""


def _fmt_persona(p, label):
    sr  = p.get("sentiment_ratio", {})
    pos = sr.get("Positive", 0)
    neu = sr.get("Neutral",  0)
    neg = sr.get("Negative", 0)
    return "\n".join([
        "## " + label,
        "- name: "        + p.get("persona_name", ""),
        "- keywords: "    + ", ".join(p.get("keywords", [])),
        "- core_value: "  + p.get("core_value",  ""),
        "- description: " + p.get("description", ""),
        "- motivation: "  + p.get("motivation",  ""),
        "- pain_points: " + p.get("pain_points", ""),
        f"- sentiment_ratio: Pos={round(pos,2)} Neu={round(neu,2)} Neg={round(neg,2)}",
    ])


def _fmt_domain_topics(gt_personas):
    topics = set()
    for gtp in gt_personas:
        topics.update(gtp.get("keywords", []))
        topics.update(gtp.get("top_aspects", gtp.get("top_topics", [])))
    return "## DOMAIN TOPICS\n- " + "\n- ".join(sorted(topics))


def _call_judge(gen_p, gt_p, all_gt_personas=None):
    try:
        from openai import OpenAI
        client = OpenAI()
    except Exception as e:
        print("    [ERROR] OpenAI 초기화 실패:", e)
        return None

    domain_block = ("\n\n" + _fmt_domain_topics(all_gt_personas)) if all_gt_personas else ""
    user_prompt  = (
        _fmt_persona(gt_p,  "REFERENCE (Ground Truth)")
        + "\n\n"
        + _fmt_persona(gen_p, "GENERATED")
        + domain_block
    )

    try:
        resp = client.chat.completions.create(
            model=GPT_MODEL, max_tokens=400,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
        )
        raw    = resp.choices[0].message.content.strip()
        raw    = re.sub(r"```json\s*", "", raw)
        raw    = re.sub(r"```\s*",     "", raw)
        parsed = json.loads(raw)
        DIMS   = ["keyword_alignment", "value_alignment", "pain_alignment",
                  "behavioral_alignment", "sentiment_alignment"]
        for d in DIMS:
            parsed.setdefault(d, 0.0)
            parsed[d] = round(max(0.0, min(1.0, float(parsed[d]))), 4)
        parsed["matching_score"] = round(sum(parsed[d] for d in DIMS) / len(DIMS), 4)
        parsed.setdefault("validity_score", 0.5)
        parsed["validity_score"] = round(max(0.0, min(1.0, float(parsed["validity_score"]))), 4)
        return parsed
    except Exception as e:
        print("    [WARN] Judge 실패:", e)
        return None


def evaluate_llm_judge(gen_personas, gt_personas):
    """헝가리안 알고리즘으로 최적 매칭 후 평균 점수 계산"""
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    DIMS = ["keyword_alignment", "value_alignment", "pain_alignment",
            "behavioral_alignment", "sentiment_alignment"]

    judge_matrix, detail_matrix = [], []
    print(f"    Judge: {len(gen_personas)}x{len(gt_personas)} 쌍 비교")

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
            time.sleep(API_SLEEP * 0.5)
        judge_matrix.append(row_s)
        detail_matrix.append(row_d)

    mat = np.array(judge_matrix)
    row_ind, col_ind = linear_sum_assignment(-mat)

    best_mapping = []
    for r, c in zip(row_ind, col_ind):
        d = detail_matrix[r][c]
        best_mapping.append({
            "gen_name": gen_personas[r].get("persona_name", ""),
            "gt_name":  gt_personas[c].get("persona_name",  ""),
            **{k: d.get(k, 0.0) for k in DIMS + ["matching_score", "validity_score", "reasoning"]},
            "success":  d.get("matching_score", 0.0) >= JUDGE_THRESHOLD,
        })

    matching_vals = [detail_matrix[r][c]["matching_score"] for r, c in zip(row_ind, col_ind)]
    validity_vals = [detail_matrix[r][c]["validity_score"]  for r, c in zip(row_ind, col_ind)]
    dim_avgs = {
        dim: round(float(np.mean([detail_matrix[r][c].get(dim, 0.0) for r, c in zip(row_ind, col_ind)])), 4)
        for dim in DIMS
    }
    n_success = sum(1 for m in best_mapping if m["success"])

    return {
        "judge_matrix":         judge_matrix,
        "best_mapping":         best_mapping,
        "matching_score":       round(float(np.mean(matching_vals)), 4),
        "validity_score":       round(float(np.mean(validity_vals)),  4),
        "mean_all_matching":    round(float(mat.mean()), 4),
        "dim_averages":         dim_avgs,
        "n_success":            n_success,
        "mapping_success_rate": round(n_success / max(len(gt_personas), 1), 4),
    }


def _best_match_sim(gen_words, gt_words):
    if not gen_words or not gt_words:
        return 0.0
    gt_vecs = [(w, get_embedding(w)) for w in gt_words if w]
    gt_vecs = [(w, v) for w, v in gt_vecs if v is not None]
    if not gt_vecs:
        return 0.0
    sims = []
    for gw in gen_words:
        vec_gw = get_embedding(gw)
        if vec_gw is None:
            continue
        sims.append(max(cosine_sim(vec_gw, v) for _, v in gt_vecs))
    return round(sum(sims) / len(sims), 4) if sims else 0.0


def evaluate_topic_sim(gen_personas, gt_personas, best_mapping):
    pair_details, sims = [], []
    for m in best_mapping:
        gen_p = next((p for p in gen_personas if p.get("persona_name") == m["gen_name"]), None)
        gt_p  = next((p for p in gt_personas  if p.get("persona_name") == m["gt_name"]),  None)
        if gen_p is None or gt_p is None:
            pair_details.append({"gen_name": m["gen_name"], "gt_name": m["gt_name"], "topic_sim": 0.0})
            sims.append(0.0)
            continue
        gen_asp = gen_p.get("top_aspects", gen_p.get("top_topics", []))
        gt_asp  = gt_p.get("top_aspects",  gt_p.get("top_topics",  []))
        sim = _best_match_sim(gen_asp, gt_asp)
        sims.append(sim)
        pair_details.append({
            "gen_name":   m["gen_name"],
            "gt_name":    m["gt_name"],
            "gen_aspects": gen_asp,
            "gt_aspects":  gt_asp,
            "topic_sim":   sim,
        })
        print(f"      topic_sim: {m['gen_name'][:20]} vs {m['gt_name'][:20]} = {sim:.3f}")

    topic_sim = round(sum(sims) / len(sims), 4) if sims else 0.0
    return {"topic_sim": topic_sim, "pair_details": pair_details}


def load_gt_personas(file_key):
    path = FILE_CONFIG[file_key]["gt_persona_path"]
    if not os.path.exists(path):
        print(f"    [WARN] GT 파일 없음: {path}")
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_single(file_key, run_id, persona_dir, eval_out_dir, gt_personas):
    """생성된 EXP_RAW 페르소나를 GT와 비교 평가"""
    path = os.path.join(persona_dir, file_key, "EXP_RAW_persona.json")
    if not os.path.exists(path):
        print(f"    [SKIP] 생성 결과 없음: {path}")
        return None

    with open(path, encoding="utf-8") as f:
        generated = json.load(f)

    gen_personas = generated.get("generated_personas", [])
    if not gen_personas or not gt_personas:
        print(f"    [SKIP] 페르소나 또는 GT 없음")
        return None

    print(f"    [Judge] [{file_key}] EXP_RAW")
    judge = evaluate_llm_judge(gen_personas, gt_personas)
    print(f"      matching={judge['matching_score']} validity={judge['validity_score']}")

    print(f"    [topic_sim] [{file_key}] EXP_RAW")
    tsim = evaluate_topic_sim(gen_personas, gt_personas, judge["best_mapping"])
    print(f"      topic_sim={tsim['topic_sim']}")

    result = {
        "file_key":          file_key,
        "exp_id":            "EXP_RAW",
        "method":            "RAW_REVIEW_INJECTION",
        "lang":              generated.get("lang"),
        "n_reviews_injected": generated.get("n_reviews_injected"),
        "judge":             judge,
        "topic_sim_result":  tsim,
        "summary": {
            "matching_score": judge["matching_score"],
            "validity_score": judge["validity_score"],
            "topic_sim":      tsim["topic_sim"],
            "dim_averages":   judge["dim_averages"],
        },
    }

    out_dir = os.path.join(eval_out_dir, file_key)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "EXP_RAW_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"    저장: {out_path}")
    return result


# =================================================================
# 6. Run 실행
# =================================================================

def run_persona_dir(run_id):
    return os.path.join(BASE_DIR, f"run_{run_id:02d}", "persona_results")

def run_eval_dir(run_id):
    return os.path.join(BASE_DIR, f"run_{run_id:02d}", "evaluation_results")


def run_one(run_id, file_keys):
    """1회 실행: 생성 + 평가"""
    persona_dir = run_persona_dir(run_id)
    eval_dir    = run_eval_dir(run_id)
    os.makedirs(persona_dir, exist_ok=True)
    os.makedirs(eval_dir,    exist_ok=True)

    # run마다 다른 seed (샘플링 다양성)
    run_seed = RANDOM_SEED + run_id * 7

    SEP = "=" * 65
    print(f"\n{SEP}")
    print(f"  RUN {run_id:02d}  |  EXP_RAW  |  대상: {file_keys}")
    print(f"{SEP}")

    eval_results = []

    for file_key in file_keys:
        gt_personas = load_gt_personas(file_key)

        print(f"\n  [{file_key}] EXP_RAW 생성 시작")

        # ── 생성 (이미 있으면 스킵)
        gen_path = os.path.join(persona_dir, file_key, "EXP_RAW_persona.json")
        if os.path.exists(gen_path):
            print("    생성 스킵 (이미 존재)")
        else:
            generate_raw_persona(file_key, persona_dir, run_seed)
            time.sleep(API_SLEEP)

        # ── 평가 (이미 있으면 스킵)
        eval_path = os.path.join(eval_dir, file_key, "EXP_RAW_eval.json")
        if os.path.exists(eval_path):
            print("    평가 스킵 (이미 존재)")
            with open(eval_path, encoding="utf-8") as f:
                eval_results.append(json.load(f))
        else:
            r = evaluate_single(file_key, run_id, persona_dir, eval_dir, gt_personas)
            if r:
                eval_results.append(r)
            time.sleep(API_SLEEP)

    return eval_results


# =================================================================
# 7. 통계 집계
# =================================================================

def compute_statistics(file_keys):
    import numpy as np

    data = {}   # {file_key: {metric: [values]}}

    for run_id in range(1, N_RUNS + 1):
        eval_dir = run_eval_dir(run_id)
        for file_key in file_keys:
            path = os.path.join(eval_dir, file_key, "EXP_RAW_eval.json")
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as f:
                r = json.load(f)
            if file_key not in data:
                data[file_key] = {"matching": [], "validity": [], "topic_sim": []}
            s = r.get("summary", {})
            if s.get("matching_score") is not None:
                data[file_key]["matching"].append(s["matching_score"])
            if s.get("validity_score") is not None:
                data[file_key]["validity"].append(s["validity_score"])
            if s.get("topic_sim") is not None:
                data[file_key]["topic_sim"].append(s["topic_sim"])

    SEP = "=" * 90
    print(f"\n{SEP}")
    print(f"  EXP_RAW 최종 통계  |  N_RUNS={N_RUNS}")
    print(SEP)
    print(f"  {'파일':<14} {'N':>3} {'Matching':>16} {'Validity':>16} {'Topic_Sim':>16}")
    print(f"  {'-'*70}")

    stats_rows = []
    for file_key in file_keys:
        if file_key not in data:
            continue
        d   = data[file_key]
        row = {"file_key": file_key, "exp_id": "EXP_RAW"}
        for metric, vals in d.items():
            if vals:
                row[f"{metric}_mean"] = round(float(np.mean(vals)), 4)
                row[f"{metric}_std"]  = round(float(np.std(vals)),  4)
                row[f"{metric}_n"]    = len(vals)
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"]  = None
                row[f"{metric}_n"]    = 0
        stats_rows.append(row)

        n = row.get("matching_n", 0)
        m = f"{row.get('matching_mean',0):.3f}±{row.get('matching_std',0):.3f}"
        v = f"{row.get('validity_mean',0):.3f}±{row.get('validity_std',0):.3f}"
        t = f"{row.get('topic_sim_mean',0):.3f}±{row.get('topic_sim_std',0):.3f}"
        print(f"  {file_key:<14} {n:>3} {m:>16} {v:>16} {t:>16}")

    print(SEP)

    # 저장
    stats_path_dir = os.path.join(BASE_DIR, "statistics")
    os.makedirs(stats_path_dir, exist_ok=True)
    stats_path = os.path.join(stats_path_dir, "EXP_RAW_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_rows, f, ensure_ascii=False, indent=2)
    print(f"\n  통계 저장: {stats_path}")
    return stats_rows


# =================================================================
# 8. 진입점
# =================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EXP_RAW 단독 실험 파이프라인")
    parser.add_argument("--runs",  type=int, default=N_RUNS,
                        help=f"반복 횟수 (기본: {N_RUNS})")
    parser.add_argument("--files", nargs="+",
                        choices=list(FILE_CONFIG.keys()),
                        default=DEFAULT_FILE_KEYS,
                        help=f"실행할 파일 키 (기본: {DEFAULT_FILE_KEYS})")
    parser.add_argument("--stats-only", action="store_true",
                        help="생성/평가 없이 통계만 재계산")
    args = parser.parse_args()

    file_keys = args.files
    N_RUNS    = args.runs

    if args.stats_only:
        print(f"[STATS ONLY]  파일: {file_keys}  N_RUNS={N_RUNS}")
        compute_statistics(file_keys)
    else:
        print(f"\n총 {N_RUNS}회 반복 시작")
        print(f"파일: {file_keys}")
        print(f"GPT 모델: {GPT_MODEL}")
        print(f"리뷰 샘플 수: VOC={MAX_REVIEWS_VOC}, VAD={MAX_REVIEWS_VAD}")
        print()

        for run_id in range(1, N_RUNS + 1):
            print(f"\n{'#'*65}")
            print(f"  RUN {run_id}/{N_RUNS}")
            print(f"{'#'*65}")
            run_one(run_id, file_keys)

        print("\n\n모든 run 완료 → 통계 집계")
        compute_statistics(file_keys)
