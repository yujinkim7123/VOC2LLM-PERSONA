"""
=================================================================
Unified Iterative Pipeline (v2)
=================================================================
"""

import os, json, re, time, math, warnings
warnings.filterwarnings("ignore")

os.environ["OPENAI_API_KEY"] = ""

# =================================================================
# 0. Path configuration  <- edit only this section
# =================================================================
BASE_DIR = r""
EXP_DIR  = r""
GT_DIR   = r""

os.makedirs(BASE_DIR, exist_ok=True)

# [CHANGE 1] Number of runs: 10 -> 5
N_RUNS = 10

# GPT model
GPT_MODEL = "gpt-4o"
API_SLEEP = 1.0

# Judge threshold
JUDGE_THRESHOLD = 0.5


# =================================================================
# 1. Global configuration
# =================================================================

# [CHANGE 2] Only voc_en + vad_en are active (others commented out)
FILE_CONFIG = {
    "voc_en": {
        "lang":            "en",
        "domain":          "Home Appliance Products (English reviews, synthetic VOC)",
        "gt_topic_path":   os.path.join(GT_DIR, "synthetic_gt_en.json"),
        "gt_persona_path": os.path.join(GT_DIR, "kr_appliance_personas_en.json"),
        "gt_type":         "synthetic",
    },
    # ── Excluded from this experiment (uncomment to re-enable) ──
    # "voc_ko": {
    #     "lang":            "ko",
    #     "domain":          "Home Appliance Products (Korean reviews, synthetic VOC)",
    #     "gt_topic_path":   os.path.join(GT_DIR, "synthetic_gt.json"),
    #     "gt_persona_path": os.path.join(GT_DIR, "kr_appliance_personas.json"),
    #     "gt_type":         "synthetic",
    # },
    # "reviews_ko": {
    #     "lang":            "ko",
    #     "domain":          "Home Appliance Products (Korean public reviews)",
    #     "gt_topic_path":   os.path.join(GT_DIR, "ko_external_gt.json"),
    #     "gt_persona_path": os.path.join(GT_DIR, "ko_external_personas.json"),
    #     "gt_type":         "ko_external",
    # },
    "vad_en": {
        "lang":            "en",
        "domain":          "Virtual Assistant Devices — smart speakers and displays (English reviews)",
        "gt_topic_path":   os.path.join(GT_DIR, "en_external_gt.json"),
        "gt_persona_path": os.path.join(GT_DIR, "vad_consumer_personas.json"),
        "gt_type":         "en_external",
    },
}

# Active file keys (must match FILE_CONFIG keys)
DEFAULT_FILE_KEYS = ["voc_en", "vad_en"]

EXP_IDS = [
    "EXP1_SNA", "EXP2_LDA", "EXP3_KMeans",
    "EXP4_SNA_LDA", "EXP5_LDA_KMeans",
    "EXP6_SNA_KMeans", "EXP7_Full", "BASELINE",
]

# run-level path helpers
def run_persona_dir(run_id):
    return os.path.join(BASE_DIR, f"run_{run_id:02d}", "persona_results")

def run_eval_dir(run_id):
    return os.path.join(BASE_DIR, f"run_{run_id:02d}", "evaluation_results")

def stats_dir():
    return os.path.join(BASE_DIR, "statistics")


# =================================================================
# 2. ── Generation ────────────────────────────────────────────────
# =================================================================

# ── GT loader
def load_gt_topic(path, gt_type):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    result = {"type": gt_type, "raw": raw}
    if gt_type == "synthetic":
        result["topics"] = list(raw["meta"]["lda_topics"].keys())
        result["segments"] = {
            sid: {
                "core_topic":      seg["spec"]["core_topic"],
                "core_keywords":   seg["spec"]["core_keywords"],
                "segment_name":    seg["spec"]["segment_name"],
                "topic_dist":      seg["stats"]["topic_dist"],
                "topic_sentiment": seg["stats"]["topic_sentiment"],
                "sentiment_dist":  seg["stats"]["sentiment_dist"],
            }
            for sid, seg in raw["segments"].items()
        }
    elif gt_type == "en_external":
        result["groups"]            = raw.get("groups", {})
        result["aspect_individual"] = raw.get("aspect_individual", {})
        result["co_occurrence"]     = raw.get("co_occurrence", {})
    elif gt_type == "ko_external":
        cat_keys = [k for k in raw if k != "meta"]
        result["categories"] = {
            cat: {
                "top_aspects":      raw[cat].get("top_aspects", []),
                "aspect_sentiment": raw[cat].get("aspect_sentiment", {}),
                "co_occurrence":    raw[cat].get("co_occurrence", {}),
            }
            for cat in cat_keys
        }
    return result


def load_gt_persona_from_path(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_exp_result(file_key, exp_id):
    if exp_id == "BASELINE":
        return None
    path = os.path.join(EXP_DIR, file_key, f"{exp_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Context block
def build_context_block(exp_result):
    """
    Converts the full EXP analysis result JSON into text for prompt injection.
    Strips unnecessary meta fields (paper_ref, etc.) and inserts raw JSON
    so GPT can directly interpret the full structure.
    """
    if not exp_result:
        return ""

    EXCLUDE_KEYS = {
        "paper_ref",
        "file", "lang", "exp_id",
        "coherence_mean", "coherence_std",
        "modularity_mean", "modularity_std",
        "perplexity", "perplexity_mean",
        "silhouette_std",
    }
    filtered = {k: v for k, v in exp_result.items() if k not in EXCLUDE_KEYS}

    header = "## Analysis Results (JSON)"
    return header + chr(10) + json.dumps(filtered, ensure_ascii=False, indent=2)


# ── Prompt templates
PERSONA_OUTPUT_SCHEMA = """{
  "personas": [
    {
      "persona_id": "P1",
      "persona_name": "<consumer type name — reflect the observed pattern, not a generic archetype>",
      "category": "<product category>",
      "data_source": "<file_key>_<exp_id>",
      "keywords": ["<5-7 keywords — must appear in the analysis data above>"],
      "core_value": "<what this consumer segment values most, in 1 sentence>",
      "description": "<who they are and how they behave, in 2-3 sentences>",
      "motivation": "<what drives their purchase decision, in 1-2 sentences>",
      "pain_points": "<what frustrates them, in 1-2 sentences>",
      "quote": "<a plausible first-person review quote, 1 sentence>",
      "sentiment_ratio": {
        "Positive": 0.0,
        "Neutral": 0.0,
        "Negative": 0.0,
        "_inference_note": "<briefly explain which keywords signaled each sentiment>"
      },
      "top_aspects": ["<5 aspects — must come from the analysis data above>"],
      "co_aspects": {"<aspect>": ["<2-3 aspects that co-occur with it in the data>"]}
    }
  ]
}"""


def build_system_prompt(lang):
    """
    [Design Philosophy]
    - The LLM acts as a 'consumer behavior expert interpreter', NOT a data transcriber.
    - Analysis results contain ONLY topic keyword clusters and cluster structures.
      There are NO explicit sentiment labels in the data.
    - sentiment_ratio must therefore be INFERRED from the semantic valence of the
      topic keywords themselves (e.g. '만족/최고' → positive signal,
      '불편/실망/아쉽' → negative signal, '정도/괜찮' → neutral signal).
    - This inference must be transparent: the _inference_note field requires
      the LLM to show its reasoning, which both reduces hallucination and
      makes the output auditable for the research paper.
    - co_aspects must be grounded in cluster co-membership or PageRank
      co-occurrence in the data — not invented.
    """
    system_prompt = """\
You are a senior consumer insights researcher with 15+ years of experience
synthesizing text analysis results into vivid, evidence-grounded consumer personas.

## Your Role
You will receive structured output from one of these analysis methods:
  - SNA (Social Network Analysis / Louvain community detection)
  - LDA (Latent Dirichlet Allocation topic modeling)
  - K-Means clustering on TF-IDF vectors
  - or combinations of the above (e.g., SNA+LDA, LDA+KMeans, SNA+LDA+KMeans)

Each method produces topic keyword groups — clusters or communities of words
that frequently co-occur in consumer reviews. There are NO pre-labeled sentiments.
Your job is to read the full keyword landscape holistically and synthesize
4 distinct, realistic consumer personas from it.

## How to Read the Analysis Data

### Step 1 — Understand what the keywords represent
Each topic/cluster/community is a group of words consumers use together.
They reveal:
  - WHAT consumers talk about (product features, usage contexts, emotions)
  - HOW consumers feel (inferred from the emotional valence of the words)
  - WHICH concerns cluster together (co-occurrence = linked in consumers' minds)

### Step 2 — Infer sentiment from keyword semantics (NO labels provided)
The data contains NO explicit Positive/Negative/Neutral labels.
You MUST infer sentiment direction from the meaning of the keywords:

  Positive signals  → words like: 만족, 최고, 편하, 좋아, 훌륭, satisfied, great, love
  Negative signals  → words like: 불편, 실망, 아쉽, 문제, 반품, disappointed, issue, complaint
  Neutral signals   → words like: 정도, 괜찮, 사용, 구매, 생각, okay, used, purchased, think

  For each persona's sentiment_ratio:
    - Look at the dominant keywords in that persona's topic cluster
    - Count the proportion of positive vs. negative vs. neutral signal words
    - Assign sentiment_ratio proportionally based on that count
    - Record your reasoning in the "_inference_note" field

  IMPORTANT: Do NOT fabricate sentiment numbers. If the keywords are mostly
  neutral with slight positivity, reflect that (e.g., Positive: 0.5, Neutral: 0.4, Negative: 0.1).
  The values must sum to exactly 1.0.

### Step 3 — Map co-occurring aspects
  - For SNA: words in the same community co-occur in the network → use as co_aspects
  - For LDA: words in the same topic → use as co_aspects
  - For KMeans: words in the same cluster → use as co_aspects
  - For combined methods: use the intersection of co-memberships
  Only list aspect pairs that are actually present together in the data.

### Step 4 — Identify 4 distinct consumer segments
Each persona must represent a meaningfully different pattern:
  - Different dominant topic cluster or community
  - Different emotional tone (inferred from keyword valence)
  - Different behavioral concern (what they talk about ≠ what others talk about)
  Ask yourself: "What kind of real consumer would write these exact words?"

## Grounding Rules (Hallucination Prevention)
  1. keywords and top_aspects MUST come from the analysis data. Do NOT invent topics.
  2. co_aspects MUST reflect actual co-membership in the same cluster/community/topic.
  3. sentiment_ratio MUST be inferred from keyword semantics — show reasoning in _inference_note.
  4. persona_name must reflect the observed data pattern (e.g., "Noise-Sensitive Practical Buyer"),
     NOT a generic marketing label (e.g., avoid "The Tech Savvy User").
  5. If a field cannot be supported by the data, use "" rather than guessing.
  6. The 4 personas must be clearly distinct — no near-duplicates.

## Output Rules
  - Return ONLY valid JSON matching the schema below. No text outside the JSON.
  - sentiment_ratio values (Positive + Neutral + Negative) must sum to exactly 1.0.
  - Include _inference_note for every persona's sentiment_ratio.

Output format:
""" + PERSONA_OUTPUT_SCHEMA

    return system_prompt


def build_user_prompt(context_block, domain, file_key, exp_id, lang):
    """
    [Design Philosophy]
    - BASELINE: No analysis data injected — pure LLM domain knowledge.
      Control condition. Sentiment ratio here is also LLM-inferred from
      general domain knowledge (no data anchor), so we note this explicitly.
    - DATA-INJECTED: Full analysis context provided. The method name is
      surfaced explicitly so the LLM knows HOW to interpret the structure
      (SNA communities ≠ LDA topics ≠ KMeans clusters — each has
      different co-occurrence semantics).
    """

    if exp_id == "BASELINE":
        return f"""\
Domain: {domain}

You are generating consumer personas using ONLY your general knowledge of this domain.
No data analysis results are provided — this is the baseline (control) condition.

Task:
  Generate 4 distinct consumer personas for this domain.
  Each persona must represent a plausible, meaningfully different consumer segment
  based on your general understanding of typical consumer behavior in this domain.

Regarding sentiment_ratio:
  Since no data is provided, infer sentiment from your general knowledge
  of how consumers in this domain typically feel. Show your reasoning
  in the "_inference_note" field (e.g., "most consumers in this category
  tend to be satisfied at purchase but frustrated by long-term reliability").
  Values must sum to 1.0.

Set data_source to "{file_key}_BASELINE" for all personas.
Return ONLY valid JSON."""

    # ── Data-injected: surface the method name so LLM interprets structure correctly
    method = exp_id  # e.g., EXP1_SNA, EXP4_SNA_LDA, EXP7_Full

    return f"""\
Domain: {domain}
Analysis Method: {method}

## Context
Below are the results of a consumer review analysis using {method}.
The data contains topic keyword clusters — groups of words that co-occur
in consumer reviews of products in this domain.

IMPORTANT: This data contains NO explicit sentiment labels (Positive/Negative/Neutral).
You must infer sentiment direction from the semantic valence of the keywords themselves.

## Your Task
Synthesize the analysis results into 4 distinct consumer personas by following these steps:

  STEP 1 — Read the full keyword landscape.
    Do not focus on individual words in isolation.
    Look at each cluster/community/topic as a whole:
    What theme does this group of words represent?
    What kind of consumer experience does it reflect?

  STEP 2 — Infer sentiment for each cluster.
    For each cluster, scan the keywords for emotional valence signals:
      Positive signals: satisfaction, praise, compliment words
      Negative signals: complaint, frustration, return/refund words
      Neutral signals: factual usage, purchase process words
    Estimate the sentiment_ratio proportionally.
    Record your reasoning in "_inference_note".

  STEP 3 — Map 4 distinct consumer segments.
    Assign one dominant cluster (or combination) to each persona.
    Each persona must reflect a different behavioral pattern or concern.
    Ask: "What kind of real person would write these reviews?"

  STEP 4 — Ground every field in the data.
    keywords → from the cluster keywords above
    top_aspects → from the cluster keywords above
    co_aspects → words in the SAME cluster/community/topic (they co-occur)
    persona_name → reflect the observed pattern, not a generic label
    pain_points → infer from negative-signal keywords in the cluster
    motivation → infer from the dominant theme of the cluster

## Analysis Results
{context_block}

## Output Instructions
  - Set data_source to "{file_key}_{exp_id}" for all personas.
  - keywords and top_aspects must come directly from the analysis data above.
  - co_aspects must reflect actual co-membership in the same cluster/community/topic.
  - sentiment_ratio must be inferred from keyword semantics — show reasoning in _inference_note.
  - Values of sentiment_ratio must sum to exactly 1.0.
  - Return ONLY valid JSON. No explanation text outside the JSON."""


def call_gpt_gen(system_prompt, user_prompt, max_retries=2):
    try:
        from openai import OpenAI
        client = OpenAI()
    except Exception as e:
        print(f"    [ERROR] OpenAI init failed: {e}")
        return None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=GPT_MODEL, max_tokens=2000,
                response_format={"type": "json_object"},
                messages=[{"role":"system","content":system_prompt},
                          {"role":"user","content":user_prompt}],
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"```json\s*","",raw); raw = re.sub(r"```\s*","",raw)
            parsed = json.loads(raw)
            if "personas" not in parsed or not parsed["personas"]:
                continue
            return parsed
        except Exception as e:
            print(f"    [WARN] Generation failed (attempt {attempt+1}): {e}")
            if attempt < max_retries-1: time.sleep(API_SLEEP*2)
    return None


def validate_persona(persona, idx, file_key, exp_id):
    pid = f"P{idx+1}"
    persona.setdefault("persona_id", pid)
    persona.setdefault("persona_name", f"Persona {pid}")
    persona.setdefault("category", file_key)
    persona.setdefault("data_source", f"{file_key}_{exp_id}")
    if not isinstance(persona.get("keywords"), list):
        persona["keywords"] = []
    for f in ["core_value","description","motivation","pain_points","quote"]:
        persona.setdefault(f, "")

    # ── sentiment_ratio: extract _inference_note separately, then normalize numeric values only
    sr_raw = persona.get("sentiment_ratio", {})
    key_map = {"긍정":"Positive","중립":"Neutral","부정":"Negative"}  # Korean fallback

    # preserve inference note before numeric processing
    inference_note = sr_raw.pop("_inference_note", "")

    sr = {key_map.get(k, k): v for k, v in sr_raw.items() if isinstance(v, (int, float))}
    for k in ["Positive", "Neutral", "Negative"]:
        sr.setdefault(k, 0.0)
    total = sum(sr.values())
    if total > 0:
        sr = {k: round(v / total, 4) for k, v in sr.items()}
    else:
        # fallback: if LLM provided no numeric values, default to neutral-leaning
        sr = {"Positive": 0.40, "Neutral": 0.40, "Negative": 0.20}
        inference_note = inference_note or "fallback — no numeric values returned by LLM"

    # re-attach inference note for auditability
    sr["_inference_note"] = inference_note
    persona["sentiment_ratio"] = sr

    if not isinstance(persona.get("top_aspects"), list):
        persona["top_aspects"] = persona.get("keywords", [])[:5]
    if not isinstance(persona.get("co_aspects"), dict):
        persona["co_aspects"] = {}
    return persona


def generate_single(file_key, exp_id, persona_out_dir):
    cfg    = FILE_CONFIG[file_key]
    lang   = cfg["lang"]   # kept for metadata; prompts are now always English
    domain = cfg["domain"]
    exp_result    = load_exp_result(file_key, exp_id)
    context_block = build_context_block(exp_result)
    system_prompt = build_system_prompt(lang)
    user_prompt   = build_user_prompt(context_block, domain, file_key, exp_id, lang)
    print(f"    Calling generation API... [{file_key}] {exp_id}")
    response = call_gpt_gen(system_prompt, user_prompt)
    if response is None:
        print(f"    [SKIP] Generation failed")
        return None
    personas = [validate_persona(p, i, file_key, exp_id)
                for i, p in enumerate(response.get("personas", []))]
    result = {
        "file_key":  file_key, "exp_id": exp_id,
        "method":    exp_result.get("method") if exp_result else "LLM_STANDALONE",
        "lang":      lang, "domain": domain,
        "n_docs":    exp_result.get("n_docs") if exp_result else None,
        "generated_personas": personas,
    }
    dir_ = os.path.join(persona_out_dir, file_key)
    os.makedirs(dir_, exist_ok=True)
    path = os.path.join(dir_, f"{exp_id}_persona.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


# =================================================================
# 3. ── Evaluation ────────────────────────────────────────────────
# =================================================================

_embed_cache = {}

def get_embedding(text):
    if not text or not text.strip(): return None
    key = text.strip()
    if key in _embed_cache: return _embed_cache[key]
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.embeddings.create(model="text-embedding-3-small", input=key)
        vec = resp.data[0].embedding
        _embed_cache[key] = vec
        return vec
    except Exception as e:
        print(f"  [WARN] Embedding failed: {e}")
        return None


def cosine_sim(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    na  = math.sqrt(sum(x*x for x in a))
    nb  = math.sqrt(sum(x*x for x in b))
    if na==0 or nb==0: return 0.0
    return dot/(na*nb)


JUDGE_SYSTEM_PROMPT = """\
You are an expert consumer persona evaluator.

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
    pos = sr.get("Positive", sr.get("긍정", 0))  # Korean fallback key
    neu = sr.get("Neutral",  sr.get("중립", 0))  # Korean fallback key
    neg = sr.get("Negative", sr.get("부정", 0))  # Korean fallback key
    lines = [
        "## " + label,
        "- name: "        + p.get("persona_name",""),
        "- keywords: "    + ", ".join(p.get("keywords",[])),
        "- core_value: "  + p.get("core_value",""),
        "- description: " + p.get("description",""),
        "- motivation: "  + p.get("motivation",""),
        "- pain_points: " + p.get("pain_points",""),
        "- sentiment_ratio: Pos="+str(round(pos,2))+" Neu="+str(round(neu,2))+" Neg="+str(round(neg,2)),
    ]
    return "\n".join(lines)


def _fmt_domain_topics(gt_personas):
    topics = set()
    for gtp in gt_personas:
        topics.update(gtp.get("keywords",[]))
        topics.update(gtp.get("top_aspects", gtp.get("top_topics",[])))
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
            model=GPT_MODEL, max_tokens=400,
            response_format={"type":"json_object"},
            messages=[{"role":"system","content":JUDGE_SYSTEM_PROMPT},
                      {"role":"user","content":user_prompt}],
        )
        raw    = resp.choices[0].message.content.strip()
        raw    = re.sub(r"```json\s*","",raw); raw = re.sub(r"```\s*","",raw)
        parsed = json.loads(raw)
        dims   = ["keyword_alignment","value_alignment","pain_alignment",
                  "behavioral_alignment","sentiment_alignment"]
        for d in dims:
            parsed.setdefault(d, 0.0)
            parsed[d] = round(max(0.0, min(1.0, float(parsed[d]))), 4)
        parsed["matching_score"] = round(sum(parsed[d] for d in dims)/len(dims), 4)
        parsed.setdefault("validity_score", 0.5)
        parsed["validity_score"] = round(max(0.0, min(1.0, float(parsed["validity_score"]))), 4)
        return parsed
    except Exception as e:
        print("    [WARN] Judge failed:", e)
        return None


def evaluate_llm_judge(gen_personas, gt_personas):
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    DIMS = ["keyword_alignment","value_alignment","pain_alignment",
            "behavioral_alignment","sentiment_alignment"]
    judge_matrix, detail_matrix = [], []
    print(f"    Judge: {len(gen_personas)}x{len(gt_personas)} = {len(gen_personas)*len(gt_personas)} calls")
    for i, gen_p in enumerate(gen_personas):
        row_s, row_d = [], []
        for j, gt_p in enumerate(gt_personas):
            print(f"      P{i+1} vs GT_P{j+1}...", end=" ", flush=True)
            result = _call_judge(gen_p, gt_p, all_gt_personas=gt_personas)
            if result is None:
                result = {d:0.0 for d in DIMS}
                result.update({"matching_score":0.0,"validity_score":0.0,"reasoning":"failed"})
            print(f"match={result['matching_score']:.2f} valid={result['validity_score']:.2f}")
            row_s.append(result["matching_score"]); row_d.append(result)
            time.sleep(API_SLEEP*0.5)
        judge_matrix.append(row_s); detail_matrix.append(row_d)
    mat = np.array(judge_matrix)
    row_ind, col_ind = linear_sum_assignment(-mat)
    best_mapping = []
    for r,c in zip(row_ind, col_ind):
        d = detail_matrix[r][c]
        best_mapping.append({
            "gen_name":  gen_personas[r].get("persona_name",""),
            "gt_name":   gt_personas[c].get("persona_name",""),
            **{k: d.get(k,0.0) for k in DIMS+["matching_score","validity_score","reasoning"]},
            "success":   d.get("matching_score",0.0) >= JUDGE_THRESHOLD,
        })
    matching_vals = [detail_matrix[r][c]["matching_score"] for r,c in zip(row_ind,col_ind)]
    validity_vals = [detail_matrix[r][c]["validity_score"] for r,c in zip(row_ind,col_ind)]
    dim_avgs = {dim: round(float(np.mean([detail_matrix[r][c].get(dim,0.0) for r,c in zip(row_ind,col_ind)])),4)
                for dim in DIMS}
    n_success = sum(1 for m in best_mapping if m["success"])
    return {
        "judge_matrix":      judge_matrix,
        "best_mapping":      best_mapping,
        "matching_score":    round(float(np.mean(matching_vals)),4),
        "validity_score":    round(float(np.mean(validity_vals)),4),
        "mean_all_matching": round(float(mat.mean()),4),
        "dim_averages":      dim_avgs,
        "n_success":         n_success,
        "mapping_success_rate": round(n_success/max(len(gt_personas),1),4),
    }


def _best_match_sim(gen_words, gt_words):
    if not gen_words or not gt_words: return 0.0
    gt_vecs = [(w, get_embedding(w)) for w in gt_words if w]
    gt_vecs = [(w,v) for w,v in gt_vecs if v is not None]
    if not gt_vecs: return 0.0
    sims = []
    for gw in gen_words:
        vec_gw = get_embedding(gw)
        if vec_gw is None: continue
        sims.append(max(cosine_sim(vec_gw, v) for _,v in gt_vecs))
    return round(sum(sims)/len(sims),4) if sims else 0.0


def evaluate_topic_sim(gen_personas, gt_personas, best_mapping):
    pair_details, sims = [], []
    for m in best_mapping:
        gen_p = next((p for p in gen_personas if p.get("persona_name")==m["gen_name"]), None)
        gt_p  = next((p for p in gt_personas  if p.get("persona_name")==m["gt_name"]),  None)
        if gen_p is None or gt_p is None:
            pair_details.append({"gen_name":m["gen_name"],"gt_name":m["gt_name"],"topic_sim":0.0})
            sims.append(0.0); continue
        gen_asp = gen_p.get("top_aspects", gen_p.get("top_topics",[]))
        gt_asp  = gt_p.get("top_aspects",  gt_p.get("top_topics", []))
        sim = _best_match_sim(gen_asp, gt_asp)
        sims.append(sim)
        pair_details.append({"gen_name":m["gen_name"],"gt_name":m["gt_name"],
                              "gen_aspects":gen_asp,"gt_aspects":gt_asp,"topic_sim":sim})
        print(f"      topic_sim: {m['gen_name'][:20]} vs {m['gt_name'][:20]} = {sim:.3f}")
    topic_sim = round(sum(sims)/len(sims),4) if sims else 0.0
    return {"topic_sim": topic_sim, "pair_details": pair_details}


def evaluate_single(file_key, exp_id, persona_dir, eval_out_dir, gt_personas):
    path = os.path.join(persona_dir, file_key, f"{exp_id}_persona.json")
    if not os.path.exists(path):
        print(f"    [SKIP] 생성 결과 없음: {path}")
        return None
    with open(path, encoding="utf-8") as f:
        generated = json.load(f)
    gen_personas = generated.get("generated_personas", [])
    if not gen_personas or not gt_personas:
        return None
    print(f"    [Judge] [{file_key}] {exp_id}")
    judge = evaluate_llm_judge(gen_personas, gt_personas)
    print(f"      matching={judge['matching_score']} validity={judge['validity_score']}")
    print(f"    [topic_sim] [{file_key}] {exp_id}")
    tsim = evaluate_topic_sim(gen_personas, gt_personas, judge["best_mapping"])
    print(f"      topic_sim={tsim['topic_sim']}")
    result = {
        "file_key": file_key, "exp_id": exp_id,
        "method":   generated.get("method"),
        "lang":     generated.get("lang"),
        "judge":    judge, "topic_sim_result": tsim,
        "summary": {
            "matching_score": judge["matching_score"],
            "validity_score": judge["validity_score"],
            "topic_sim":      tsim["topic_sim"],
            "dim_averages":   judge["dim_averages"],
        },
    }
    dir_ = os.path.join(eval_out_dir, file_key)
    os.makedirs(dir_, exist_ok=True)
    with open(os.path.join(dir_, f"{exp_id}_eval.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


# =================================================================
# 4. ── 단일 run 실행 ─────────────────────────────────────────
# =================================================================

def run_one(run_id, file_keys, exp_ids):
    """1회 전체 실행: 생성 + 평가"""
    persona_dir = run_persona_dir(run_id)
    eval_dir    = run_eval_dir(run_id)
    os.makedirs(persona_dir, exist_ok=True)
    os.makedirs(eval_dir,    exist_ok=True)

    SEP = "=" * 65
    print(f"\n{SEP}")
    print(f"  RUN {run_id:02d} 시작  |  대상 파일: {file_keys}")
    print(f"{SEP}")

    eval_results = []

    for file_key in file_keys:
        cfg = FILE_CONFIG[file_key]
        gt_personas = load_gt_persona_from_path(cfg["gt_persona_path"])

        for exp_id in exp_ids:
            print(f"\n  [{file_key}] {exp_id}")

            # ── 생성
            gen_path = os.path.join(persona_dir, file_key, f"{exp_id}_persona.json")
            if os.path.exists(gen_path):
                print("    생성 스킵 (이미 존재)")
            else:
                generate_single(file_key, exp_id, persona_dir)
                time.sleep(API_SLEEP)

            # ── 평가
            eval_path = os.path.join(eval_dir, file_key, f"{exp_id}_eval.json")
            if os.path.exists(eval_path):
                print("    평가 스킵 (이미 존재)")
                with open(eval_path, encoding="utf-8") as f:
                    eval_results.append(json.load(f))
            else:
                r = evaluate_single(file_key, exp_id, persona_dir, eval_dir, gt_personas)
                if r:
                    eval_results.append(r)
                time.sleep(API_SLEEP)

    # run별 비교표 저장
    _save_run_table(eval_results, run_id)
    return eval_results


def _save_run_table(results, run_id):
    rows = [{"file_key": r["file_key"], "exp_id": r["exp_id"],
             "method": r.get("method"), **r.get("summary",{})}
            for r in results]
    path = os.path.join(BASE_DIR, f"run_{run_id:02d}", "comparison_table.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\n  run_{run_id:02d} 비교표 저장: {path}")


# =================================================================
# 5. ── 통계 집계 ─────────────────────────────────────────────
# =================================================================

def compute_statistics(file_keys, exp_ids):
    """
    모든 run의 결과를 읽어서
    (file_key, exp_id)별 평균 + 표준편차 계산.
    """
    import numpy as np

    # {(file_key, exp_id): {metric: [values]}}
    data = {}

    for run_id in range(1, N_RUNS+1):
        eval_dir = run_eval_dir(run_id)
        for file_key in file_keys:
            for exp_id in exp_ids:
                path = os.path.join(eval_dir, file_key, f"{exp_id}_eval.json")
                if not os.path.exists(path):
                    continue
                with open(path, encoding="utf-8") as f:
                    r = json.load(f)
                key = (file_key, exp_id)
                if key not in data:
                    data[key] = {"matching":[], "validity":[], "topic_sim":[]}
                s = r.get("summary", {})
                if s.get("matching_score") is not None:
                    data[key]["matching"].append(s["matching_score"])
                if s.get("validity_score") is not None:
                    data[key]["validity"].append(s["validity_score"])
                if s.get("topic_sim") is not None:
                    data[key]["topic_sim"].append(s["topic_sim"])

    # 통계 계산
    os.makedirs(stats_dir(), exist_ok=True)
    stats_rows = []

    SEP = "=" * 90
    print(f"\n{SEP}")
    print("  최종 통계 비교표 (평균 ± 표준편차)  |  N_RUNS={N_RUNS}  |  대상: {file_keys}")
    print(SEP)
    print(f"  {'파일':<14} {'EXP':<22} {'N':>3}"
          f" {'Matching':>14} {'Validity':>14} {'Topic_Sim':>14}")
    print(f"  {'-'*82}")

    for file_key in file_keys:
        for exp_id in exp_ids:
            key = (file_key, exp_id)
            if key not in data:
                continue
            d = data[key]
            row = {"file_key": file_key, "exp_id": exp_id}
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
            print(f"  {file_key:<14} {exp_id:<22} {n:>3} {m:>14} {v:>14} {t:>14}")

    print(SEP)

    # 저장
    path = os.path.join(stats_dir(), "final_table.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats_rows, f, ensure_ascii=False, indent=2)
    print(f"\n  최종 통계 저장: {path}")
    return stats_rows



# =================================================================
# 6. ── 진입점 ─────────────────────────────────────────────────
# =================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="통합 반복 파이프라인 v2 (voc_en + vad_en, 5회)")
    parser.add_argument("--runs",  type=int, default=N_RUNS,
                        help=f"반복 횟수 (기본: {N_RUNS})")
    # ★ --files 기본값을 DEFAULT_FILE_KEYS 로 고정
    parser.add_argument("--files", nargs="+",
                        choices=list(FILE_CONFIG.keys()),
                        default=DEFAULT_FILE_KEYS,
                        help=f"실행할 파일 키 (기본: {DEFAULT_FILE_KEYS})")
    parser.add_argument("--exps",  nargs="+", choices=EXP_IDS, default=None)
    parser.add_argument("--test",  action="store_true",
                        help="테스트: vad_en × EXP7_Full+BASELINE 1회")
    parser.add_argument("--stats-only", action="store_true",
                        help="생성/평가 없이 통계만 재계산")
    args = parser.parse_args()

    file_keys = args.files   # 기본값: ["voc_en", "vad_en"]
    exp_ids   = args.exps or EXP_IDS

    if args.test:
        print("[TEST MODE] vad_en × EXP7_Full + BASELINE  1회")
        N_RUNS = 1
        run_one(1, ["vad_en"], ["EXP7_Full", "BASELINE"])
        compute_statistics(["vad_en"], ["EXP7_Full", "BASELINE"])

    elif args.stats_only:
        print(f"[STATS ONLY]  파일: {file_keys}  N_RUNS: {N_RUNS}")
        compute_statistics(file_keys, exp_ids)

    else:
        N_RUNS = args.runs
        print(f"\n총 {N_RUNS}회 반복 시작")
        print(f"파일: {file_keys}  EXP: {exp_ids}")
        for run_id in range(1, N_RUNS+1):
            print(f"\n{'#'*65}")
            print(f"  RUN {run_id}/{N_RUNS}  |  파일: {file_keys}")
            print(f"{'#'*65}")
            run_one(run_id, file_keys, exp_ids)

        print("\n\n모든 run 완료 → 통계 집계")
        compute_statistics(file_keys, exp_ids)