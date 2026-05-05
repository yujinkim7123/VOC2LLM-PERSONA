"""
=================================================================
7가지 조합 실험 파이프라인  [Grid-Search Tuned Version]
=================================================================

입력: preprocess_pipeline.py 가 생성한 전처리 JSON
  preprocessed/preprocessed_voc_en.json
  preprocessed/preprocessed_vad_en.json

실험 구성 (파일별 독립 실행):
  EXP-1 : SNA only         (Louvain)
  EXP-2 : LDA only         (Gensim LDA)
  EXP-3 : KMeans only      (TF-IDF → KMeans)
  EXP-4 : SNA → LDA        (커뮤니티 내 LDA)
  EXP-5 : LDA → KMeans     (토픽 벡터 → KMeans)
  EXP-6 : SNA → KMeans     (PageRank 필터 → KMeans)
  EXP-7 : SNA → LDA → KMeans (전체 파이프라인)

핵심 원칙:
  - 리뷰 텍스트(tokens)만 사용. 메타데이터 일절 사용 안 함
  - 4개 파일 독립 실행 → 결과 병합 없음
  - 공통 평가 지표: Coherence C_v · Modularity · Silhouette
    ※ 메타 미사용 → GT 레이블 없음 → ARI/NMI/Purity 제외

=================================================================
★ 그리드서치 최적 파라미터 확정  [param_gridsearch_result.json 기반]
=================================================================

  [K — 페르소나/토픽 수]
  ┌──────────┬──────────────────────────────────────────────────────┐
  │ 데이터셋 │ K 결정 방식                                          │
  ├──────────┼──────────────────────────────────────────────────────┤
  │ voc_en   │ K=4 고정 (Top-Down 설계, Salminen et al. 2022)       │
  │ vad_en   │ K=6 탐색 결정 (Bottom-Up, STEP 6 LDA C_v 최고값)    │
  │          │   LDA C_v: k=6→0.5323 (최고), k=4→0.5179 (엘보우)  │
  │          │   KMeans Sil: 단조증가 → LDA 기준 따름              │
  └──────────┴──────────────────────────────────────────────────────┘

  [SNA — STEP1 기준]
  ┌──────────┬────────┬──────────┬──────────────┬───────────────┐
  │ 데이터셋 │ window │ min_freq │ Coherence C_v│ kept_edges    │
  ├──────────┼────────┼──────────┼──────────────┼───────────────┤
  │ voc_en   │ 7      │ 5        │ 0.6108       │ 1,817         │
  │ vad_en   │ 10     │ 5        │ 0.7238       │ 11,517        │
  └──────────┴────────┴──────────┴──────────────┴───────────────┘

  [PageRank Top-N — STEP3 기준, coverage ≥ 95% 최소 N]
  ┌──────────┬───────┬──────────┬────────────────────────────────┐
  │ 데이터셋 │ Top-N │ Coverage │ 비고                           │
  ├──────────┼───────┼──────────┼────────────────────────────────┤
  │ voc_en   │ 50    │ 98.45%   │ 50→100 한계효용 +1.14%로 급감  │
  │ vad_en   │ 50    │ 97.38%   │ 50→100 한계효용 +1.52%로 급감  │
  └──────────┴───────┴──────────┴────────────────────────────────┘

  [LDA — STEP4 기준]
  ┌──────────┬────────┬──────────┬──────────┬──────────────────┐
  │ 데이터셋 │ passes │ no_below │ no_above │ 비고             │
  ├──────────┼────────┼──────────┼──────────┼──────────────────┤
  │ voc_en   │ 30     │ 2        │ 0.9      │ C_v=0.4413       │
  │ vad_en   │ 30     │ 1        │ 0.9      │ C_v=0.5048       │
  │          │        │          │          │ no_above: 전구간  │
  │          │        │          │          │ 동일→0.9 유지    │
  └──────────┴────────┴──────────┴──────────┴──────────────────┘
  ※ N_RUNS: 전 구간 std > 0.01 (수렴 안됨) → 고정 시드 앙상블 10회 유지

  [KMeans TF-IDF — STEP2 기준]
  ┌──────────┬──────────────┬────────────┐
  │ 데이터셋 │ max_features │ Silhouette │
  ├──────────┼──────────────┼────────────┤
  │ voc_en   │ 200          │ 0.0476     │
  │ vad_en   │ 200          │ 0.0419     │
  └──────────┴──────────────┴────────────┘

논문 근거:
  SNA Louvain  : Blondel et al. (2008)
  LDA          : Blei, Ng & Jordan (2003)
  KMeans       : MacQueen (1967)
  voc K=4      : Salminen et al. (2022) — 358개 연구 평균 M=4.51
  vad K=6      : Bottom-Up 귀납 탐색 (LDA C_v 엘보우 기반)
  EXP-5 조합   : Scientific Reports (2025)
  Coherence    : Röder et al. (2015, WSDM)
  Silhouette   : Rousseeuw (1987)
=================================================================
"""

import os, json, re, warnings
import numpy as np
from collections import defaultdict, Counter

import networkx as nx
import community as community_louvain

warnings.filterwarnings("ignore")


# =================================================================
# 0. 경로 설정  ← 여기만 수정하세요
# =================================================================
PREPROCESSED_DIR = r""
OUT_DIR          = r""

INPUT_FILES = {
    "voc_en": os.path.join(PREPROCESSED_DIR, "preprocessed_voc_en.json"),
    "vad_en": os.path.join(PREPROCESSED_DIR, "preprocessed_vad_en.json"),
}

os.makedirs(OUT_DIR, exist_ok=True)


# =================================================================
# 1. 하이퍼파라미터 — 데이터셋별 분리
# =================================================================

# ── 공통 고정값
N_RUNS      = 10     # 시드 앙상블 횟수 (N_RUNS std 미수렴 → 10 유지)
SEEDS       = [42, 0, 1, 7, 13, 21, 37, 55, 77, 99]
TOP_N_WORDS = 10     # 토픽 상위 단어 수 (Coherence 계산용)

# =================================================================
# ★ 그리드서치 최적값 반영 — param_gridsearch_result.json 기반
# =================================================================
#
DATASET_PARAMS = {
    "voc_en": {
        # K: Top-Down 도메인 설계 결정 (Salminen et al. 2022)
        "k":               4,
        # SNA: STEP1 최고 C_v=0.6108 (window=7, min_freq=5)
        "window":          7,
        "min_freq":        5,
        # PageRank: STEP3 coverage=98.45% (Top-50)
        "top_pagerank_n":  50,
        # LDA: STEP4 최적 (passes=30, no_below=2, no_above=0.9)
        "lda_passes":      30,
        "lda_no_below":    2,
        "lda_no_above":    0.9,
        # KMeans: STEP2 최고 Silhouette=0.0476 (max_features=200)
        "tfidf_feats":     200,
    },
    "vad_en": {
        # K: Bottom-Up 탐색 결정 (STEP6, LDA C_v 최고 k=6→0.5323)
        "k":               6,
        # SNA: STEP1 최고 C_v=0.7238 (window=10, min_freq=5)
        "window":          10,
        "min_freq":        5,
        # PageRank: STEP3 coverage=97.38% (Top-50)
        "top_pagerank_n":  50,
        # LDA: STEP4 최적 (passes=30, no_below=1, no_above=0.9)
        "lda_passes":      30,
        "lda_no_below":    1,
        "lda_no_above":    0.9,
        # KMeans: STEP2 최고 Silhouette=0.0419 (max_features=200)
        "tfidf_feats":     200,
    },
}


# =================================================================
# 2. 데이터 로드
# =================================================================
def load_preprocessed(path: str) -> list[list[str]]:
    """
    전처리 JSON → 토큰 리스트만 추출.
    메타데이터는 완전히 무시.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    items = raw["data"] if isinstance(raw, dict) else raw
    tokenized = [item["tokens"] for item in items if len(item.get("tokens", [])) >= 2]
    print(f"  로드: {path}")
    print(f"  유효 {len(tokenized)}건 / 평균 {np.mean([len(t) for t in tokenized]):.1f} 토큰")
    return tokenized


# =================================================================
# 3. 공통 유틸
# =================================================================

def compute_coherence(topic_words: list[list[str]],
                      tokenized: list[list[str]]) -> float | None:
    """
    Coherence C_v 계산 — Röder et al. (2015, WSDM)
    processes=1: 멀티프로세싱 비활성 → MemoryError 방지
    """
    try:
        from gensim.models.coherencemodel import CoherenceModel
        from gensim import corpora
        dictionary = corpora.Dictionary(tokenized)
        dictionary.filter_extremes(no_below=2, no_above=0.9)
        valid = [[w for w in words if w in dictionary.token2id]
                 for words in topic_words]
        valid = [t for t in valid if len(t) >= 3]
        if not valid:
            return None
        cm = CoherenceModel(
            topics=valid, texts=tokenized,
            dictionary=dictionary, coherence="c_v", processes=1
        )
        score = round(cm.get_coherence(), 4)
        print(f"  Coherence C_v: {score}")
        return score
    except Exception as e:
        print(f"  Coherence 실패: {e}")
        return None


def tokens_to_texts(tokenized: list[list[str]]) -> list[str]:
    return [" ".join(t) for t in tokenized]


def tfidf_vectorize(tokenized: list[list[str]], max_features: int) -> tuple:
    """
    TF-IDF 벡터화.
    max_features: DATASET_PARAMS["tfidf_feats"] 주입값 사용
      voc_en=200 (Sil=0.0476), vad_en=200 (Sil=0.0419)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    texts = tokens_to_texts(tokenized)
    vec = TfidfVectorizer(
        max_features=max_features,
        sublinear_tf=True,
        min_df=2,
        analyzer="word",
        token_pattern=r"\S+",
    )
    mat   = vec.fit_transform(texts)
    mat   = normalize(mat, norm="l2")
    dense = mat.toarray().astype(np.float32)
    feats = vec.get_feature_names_out()
    print(f"  TF-IDF 벡터: shape={dense.shape}  어휘={len(feats)}개  (max_features={max_features})")
    return dense, feats


def tfidf_top_words(vecs: np.ndarray, labels: np.ndarray,
                    feats: list, top_n: int = TOP_N_WORDS) -> list[list[str]]:
    """군집별 TF-IDF 상위 단어 추출"""
    cids = sorted(set(labels))
    result = []
    for cid in cids:
        mask = labels == cid
        center = vecs[mask].mean(axis=0)
        idx = center.argsort()[::-1][:top_n]
        result.append([feats[i] for i in idx if center[i] > 0])
    return result


def run_kmeans(vecs: np.ndarray, k: int, seeds: list = SEEDS) -> tuple:
    """
    KMeans K=k, N_RUNS 반복 → Silhouette 최고 run 선택.
    ※ k는 DATASET_PARAMS["k"] 주입 (voc_en=4, vad_en=6)
    MacQueen (1967)
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    best_labels, best_sil, all_sil = None, -1, []
    for seed in seeds:
        km  = KMeans(n_clusters=k, random_state=seed, n_init=10)
        lbl = km.fit_predict(vecs)
        try:
            sil = float(silhouette_score(vecs, lbl, metric="cosine"))
        except Exception:
            sil = -1.0
        all_sil.append(sil)
        if sil > best_sil:
            best_sil    = sil
            best_labels = lbl

    mean_sil = round(float(np.mean(all_sil)), 4)
    std_sil  = round(float(np.std(all_sil)),  4)
    print(f"  KMeans K={k} / Silhouette mean={mean_sil} std={std_sil}")
    return best_labels, mean_sil, std_sil


def save_result(result: dict, file_key: str, exp_id: str):
    dir_ = os.path.join(OUT_DIR, file_key)
    os.makedirs(dir_, exist_ok=True)
    path = os.path.join(dir_, f"{exp_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  저장: {path}")
    return path


# =================================================================
# 4. EXP-1 : SNA only — Louvain 커뮤니티 탐지
# =================================================================
def run_exp1_sna(tokenized: list[list[str]],
                 file_key: str, lang: str,
                 params: dict) -> dict:
    """
    공출현 네트워크 구축 → Louvain 커뮤니티 탐지.
    N_RUNS 반복 → 평균 Modularity 및 best Coherence 기록.

    Blondel et al. (2008): Louvain
    [그리드서치 최적]
      voc_en: window=7,  min_freq=5 → C_v=0.6108, kept_edges=1,817
      vad_en: window=10, min_freq=5 → C_v=0.7238, kept_edges=11,517
    """
    WINDOW         = params["window"]
    MIN_FREQ       = params["min_freq"]
    TOP_PAGERANK_N = params["top_pagerank_n"]

    sep = "=" * 55
    print(f"\n{sep}\n  EXP-1 · SNA only  [{file_key}]\n{sep}")
    print(f"  [파라미터] window={WINDOW}  min_freq={MIN_FREQ}  top_pagerank_n={TOP_PAGERANK_N}")

    print("\n[Step1] 공출현 네트워크 구성...")
    cooc = defaultdict(int)
    for tokens in tokenized:
        n = len(tokens)
        for i in range(n):
            for j in range(i + 1, min(i + WINDOW + 1, n)):
                pair = tuple(sorted([tokens[i], tokens[j]]))
                cooc[pair] += 1
    filtered = {k: v for k, v in cooc.items() if v >= MIN_FREQ}
    print(f"  공출현 쌍: 전체 {len(cooc):,} / 빈도≥{MIN_FREQ}: {len(filtered):,}")

    G = nx.Graph()
    for (w1, w2), freq in filtered.items():
        G.add_edge(w1, w2, weight=freq)
    print(f"  그래프: 노드 {G.number_of_nodes()} / 엣지 {G.number_of_edges()}")

    if G.number_of_nodes() < 4:
        print("  노드 부족 → 실험 스킵")
        return {}

    print(f"\n[Step2] Louvain {N_RUNS}회 반복...")
    pagerank    = nx.pagerank(G, weight="weight")
    run_records = []

    for seed in SEEDS:
        partition   = community_louvain.best_partition(G, weight="weight", random_state=seed)
        communities = defaultdict(list)
        for node, cid in partition.items():
            communities[cid].append(node)
        communities = dict(communities)
        modularity  = community_louvain.modularity(partition, G, weight="weight")

        topic_words = [
            sorted(words, key=lambda w: -pagerank.get(w, 0))[:TOP_N_WORDS]
            for words in communities.values()
        ]
        coherence = compute_coherence(topic_words, tokenized)

        run_records.append({
            "seed":          seed,
            "n_communities": len(communities),
            "modularity":    round(modularity, 4),
            "coherence_cv":  coherence,
            "communities":   communities,
            "topic_words":   topic_words,
        })

    scored = [(r["coherence_cv"] or 0) + r["modularity"] * 0.1 for r in run_records]
    best   = run_records[int(np.argmax(scored))]
    mods   = [r["modularity"] for r in run_records]
    cohs   = [r["coherence_cv"] for r in run_records if r["coherence_cv"] is not None]

    top_pagerank = [w for w, _ in sorted(pagerank.items(), key=lambda x: -x[1])[:TOP_PAGERANK_N]]

    result = {
        "exp_id":          "EXP-1",
        "method":          "SNA_LOUVAIN",
        "file":            file_key,
        "lang":            lang,
        "n_docs":          len(tokenized),
        "graph_nodes":     G.number_of_nodes(),
        "graph_edges":     G.number_of_edges(),
        "n_communities":   best["n_communities"],
        "modularity":      best["modularity"],
        "modularity_mean": round(float(np.mean(mods)), 4),
        "modularity_std":  round(float(np.std(mods)),  4),
        "coherence_cv":    best["coherence_cv"],
        "coherence_mean":  round(float(np.mean(cohs)), 4) if cohs else None,
        "top_pagerank_keywords": top_pagerank,
        "topic_words": {
            f"community_{i}": words
            for i, words in enumerate(best["topic_words"])
        },
        "used_params": {
            "window": WINDOW, "min_freq": MIN_FREQ, "top_pagerank_n": TOP_PAGERANK_N
        },
        "paper_ref": "Blondel et al. (2008) Louvain / Grid-Search Tuned",
    }
    print(f"\n  결과: 커뮤니티={best['n_communities']}  Modularity={best['modularity']}  C_v={best['coherence_cv']}")
    save_result(result, file_key, "EXP1_SNA")
    return result


# =================================================================
# 5. EXP-2 : LDA only — Gensim LDA
# =================================================================
def run_exp2_lda(tokenized: list[list[str]],
                 file_key: str, lang: str,
                 params: dict) -> dict:
    """
    Gensim LDA, K=params["k"], N_RUNS 반복.
    Blei, Ng & Jordan (2003) / Röder et al. (2015)

    [그리드서치 최적]
      voc_en: K=4, passes=30, no_below=2, no_above=0.9 → C_v=0.4413
      vad_en: K=6, passes=30, no_below=1, no_above=0.9 → C_v=0.5088
    """
    K            = params["k"]
    LDA_PASSES   = params["lda_passes"]
    LDA_NO_BELOW = params["lda_no_below"]
    LDA_NO_ABOVE = params["lda_no_above"]

    sep = "=" * 55
    print(f"\n{sep}\n  EXP-2 · LDA only  [{file_key}]\n{sep}")
    print(f"  [파라미터] K={K}  passes={LDA_PASSES}  no_below={LDA_NO_BELOW}  no_above={LDA_NO_ABOVE}")

    from gensim import corpora
    from gensim.models import LdaModel

    print("\n[Step1] Gensim corpus 구성...")
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=LDA_NO_BELOW, no_above=LDA_NO_ABOVE)
    corpus = [dictionary.doc2bow(t) for t in tokenized]
    print(f"  어휘: {len(dictionary)}개  문서: {len(corpus)}건")

    if len(dictionary) < K:
        print("  어휘 부족 → 실험 스킵")
        return {}

    print(f"\n[Step2] LDA K={K}, {N_RUNS}회 반복...")
    run_records = []
    for seed in SEEDS:
        lda = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=K,
            random_state=seed,
            passes=LDA_PASSES,
            alpha="auto",
            eta="auto",
            per_word_topics=False,
        )
        topic_words = [
            [w for w, _ in lda.show_topic(t, topn=TOP_N_WORDS)]
            for t in range(K)
        ]
        coherence  = compute_coherence(topic_words, tokenized)
        perplexity = lda.log_perplexity(corpus)

        run_records.append({
            "seed":        seed,
            "lda":         lda,
            "topic_words": topic_words,
            "coherence_cv": coherence,
            "perplexity":   round(float(perplexity), 4),
        })

    scored = [r["coherence_cv"] or 0 for r in run_records]
    best   = run_records[int(np.argmax(scored))]
    cohs   = [r["coherence_cv"] for r in run_records if r["coherence_cv"] is not None]
    perps  = [r["perplexity"] for r in run_records]

    # 문서별 토픽 분포 벡터 (EXP-5/7 에서 재사용)
    best_lda   = best["lda"]
    topic_vecs = np.zeros((len(corpus), K), dtype=np.float32)
    for i, bow in enumerate(corpus):
        dist = dict(best_lda.get_document_topics(bow, minimum_probability=0.0))
        for t in range(K):
            topic_vecs[i, t] = dist.get(t, 0.0)

    result = {
        "exp_id":          "EXP-2",
        "method":          "LDA",
        "file":            file_key,
        "lang":            lang,
        "n_docs":          len(tokenized),
        "n_topics":        K,
        "coherence_cv":    best["coherence_cv"],
        "coherence_mean":  round(float(np.mean(cohs)),  4) if cohs else None,
        "coherence_std":   round(float(np.std(cohs)),   4) if cohs else None,
        "perplexity":      best["perplexity"],
        "perplexity_mean": round(float(np.mean(perps)), 4),
        "topic_words": {
            f"topic_{i}": words
            for i, words in enumerate(best["topic_words"])
        },
        "used_params": {
            "k": K, "passes": LDA_PASSES, "no_below": LDA_NO_BELOW, "no_above": LDA_NO_ABOVE
        },
        "paper_ref": "Blei, Ng & Jordan (2003) / Röder et al. (2015) / Grid-Search Tuned",
    }
    print(f"\n  결과: K={K}  C_v={best['coherence_cv']}  Perplexity={best['perplexity']}")
    save_result(result, file_key, "EXP2_LDA")

    return result, topic_vecs, best_lda, corpus, dictionary


# =================================================================
# 6. EXP-3 : KMeans only — TF-IDF → KMeans
# =================================================================
def run_exp3_kmeans(tokenized: list[list[str]],
                    file_key: str, lang: str,
                    params: dict) -> dict:
    """
    TF-IDF 벡터화 → KMeans K=params["k"].
    MacQueen (1967)

    [그리드서치 최적]
      voc_en: K=4, max_features=200 → Silhouette=0.0476
      vad_en: K=6, max_features=200 → Silhouette=0.0419
    ※ 어휘수 줄일수록 Silhouette 상승 (고차원 희소성 감소 효과)
    """
    K           = params["k"]
    TFIDF_FEATS = params["tfidf_feats"]

    sep = "=" * 55
    print(f"\n{sep}\n  EXP-3 · KMeans only  [{file_key}]\n{sep}")
    print(f"  [파라미터] K={K}  tfidf_feats={TFIDF_FEATS}")

    print("\n[Step1] TF-IDF 벡터화...")
    vecs, feats = tfidf_vectorize(tokenized, max_features=TFIDF_FEATS)

    print(f"\n[Step2] KMeans K={K}, {N_RUNS}회 반복...")
    best_labels, mean_sil, std_sil = run_kmeans(vecs, k=K)

    topic_words  = tfidf_top_words(vecs, best_labels, feats, TOP_N_WORDS)
    coherence    = compute_coherence(topic_words, tokenized)
    cluster_dist = Counter(int(l) for l in best_labels)

    result = {
        "exp_id":          "EXP-3",
        "method":          "KMEANS_TFIDF",
        "file":            file_key,
        "lang":            lang,
        "n_docs":          len(tokenized),
        "k":               K,
        "silhouette_mean": mean_sil,
        "silhouette_std":  std_sil,
        "coherence_cv":    coherence,
        "cluster_distribution": {str(k): v for k, v in sorted(cluster_dist.items())},
        "topic_words": {
            f"cluster_{i}": words
            for i, words in enumerate(topic_words)
        },
        "used_params": {"k": K, "tfidf_feats": TFIDF_FEATS},
        "paper_ref": "MacQueen (1967) / Grid-Search Tuned",
    }
    print(f"\n  결과: K={K}  Silhouette={mean_sil}  C_v={coherence}")
    print(f"  군집 분포: {dict(sorted(cluster_dist.items()))}")
    save_result(result, file_key, "EXP3_KMeans")
    return result


# =================================================================
# 7. EXP-4 : SNA → LDA — 커뮤니티 내 LDA
# =================================================================
def run_exp4_sna_lda(tokenized: list[list[str]],
                     sna_result: dict,
                     file_key: str, lang: str,
                     params: dict) -> dict:
    """
    EXP-1 SNA 커뮤니티별 문서 분리 후 각 커뮤니티 내 LDA 실행.
    토픽 세분화 효과 검증.

    Austin et al. (COLING 2022): SNA 커뮤니티 기반 토픽 세분화
    [그리드서치 최적] lda_passes, no_below, no_above → params 주입
    """
    LDA_PASSES   = params["lda_passes"]
    LDA_NO_BELOW = params["lda_no_below"]
    LDA_NO_ABOVE = params["lda_no_above"]

    sep = "=" * 55
    print(f"\n{sep}\n  EXP-4 · SNA → LDA  [{file_key}]\n{sep}")
    print(f"  [파라미터] lda_passes={LDA_PASSES}  no_below={LDA_NO_BELOW}")

    if not sna_result:
        print("  EXP-1 결과 없음 → 스킵")
        return {}

    from gensim import corpora
    from gensim.models import LdaModel

    print("\n[Step1] 커뮤니티별 문서 배정 (토큰 다수결)...")
    word_to_comm = {}
    for cid_str, words in sna_result.get("topic_words", {}).items():
        cid = int(cid_str.split("_")[-1])
        for w in words:
            word_to_comm[w] = cid

    doc_comm_labels = []
    for tokens in tokenized:
        votes = Counter(word_to_comm[w] for w in tokens if w in word_to_comm)
        label = votes.most_common(1)[0][0] if votes else 0
        doc_comm_labels.append(label)

    comm_doc_map = defaultdict(list)
    for idx, label in enumerate(doc_comm_labels):
        comm_doc_map[label].append(tokenized[idx])

    print(f"  커뮤니티 수: {len(comm_doc_map)}  문서 배정: {len(doc_comm_labels)}건")

    print(f"\n[Step2] 커뮤니티별 LDA (passes={LDA_PASSES})...")
    all_topic_words  = []
    community_results = {}

    for cid, docs in sorted(comm_doc_map.items()):
        if len(docs) < 5:
            print(f"  Community {cid}: 문서 {len(docs)}건 — 스킵 (< 5건)")
            continue
        k_inner    = max(1, min(2, len(docs) // 30))
        dictionary = corpora.Dictionary(docs)
        dictionary.filter_extremes(no_below=LDA_NO_BELOW, no_above=LDA_NO_ABOVE)
        if len(dictionary) < k_inner:
            continue
        corpus_c = [dictionary.doc2bow(t) for t in docs]
        try:
            lda = LdaModel(corpus=corpus_c, id2word=dictionary,
                           num_topics=k_inner, random_state=42,
                           passes=LDA_PASSES, alpha="auto", eta="auto")
            words = [w for w, _ in lda.show_topic(0, topn=TOP_N_WORDS)]
            all_topic_words.append(words)
            community_results[f"community_{cid}"] = {
                "n_docs":      len(docs),
                "k_inner":     k_inner,
                "topic_words": words,
            }
            print(f"  Community {cid}: {len(docs)}건  K={k_inner}  top: {words[:5]}")
        except Exception as e:
            print(f"  Community {cid}: LDA 실패 ({e})")

    coherence = compute_coherence(all_topic_words, tokenized) if all_topic_words else None

    result = {
        "exp_id":                 "EXP-4",
        "method":                 "SNA_LDA",
        "file":                   file_key,
        "lang":                   lang,
        "n_docs":                 len(tokenized),
        "n_communities":          sna_result.get("n_communities"),
        "n_communities_with_lda": len(community_results),
        "coherence_cv":           coherence,
        "community_results":      community_results,
        "used_params": {
            "lda_passes": LDA_PASSES, "lda_no_below": LDA_NO_BELOW, "lda_no_above": LDA_NO_ABOVE
        },
        "paper_ref": "Austin et al. (COLING 2022) / Blei et al. (2003) / Grid-Search Tuned",
    }
    print(f"\n  결과: 유효 커뮤니티={len(community_results)}  C_v={coherence}")
    save_result(result, file_key, "EXP4_SNA_LDA")
    return result


# =================================================================
# 8. EXP-5 : LDA → KMeans — 토픽 분포 벡터 → 군집화
# =================================================================
def run_exp5_lda_kmeans(tokenized: list[list[str]],
                        topic_vecs: np.ndarray,
                        lda_result: dict,
                        file_key: str, lang: str,
                        params: dict) -> dict:
    """
    LDA 문서별 토픽 분포 벡터(K차원) → KMeans 군집화.
    ※ KMeans 입력이 LDA 토픽벡터(K차원)이므로 tfidf_feats 미적용

    Scientific Reports (2025): LDA 토픽벡터 → KMeans 페르소나 생성 실증
    Blei et al. (2003) + MacQueen (1967)
    """
    K = params["k"]

    sep = "=" * 55
    print(f"\n{sep}\n  EXP-5 · LDA → KMeans  [{file_key}]\n{sep}")
    print(f"  [파라미터] K={K} (LDA 토픽벡터 차원, tfidf_feats 미적용)")

    if topic_vecs is None or len(topic_vecs) == 0:
        print("  토픽 벡터 없음 → 스킵")
        return {}

    print(f"\n[Step1] 토픽 벡터 확인: shape={topic_vecs.shape}")
    print(f"\n[Step2] KMeans K={K}, {N_RUNS}회 반복...")
    best_labels, mean_sil, std_sil = run_kmeans(topic_vecs, k=K)

    cluster_topic_dist = {}
    for cid in range(K):
        mask = best_labels == cid
        if mask.sum() > 0:
            mean_dist      = topic_vecs[mask].mean(axis=0)
            dominant_topic = int(mean_dist.argmax())
            cluster_topic_dist[f"cluster_{cid}"] = {
                "n_docs":         int(mask.sum()),
                "dominant_topic": dominant_topic,
                "topic_dist":     [round(float(v), 4) for v in mean_dist],
            }

    lda_topic_words = list(lda_result.get("topic_words", {}).values())
    coherence       = compute_coherence(lda_topic_words, tokenized) if lda_topic_words else None
    cluster_dist    = Counter(int(l) for l in best_labels)

    result = {
        "exp_id":              "EXP-5",
        "method":              "LDA_KMEANS",
        "file":                file_key,
        "lang":                lang,
        "n_docs":              len(tokenized),
        "k":                   K,
        "silhouette_mean":     mean_sil,
        "silhouette_std":      std_sil,
        "coherence_cv":        coherence,
        "cluster_distribution": {str(k): v for k, v in sorted(cluster_dist.items())},
        "cluster_topic_dist":   cluster_topic_dist,
        "lda_topic_words":      lda_result.get("topic_words", {}),
        "used_params":          {"k": K},
        "paper_ref": "Scientific Reports(2025) / Blei(2003) / MacQueen(1967) / Grid-Search Tuned",
    }
    print(f"\n  결과: K={K}  Silhouette={mean_sil}  C_v={coherence}")
    print(f"  군집 분포: {dict(sorted(cluster_dist.items()))}")
    for cid, info in cluster_topic_dist.items():
        print(f"  {cid}: {info['n_docs']}건  dominant_topic={info['dominant_topic']}"
              f"  dist={info['topic_dist']}")
    save_result(result, file_key, "EXP5_LDA_KMeans")
    return result


# =================================================================
# 9. EXP-6 : SNA → KMeans — PageRank 필터 → KMeans
# =================================================================
def run_exp6_sna_kmeans(tokenized: list[list[str]],
                        sna_result: dict,
                        file_key: str, lang: str,
                        params: dict) -> dict:
    """
    SNA PageRank 상위 키워드로 어휘 필터링 → TF-IDF → KMeans.
    노이즈 토큰 사전 제거 효과 검증.

    Blondel et al. (2008) + MacQueen (1967)
    [그리드서치 최적]
      voc_en: K=4, tfidf_feats=200 → Sil=0.0476
      vad_en: K=6, tfidf_feats=200 → Sil=0.0419
    """
    K           = params["k"]
    TFIDF_FEATS = params["tfidf_feats"]

    sep = "=" * 55
    print(f"\n{sep}\n  EXP-6 · SNA → KMeans  [{file_key}]\n{sep}")
    print(f"  [파라미터] K={K}  tfidf_feats={TFIDF_FEATS}")

    if not sna_result:
        print("  EXP-1 결과 없음 → 스킵")
        return {}

    top_keywords = set(sna_result.get("top_pagerank_keywords", []))
    print(f"\n[Step1] PageRank 필터 키워드: {len(top_keywords)}개")
    print(f"  샘플: {list(top_keywords)[:10]}")

    filtered_tokenized = [
        [w for w in tokens if w in top_keywords]
        for tokens in tokenized
    ]
    valid = [(ft, ot) for ft, ot in zip(filtered_tokenized, tokenized) if len(ft) >= 2]
    if not valid:
        print("  필터 후 유효 문서 없음 → 원본 토큰으로 fallback")
        filtered_tokenized = tokenized
    else:
        filtered_tokenized, _ = zip(*valid)
        filtered_tokenized = list(filtered_tokenized)
    print(f"  필터 후 유효 문서: {len(filtered_tokenized)}건")

    print(f"\n[Step2] TF-IDF 벡터화 (max_features={TFIDF_FEATS})...")
    vecs, feats = tfidf_vectorize(filtered_tokenized, max_features=TFIDF_FEATS)

    print(f"\n[Step3] KMeans K={K}, {N_RUNS}회 반복...")
    best_labels, mean_sil, std_sil = run_kmeans(vecs, k=K)

    topic_words  = tfidf_top_words(vecs, best_labels, feats, TOP_N_WORDS)
    coherence    = compute_coherence(topic_words, filtered_tokenized)
    cluster_dist = Counter(int(l) for l in best_labels)

    result = {
        "exp_id":              "EXP-6",
        "method":              "SNA_KMEANS",
        "file":                file_key,
        "lang":                lang,
        "n_docs":              len(filtered_tokenized),
        "k":                   K,
        "sna_filter_keywords": list(top_keywords),
        "silhouette_mean":     mean_sil,
        "silhouette_std":      std_sil,
        "coherence_cv":        coherence,
        "cluster_distribution": {str(k): v for k, v in sorted(cluster_dist.items())},
        "topic_words": {
            f"cluster_{i}": words
            for i, words in enumerate(topic_words)
        },
        "used_params": {"k": K, "tfidf_feats": TFIDF_FEATS},
        "paper_ref": "Blondel et al. (2008) + MacQueen (1967) / Grid-Search Tuned",
    }
    print(f"\n  결과: K={K}  Silhouette={mean_sil}  C_v={coherence}")
    save_result(result, file_key, "EXP6_SNA_KMeans")
    return result


# =================================================================
# 10. EXP-7 : SNA → LDA → KMeans — 전체 파이프라인
# =================================================================
def run_exp7_full(tokenized: list[list[str]],
                  sna_result: dict,
                  topic_vecs: np.ndarray,
                  lda_result: dict,
                  file_key: str, lang: str,
                  params: dict) -> dict:
    """
    SNA PageRank 필터 → LDA 토픽 벡터 → KMeans.
    EXP-5 대비 SNA 선처리 추가 효과 측정 — 논문 핵심 contribution.

    [그리드서치 최적] 모든 파라미터 params에서 주입
    """
    K            = params["k"]
    LDA_PASSES   = params["lda_passes"]
    LDA_NO_BELOW = params["lda_no_below"]
    LDA_NO_ABOVE = params["lda_no_above"]

    sep = "=" * 55
    print(f"\n{sep}\n  EXP-7 · SNA → LDA → KMeans (전체)  [{file_key}]\n{sep}")
    print(f"  [파라미터] K={K}  lda_passes={LDA_PASSES}  no_below={LDA_NO_BELOW}")

    if not sna_result or topic_vecs is None:
        print("  선행 실험 결과 없음 → 스킵")
        return {}

    top_keywords = set(sna_result.get("top_pagerank_keywords", []))
    filtered_tokenized = [
        [w for w in tokens if w in top_keywords]
        for tokens in tokenized
    ]
    valid_filtered = [t for t in filtered_tokenized if len(t) >= 2]

    if len(valid_filtered) < K * 5:
        print("  SNA 필터 후 문서 부족 → EXP-5 토픽 벡터 그대로 사용")
        filtered_vecs        = topic_vecs
        sna_filter_topic_words = list(lda_result.get("topic_words", {}).values())
    else:
        from gensim import corpora
        from gensim.models import LdaModel
        print(f"\n[Step1] SNA 필터 후 LDA 재실행 (유효 {len(valid_filtered)}건, passes={LDA_PASSES})...")
        dictionary = corpora.Dictionary(valid_filtered)
        dictionary.filter_extremes(no_below=LDA_NO_BELOW, no_above=LDA_NO_ABOVE)
        corpus = [dictionary.doc2bow(t) for t in valid_filtered]
        lda    = LdaModel(corpus=corpus, id2word=dictionary, num_topics=K,
                          random_state=42, passes=LDA_PASSES, alpha="auto", eta="auto")

        # SNA 필터 통과 문서만 벡터 생성 (모집단 일치)
        full_corpus   = [dictionary.doc2bow(t) for t in valid_filtered]
        filtered_vecs = np.zeros((len(valid_filtered), K), dtype=np.float32)
        for i, bow in enumerate(full_corpus):
            dist = dict(lda.get_document_topics(bow, minimum_probability=0.0))
            for t in range(K):
                filtered_vecs[i, t] = dist.get(t, 0.0)

        sna_filter_topic_words = [
            [w for w, _ in lda.show_topic(t, topn=TOP_N_WORDS)]
            for t in range(K)
        ]
        print(f"  SNA 필터 LDA 토픽 키워드:")
        for i, words in enumerate(sna_filter_topic_words):
            print(f"    Topic {i}: {words[:6]}")

    print(f"\n[Step2] KMeans K={K}, {N_RUNS}회 반복 (필터된 토픽 벡터)...")
    best_labels, mean_sil, std_sil = run_kmeans(filtered_vecs, k=K)

    cluster_topic_dist = {}
    for cid in range(K):
        mask = best_labels == cid
        if mask.sum() > 0:
            mean_dist = filtered_vecs[mask].mean(axis=0)
            cluster_topic_dist[f"cluster_{cid}"] = {
                "n_docs":         int(mask.sum()),
                "dominant_topic": int(mean_dist.argmax()),
                "topic_dist":     [round(float(v), 4) for v in mean_dist],
            }

    lda_topic_words = list(lda_result.get("topic_words", {}).values())
    coherence       = compute_coherence(lda_topic_words, valid_filtered)
    cluster_dist    = Counter(int(l) for l in best_labels)

    result = {
        "exp_id":              "EXP-7",
        "method":              "SNA_LDA_KMEANS",
        "file":                file_key,
        "lang":                lang,
        "n_docs":              len(valid_filtered),
        "n_docs_original":     len(tokenized),
        "k":                   K,
        "sna_filter_keywords": list(top_keywords),
        "silhouette_mean":     mean_sil,
        "silhouette_std":      std_sil,
        "coherence_cv":        coherence,
        "cluster_distribution": {str(k): v for k, v in sorted(cluster_dist.items())},
        "cluster_topic_dist":   cluster_topic_dist,
        "used_params": {
            "k": K, "lda_passes": LDA_PASSES,
            "lda_no_below": LDA_NO_BELOW, "lda_no_above": LDA_NO_ABOVE
        },
        "paper_ref": "Scientific Reports(2025) / Blondel(2008) / Blei(2003) / MacQueen(1967) / Grid-Search Tuned",
    }
    print(f"\n  결과: K={K}  Silhouette={mean_sil}  C_v={coherence}")
    save_result(result, file_key, "EXP7_Full")
    return result


# =================================================================
# 11. 비교 요약 출력
# =================================================================
def print_summary(all_results: dict):
    sep = "=" * 70
    print(f"\n{sep}")
    print("  실험 결과 요약")
    print(sep)

    for file_key, exps in all_results.items():
        k_used = DATASET_PARAMS[file_key]["k"]
        print(f"\n  ── {file_key}  (K={k_used}) ──")
        print(f"  {'실험':<22} {'C_v':>8} {'Modularity':>12} {'Silhouette':>12}")
        print(f"  {'-'*56}")
        for exp_id, res in sorted(exps.items()):
            if not res:
                continue
            cv  = res.get("coherence_cv")
            mod = res.get("modularity")
            sil = res.get("silhouette_mean")
            print(f"  {res.get('method',''):<22}"
                  f" {str(cv) if cv else '-':>8}"
                  f" {str(mod) if mod else '-':>12}"
                  f" {str(sil) if sil else '-':>12}")
    print(f"\n{sep}")


def save_summary(all_results: dict):
    summary = {}
    for file_key, exps in all_results.items():
        summary[file_key] = {"k_used": DATASET_PARAMS[file_key]["k"]}
        for exp_id, res in exps.items():
            if not res:
                continue
            summary[file_key][exp_id] = {
                "method":       res.get("method"),
                "k":            res.get("k") or res.get("n_topics"),
                "coherence_cv": res.get("coherence_cv"),
                "modularity":   res.get("modularity"),
                "silhouette":   res.get("silhouette_mean"),
                "n_docs":       res.get("n_docs"),
            }
    path = os.path.join(OUT_DIR, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  요약 저장: {path}")


# =================================================================
# 12. 메인
# =================================================================
if __name__ == "__main__":
    SEP = "=" * 70
    print(f"\n{SEP}")
    print("  7가지 조합 실험 파이프라인 시작  [Grid-Search Tuned]")
    print(f"  N_RUNS={N_RUNS}  TOP_N_WORDS={TOP_N_WORDS}")
    print(f"\n  ★ 데이터셋별 최적 파라미터 (그리드서치 결과 반영):")
    for dk, dp in DATASET_PARAMS.items():
        print(f"    [{dk}]  K={dp['k']}  window={dp['window']}  min_freq={dp['min_freq']}"
              f"  top_pr={dp['top_pagerank_n']}  passes={dp['lda_passes']}"
              f"  no_below={dp['lda_no_below']}  tfidf_feats={dp['tfidf_feats']}")
    print(f"  ※ voc_en K=4: Top-Down 도메인 설계 (Salminen et al. 2022)")
    print(f"  ※ vad_en K=6: Bottom-Up 귀납 탐색 (STEP6 LDA C_v 최고: 0.5323)")
    print(f"{SEP}\n")

    LANG_MAP = {"voc_en": "en", "vad_en": "en"}
    all_results = {}

    for file_key, path in INPUT_FILES.items():
        if not os.path.exists(path):
            print(f"\n[SKIP] 파일 없음: {path}")
            continue
        if file_key not in DATASET_PARAMS:
            print(f"\n[SKIP] DATASET_PARAMS에 '{file_key}' 없음 → 추가 후 재실행")
            continue

        params = DATASET_PARAMS[file_key]
        lang   = LANG_MAP[file_key]

        print(f"\n{'#'*70}")
        print(f"  파일: {file_key}  언어: {lang}  K={params['k']}")
        print(f"  파라미터: {params}")
        print(f"{'#'*70}")

        tokenized = load_preprocessed(path)
        if len(tokenized) < 10:
            print(f"  문서 부족 ({len(tokenized)}건) → 스킵")
            continue

        exps = {}

        # EXP-1: SNA only
        exps["EXP-1"] = run_exp1_sna(tokenized, file_key, lang, params)

        # EXP-2: LDA only (토픽 벡터도 반환)
        lda_return = run_exp2_lda(tokenized, file_key, lang, params)
        if isinstance(lda_return, tuple):
            lda_result, topic_vecs, best_lda, corpus, dictionary = lda_return
        else:
            lda_result, topic_vecs = lda_return, None
        exps["EXP-2"] = lda_result

        # EXP-3: KMeans only
        exps["EXP-3"] = run_exp3_kmeans(tokenized, file_key, lang, params)

        # EXP-4: SNA → LDA
        exps["EXP-4"] = run_exp4_sna_lda(tokenized, exps["EXP-1"], file_key, lang, params)

        # EXP-5: LDA → KMeans (LDA 토픽벡터 사용, tfidf_feats 미적용)
        exps["EXP-5"] = run_exp5_lda_kmeans(
            tokenized, topic_vecs, lda_result, file_key, lang, params
        )

        # EXP-6: SNA → KMeans
        exps["EXP-6"] = run_exp6_sna_kmeans(tokenized, exps["EXP-1"], file_key, lang, params)

        # EXP-7: SNA → LDA → KMeans (전체)
        exps["EXP-7"] = run_exp7_full(
            tokenized, exps["EXP-1"], topic_vecs, lda_result, file_key, lang, params
        )

        all_results[file_key] = exps

    print_summary(all_results)
    save_summary(all_results)

    print(f"\n{SEP}")
    print(f"  완료  —  결과 디렉토리: {os.path.abspath(OUT_DIR)}")
    print(f"{SEP}\n")