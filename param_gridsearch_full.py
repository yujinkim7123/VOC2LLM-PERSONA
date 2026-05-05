"""
=================================================================
파라미터 그리디 서치 — 전체 데이터 기준
VOC2Persona-LLM 연구 | 파라미터 수학적 근거 마련용
=================================================================
대상 데이터:
  - voc_en : merged_voc.csv             (965건)
  - vad_en : Virtual_Assistant_Devices_Dataset__Final_Version.xlsx (2,370건)

탐색 파라미터:
  - SNA    : WINDOW x MIN_FREQ           -> Coherence C_v
  - KMeans : TFIDF_FEATS                 -> Silhouette Score
  - PageRank: Top-N                      -> 문서 커버리지
  - LDA    : passes / no_below / no_above -> Coherence C_v + Perplexity
  - N_RUNS : 반복 횟수                   -> Coherence std 수렴
  ★ [신규] VAD K 탐색 (STEP 6):
  - vad_en 전용 : LDA k ∈ {2~10}        -> Coherence C_v + Perplexity
                  KMeans k ∈ {2~10}      -> Silhouette Score
  - voc_en : K=4 고정 (Top-Down 도메인 설계 결정, Salminen et al. 2022)
  - vad_en : K를 데이터에서 귀납적으로 탐색 (Bottom-Up)

전처리 방식 (preprocess_en.py 동일):
  - voc_en: S1 -> review(EN 원문), S2~S4 -> review_translation(EN 번역)
  - vad_en: Description + Unnamed:4 합치기 (병합셀 ffill 적용)
  - 토크나이저: spaCy lemmatization (미설치 시 regex fallback)
  - 불용어: NLTK stopwords (미설치 시 내장 fallback)

실행 방법:
  python param_gridsearch_full.py

필요 패키지:
  pip install pandas openpyxl numpy networkx python-louvain gensim scikit-learn
  pip install spacy && python -m spacy download en_core_web_sm
=================================================================
"""

import re
import json
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict

import networkx as nx
import community as community_louvain

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora

warnings.filterwarnings("ignore")


# =================================================================
# 0. 경로 설정 <- 필요 시 수정
# =================================================================
VOC_PATH = r""
VAD_PATH = r""
OUT_PATH = r"param_gridsearch_result.json"


# =================================================================
# 1. 불용어 설정 (preprocess_en.py 동일)
# =================================================================

def get_en_stopwords():
    """NLTK 불용어 로드. 미설치 시 내장 fallback."""
    try:
        import nltk
        try:
            from nltk.corpus import stopwords
            return set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            return set(stopwords.words("english"))
    except ImportError:
        pass
    return set([
        "i","me","my","myself","we","our","ours","ourselves","you","your",
        "yours","yourself","yourselves","he","him","his","himself","she",
        "her","hers","herself","it","its","itself","they","them","their",
        "theirs","themselves","what","which","who","whom","this","that",
        "these","those","am","is","are","was","were","be","been","being",
        "have","has","had","having","do","does","did","doing","a","an",
        "the","and","but","if","or","because","as","until","while","of",
        "at","by","for","with","about","against","between","into",
        "through","during","before","after","above","below","to","from",
        "up","down","in","out","on","off","over","under","again","further",
        "then","once","here","there","when","where","why","how","all",
        "both","each","few","more","most","other","some","such","no","nor",
        "not","only","own","same","so","than","too","very","s","t","can",
        "will","just","don","should","now","d","ll","m","o","re","ve","y",
        "also","get","got","like","would","could","even","still","really",
        "one","two","three","go","going","make","made","use","used",
        "good","well","great","much","many","little","bit",
    ])

EN_STOPS = get_en_stopwords()


# =================================================================
# 2. spaCy 싱글톤 로더 (preprocess_en.py 동일)
# =================================================================
_spacy_nlp    = None
_spacy_loaded = False

def get_spacy_nlp():
    global _spacy_nlp, _spacy_loaded
    if not _spacy_loaded:
        _spacy_loaded = True
        try:
            import spacy
            print("  [spaCy] en_core_web_sm 로딩 중...", flush=True)
            _spacy_nlp = spacy.load("en_core_web_sm",
                                    disable=["parser", "ner"])
            print("  [spaCy] 로딩 완료", flush=True)
        except (OSError, ImportError):
            print("  [경고] en_core_web_sm 미설치 -> regex fallback 사용")
            print("         설치: python -m spacy download en_core_web_sm")
            _spacy_nlp = None
    return _spacy_nlp


# =================================================================
# 3. 토크나이저 (preprocess_en.py 동일)
# =================================================================

def tokenize_en(text):
    """
    spaCy lemmatization -> 불용어/구두점/숫자 제거 -> len > 2 필터.
    spaCy 미설치 시 regex fallback.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    nlp = get_spacy_nlp()

    if nlp is not None:
        doc = nlp(text.lower())
        return [
            token.lemma_
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and not token.is_space
            and len(token.lemma_) > 2
            and token.lemma_.isalpha()
        ]
    else:
        text_clean = re.sub(r"[^a-z\s]", " ", text.lower())
        return [
            t for t in text_clean.split()
            if len(t) > 2 and t not in EN_STOPS
        ]


# =================================================================
# 4. 데이터 로드 (preprocess_en.py 동일 로직)
# =================================================================

def load_voc_en(path):
    """
    merged_voc.csv 로드.
    S1  -> review (EN 원문)
    S2~S4 -> review_translation (EN 번역)
    """
    print("  [voc_en] merged_voc.csv 로드 중...")
    df = pd.read_csv(path, encoding="utf-8")
    print(f"  [voc_en] 전체 {len(df):,}건")

    texts = []
    for _, row in df.iterrows():
        if row["segment"] == "S1":
            texts.append(str(row["review"]))
        else:
            texts.append(str(row["review_translation"]))

    tokens_list = [tokenize_en(t) for t in texts]
    tokens_list = [t for t in tokens_list if len(t) >= 2]
    print(f"  [voc_en] 유효 {len(tokens_list):,}건 | "
          f"평균 {np.mean([len(t) for t in tokens_list]):.1f} 토큰")
    return tokens_list


def load_vad_en(path):
    """
    VAD_dataset.xlsx 로드.
    Description(제목) + Unnamed:4(본문) 합치기.
    병합셀 구조 -> Device Name / Company / Source ffill 적용.
    """
    print("  [vad_en] VAD_dataset.xlsx 로드 중...")
    df = pd.read_excel(path)
    print(f"  [vad_en] 전체 {len(df):,}건")

    for col in ["Device Name", "Company", "Source"]:
        if col in df.columns:
            df[col] = df[col].ffill()

    df["full_text"] = (
        df["Description"].fillna("").astype(str).str.strip()
        + " "
        + df["Unnamed: 4"].fillna("").astype(str).str.strip()
    ).str.strip()

    tokens_list = [tokenize_en(t) for t in df["full_text"]]
    tokens_list = [t for t in tokens_list if len(t) >= 2]
    print(f"  [vad_en] 유효 {len(tokens_list):,}건 | "
          f"평균 {np.mean([len(t) for t in tokens_list]):.1f} 토큰")
    return tokens_list


def load_data():
    print("=" * 70)
    print("  데이터 로드 및 전처리 (preprocess_en.py 동일 로직)")
    print("=" * 70)
    voc_tokens = load_voc_en(VOC_PATH)
    vad_tokens = load_vad_en(VAD_PATH)
    return voc_tokens, vad_tokens


# =================================================================
# 5. 공통 유틸 — Coherence C_v
# =================================================================

def compute_coherence(topic_words, tokenized):
    """
    Coherence C_v 계산 — Röder et al. (2015, WSDM)
    값이 높을수록 토픽의 의미론적 일관성이 높음 (0.0~1.0)
    """
    try:
        dictionary = corpora.Dictionary(tokenized)
        dictionary.filter_extremes(no_below=2, no_above=0.9)
        valid = [
            [w for w in words if w in dictionary.token2id]
            for words in topic_words
        ]
        valid = [t for t in valid if len(t) >= 3]
        if not valid:
            return None
        cm = CoherenceModel(
            topics=valid, texts=tokenized,
            dictionary=dictionary, coherence="c_v",
            processes=1
        )
        return round(cm.get_coherence(), 4)
    except Exception:
        return None


# =================================================================
# 6. SNA 그리디 서치 — WINDOW × MIN_FREQ
# =================================================================

def run_sna(tokenized, window, min_freq, top_n=10):
    """
    공출현 네트워크 구축 → Louvain 커뮤니티 탐지 →
    PageRank 키워드 추출 → Coherence C_v 평가

    Parameters
    ----------
    window   : 슬라이딩 윈도우 크기 (공출현 범위)
    min_freq : 최소 공출현 빈도 (엣지 필터 임계값)
    top_n    : 커뮤니티별 상위 키워드 수

    Returns
    -------
    coherence, total_edges, kept_edges, n_communities
    """
    cooc = defaultdict(int)
    for tokens in tokenized:
        n = len(tokens)
        for i in range(n):
            for j in range(i + 1, min(i + window + 1, n)):
                pair = tuple(sorted([tokens[i], tokens[j]]))
                cooc[pair] += 1

    total_edges = len(cooc)
    filtered    = {k: v for k, v in cooc.items() if v >= min_freq}
    kept_edges  = len(filtered)

    if not filtered:
        return None, total_edges, 0, 0

    G = nx.Graph()
    for (w1, w2), freq in filtered.items():
        G.add_edge(w1, w2, weight=freq)

    if G.number_of_nodes() < 4:
        return None, total_edges, kept_edges, 0

    pagerank  = nx.pagerank(G, weight='weight')
    partition = community_louvain.best_partition(
        G, weight='weight', random_state=42
    )

    communities = defaultdict(list)
    for node, cid in partition.items():
        communities[cid].append(node)
    n_communities = len(communities)

    topic_words = [
        sorted(nodes, key=lambda w: -pagerank.get(w, 0))[:top_n]
        for nodes in communities.values()
    ]

    coherence = compute_coherence(topic_words, tokenized)
    return coherence, total_edges, kept_edges, n_communities


def sna_grid_search(datasets):
    """WINDOW × MIN_FREQ 전체 조합 탐색"""
    WINDOWS   = [2, 3, 5, 7, 10]
    MIN_FREQS = [2, 3, 5, 7, 10]

    SEP = "=" * 75
    print(f"\n{SEP}")
    print("  STEP 1 / 6   SNA 그리디 서치")
    print("  탐색: WINDOW ∈ {2,3,5,7,10}  ×  MIN_FREQ ∈ {2,3,5,7,10}")
    print("  평가: Coherence C_v  (높을수록 의미론적 일관성 높음)")
    print(f"{SEP}")

    all_results = {}

    for dsname, tokenized in datasets.items():
        print(f"\n  ── {dsname}  ({len(tokenized):,}건) ──")
        print(f"  {'WINDOW':>8} {'MIN_FREQ':>10} {'Coherence_Cv':>14}"
              f" {'잔존엣지':>10} {'총엣지':>10} {'커뮤니티수':>12}")
        print(f"  {'─'*68}")

        rows = []
        best_coh, best_cfg = -1, None

        for w in WINDOWS:
            for mf in MIN_FREQS:
                coh, total, kept, n_comm = run_sna(tokenized, w, mf)
                coh_str = f"{coh:.4f}" if coh is not None else "    N/A"
                print(f"  {w:>8} {mf:>10} {coh_str:>14}"
                      f" {kept:>10,} {total:>10,} {n_comm:>12}")
                rows.append({
                    "window"       : w,
                    "min_freq"     : mf,
                    "coherence_cv" : coh,
                    "kept_edges"   : kept,
                    "total_edges"  : total,
                    "n_communities": n_comm
                })
                if coh is not None and coh > best_coh and kept >= 10:
                    best_coh = coh
                    best_cfg = {
                        "window"       : w,
                        "min_freq"     : mf,
                        "coherence_cv" : coh,
                        "kept_edges"   : kept,
                        "n_communities": n_comm
                    }

        if best_cfg:
            print(f"\n  ★ {dsname} 최적 (잔존엣지 ≥ 10 기준)")
            print(f"     WINDOW={best_cfg['window']}, "
                  f"MIN_FREQ={best_cfg['min_freq']} "
                  f"→ Coherence={best_cfg['coherence_cv']:.4f}, "
                  f"잔존엣지={best_cfg['kept_edges']:,}, "
                  f"커뮤니티={best_cfg['n_communities']}")

        all_results[dsname] = rows

    return all_results


# =================================================================
# 7. KMeans 그리디 서치 — TFIDF_FEATS
# =================================================================

def run_kmeans(tokenized, max_features, k=4, n_seeds=3):
    """
    TF-IDF 벡터화 → L2 정규화 → K-Means →
    Silhouette Score (코사인 거리, 시드 평균)

    Parameters
    ----------
    max_features : TF-IDF 최대 어휘 수
    k            : 군집 수
    n_seeds      : 반복 시드 수 (안정성 확보)
    """
    texts = [" ".join(t) for t in tokenized]
    vec   = TfidfVectorizer(
        max_features=max_features,
        sublinear_tf=True, min_df=2,
        analyzer="word", token_pattern=r"\S+"
    )
    try:
        mat   = normalize(vec.fit_transform(texts), norm="l2")
        X     = mat.toarray().astype(np.float32)
        vocab = len(vec.get_feature_names_out())
    except Exception:
        return None, 0, 0

    seeds = [42, 0, 1, 7, 13][:n_seeds]
    sils  = []
    for seed in seeds:
        km  = KMeans(n_clusters=k, random_state=seed, n_init=10)
        lbl = km.fit_predict(X)
        try:
            sil = float(silhouette_score(X, lbl, metric='cosine'))
            sils.append(sil)
        except Exception:
            pass

    mean_sil = round(float(np.mean(sils)), 4) if sils else None
    std_sil  = round(float(np.std(sils)),  4) if len(sils) > 1 else 0.0
    return mean_sil, std_sil, vocab


def kmeans_grid_search(datasets, k=4):
    """TFIDF_FEATS 전체 후보 탐색 (K=4 고정 — VOC/VAD 모두)"""
    FEATS_LIST = [200, 500, 1000, 2000, 5000]

    SEP = "=" * 75
    print(f"\n{SEP}")
    print(f"  STEP 2 / 6   KMeans 그리디 서치  (K={k}, 페르소나 수 고정)")
    print(f"  탐색: TFIDF_FEATS ∈ {{200, 500, 1000, 2000, 5000}}")
    print("  평가: Silhouette Score (코사인 거리, 시드 3개 평균)")
    print(f"  ※ K=4는 알고리즘 최적화가 아닌 도메인 설계 결정")
    print(f"     (Salminen et al., 2022: 358개 연구 평균 M=4.51)")
    print(f"  ※ VAD K 최적값 탐색은 STEP 6 에서 별도 수행")
    print(f"{SEP}")

    all_results = {}

    for dsname, tokenized in datasets.items():
        print(f"\n  ── {dsname}  ({len(tokenized):,}건) ──")
        print(f"  {'FEATS':>10} {'실제어휘수':>12} {'Silhouette':>12} {'Std':>8}")
        print(f"  {'─'*48}")

        rows = []
        best_sil, best_cfg = -1, None

        for feats in FEATS_LIST:
            sil, std, vocab = run_kmeans(tokenized, feats, k)
            sil_str = f"{sil:.4f}" if sil is not None else "   N/A"
            std_str = f"±{std:.4f}" if std is not None else ""
            print(f"  {feats:>10,} {vocab:>12,} {sil_str:>12} {std_str:>8}")
            rows.append({
                "max_features" : feats,
                "actual_vocab" : vocab,
                "silhouette"   : sil,
                "silhouette_std": std
            })
            if sil is not None and sil > best_sil:
                best_sil = sil
                best_cfg = {
                    "max_features": feats,
                    "actual_vocab": vocab,
                    "silhouette"  : sil
                }

        if best_cfg:
            print(f"\n  ★ {dsname} 최적")
            print(f"     FEATS={best_cfg['max_features']:,} "
                  f"(실제어휘={best_cfg['actual_vocab']:,}) "
                  f"→ Silhouette={best_cfg['silhouette']:.4f}")

        all_results[dsname] = rows

    return all_results


# =================================================================
# 8. PageRank Top-N 커버리지 분석
# =================================================================

def run_coverage(tokenized, window=5, min_freq=3, top_n_list=None):
    """
    PageRank 상위 N개 키워드가 전체 문서의 몇 %를 커버하는지 측정
    → Coverage-압축률 트레이드오프 기반 Top-N 결정
    """
    if top_n_list is None:
        top_n_list = [20, 50, 100, 200, 500]

    cooc = defaultdict(int)
    for tokens in tokenized:
        n = len(tokens)
        for i in range(n):
            for j in range(i + 1, min(i + window + 1, n)):
                pair = tuple(sorted([tokens[i], tokens[j]]))
                cooc[pair] += 1

    filtered = {k: v for k, v in cooc.items() if v >= min_freq}
    G = nx.Graph()
    for (w1, w2), freq in filtered.items():
        G.add_edge(w1, w2, weight=freq)

    if G.number_of_nodes() < 4:
        return {}, 0

    pagerank     = nx.pagerank(G, weight='weight')
    sorted_nodes = sorted(pagerank, key=pagerank.get, reverse=True)
    total_vocab  = len(sorted_nodes)

    results = {}
    prev_cov = 0.0
    for n in top_n_list:
        actual_n = min(n, total_vocab)
        top_kw   = set(sorted_nodes[:actual_n])
        covered  = sum(
            1 for tokens in tokenized
            if any(t in top_kw for t in tokens)
        )
        coverage      = round(covered / len(tokenized), 4)
        marginal_gain = round(coverage - prev_cov, 4)
        results[n] = {
            "top_n"        : n,
            "actual_n"     : actual_n,
            "pct_vocab"    : round(actual_n / total_vocab * 100, 2),
            "coverage"     : coverage,
            "marginal_gain": marginal_gain,
            "covered_docs" : covered,
            "total_docs"   : len(tokenized),
            "total_vocab"  : total_vocab
        }
        prev_cov = coverage

    return results, total_vocab


def coverage_search(datasets, window=5, min_freq=3):
    """PageRank Top-N 커버리지 전체 분석"""
    TOP_N_LIST = [20, 50, 100, 200, 500]

    SEP = "=" * 75
    print(f"\n{SEP}")
    print(f"  STEP 3 / 6   PageRank Top-N 문서 커버리지 분석")
    print(f"  (WINDOW={window}, MIN_FREQ={min_freq} 고정)")
    print(f"  평가: Coverage = Top-N 키워드를 1개 이상 포함한 문서 비율")
    print(f"        Marginal Gain = 이전 N 대비 커버리지 증가분")
    print(f"{SEP}")

    all_results = {}

    for dsname, tokenized in datasets.items():
        res, total_vocab = run_coverage(
            tokenized, window=window, min_freq=min_freq,
            top_n_list=TOP_N_LIST
        )
        print(f"\n  ── {dsname}  (전체 어휘: {total_vocab:,}개) ──")
        print(f"  {'Top-N':>8} {'어휘비율':>10} {'Coverage':>12}"
              f" {'한계효용':>10} {'커버문서':>12}")
        print(f"  {'─'*58}")

        rows = []
        for n, v in res.items():
            mg_str = f"+{v['marginal_gain']:.4f}" if v['marginal_gain'] >= 0 \
                     else f"{v['marginal_gain']:.4f}"
            print(f"  {n:>8} {v['pct_vocab']:>9.2f}%"
                  f" {v['coverage']:>12.4f}"
                  f" {mg_str:>10}"
                  f" {v['covered_docs']:>6}/{v['total_docs']}")
            rows.append(v)

        opt_95 = next(
            (n for n, v in res.items() if v['coverage'] >= 0.95), None
        )
        if opt_95:
            print(f"\n  ★ Coverage 95% 달성 최소 Top-N = {opt_95}"
                  f"  (어휘비율 {res[opt_95]['pct_vocab']:.2f}%)")

        all_results[dsname] = rows

    return all_results


# =================================================================
# 9. LDA 파라미터 그리디 서치 — passes / no_below / no_above
# =================================================================

def run_lda(tokenized, passes, no_below, no_above, k=4, n_seeds=3):
    """
    LDA 학습 → Perplexity + Coherence C_v

    Parameters
    ----------
    passes   : LDA 학습 반복 횟수 (수렴 기준)
    no_below : 어휘 최소 문서 빈도 (희귀어 제거)
    no_above : 어휘 최대 문서 비율 (일반어 제거)
    k        : 토픽 수
    n_seeds  : 반복 시드 수
    """
    from gensim.models import LdaModel

    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)

    vocab_size = len(dictionary)
    if vocab_size < k * 2:
        return None, None, vocab_size

    corpus = [dictionary.doc2bow(t) for t in tokenized]
    corpus = [c for c in corpus if c]

    if len(corpus) < k:
        return None, None, vocab_size

    seeds = [42, 0, 1, 7, 13][:n_seeds]
    cohs, perps = [], []

    for seed in seeds:
        try:
            lda = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=k,
                passes=passes,
                alpha='auto', eta='auto',
                random_state=seed,
                per_word_topics=False
            )
            perp = lda.log_perplexity(corpus)
            perps.append(perp)

            topic_words = [
                [w for w, _ in lda.show_topic(i, topn=10)]
                for i in range(k)
            ]
            coh = compute_coherence(topic_words, tokenized)
            if coh is not None:
                cohs.append(coh)
        except Exception:
            pass

    mean_coh  = round(float(np.mean(cohs)),  4) if cohs  else None
    mean_perp = round(float(np.mean(perps)), 4) if perps else None
    return mean_coh, mean_perp, vocab_size


def lda_grid_search(datasets, k=4):
    """LDA passes × no_below × no_above 탐색 (K=4 고정)"""
    PASSES_LIST   = [5, 10, 15, 20, 30]
    NO_BELOW_LIST = [1, 2, 3, 5]
    NO_ABOVE_LIST = [0.70, 0.80, 0.90, 0.95]

    SEP = "=" * 75
    print(f"\n{SEP}")
    print(f"  STEP 4 / 6   LDA 파라미터 그리디 서치  (K={k} 고정)")
    print(f"  탐색A: passes   ∈ {{5,10,15,20,30}}  (no_below=2, no_above=0.9 고정)")
    print(f"  탐색B: no_below ∈ {{1,2,3,5}}        (passes=10, no_above=0.9 고정)")
    print(f"  탐색C: no_above ∈ {{0.70,0.80,0.90,0.95}} (passes=10, no_below=2 고정)")
    print(f"  평가: Coherence C_v + Perplexity (Coh 높음 & Perp 낮음 최적)")
    print(f"  ※ VAD 최적 K 탐색은 STEP 6 에서 별도 수행")
    print(f"{SEP}")

    all_results = {}

    for dsname, tokenized in datasets.items():
        print(f"\n  ── {dsname}  ({len(tokenized):,}건) ──")
        rows = {}

        # ── A: passes 탐색
        print(f"\n  [A] passes 탐색  (no_below=2, no_above=0.9 고정)")
        print(f"  {'passes':>8} {'Coherence_Cv':>14} {'Perplexity':>12} {'어휘수':>8}")
        print(f"  {'─'*48}")
        pass_rows = []
        best_coh, best_pass = -1, None
        for p in PASSES_LIST:
            coh, perp, vocab = run_lda(tokenized, p, 2, 0.9, k)
            cs = f"{coh:.4f}"  if coh  is not None else "   N/A"
            ps = f"{perp:.2f}" if perp is not None else "   N/A"
            print(f"  {p:>8} {cs:>14} {ps:>12} {vocab:>8,}")
            pass_rows.append({"passes": p, "coherence_cv": coh,
                               "perplexity": perp, "vocab": vocab})
            if coh is not None and coh > best_coh:
                best_coh  = coh
                best_pass = p
        if best_pass:
            print(f"  ★ 최적 passes = {best_pass}  (Coherence={best_coh:.4f})")
        rows["passes"] = pass_rows

        # ── B: no_below 탐색
        print(f"\n  [B] no_below 탐색  (passes=10, no_above=0.9 고정)")
        print(f"  {'no_below':>10} {'Coherence_Cv':>14} {'Perplexity':>12} {'어휘수':>8}")
        print(f"  {'─'*50}")
        nb_rows = []
        best_coh, best_nb = -1, None
        for nb in NO_BELOW_LIST:
            coh, perp, vocab = run_lda(tokenized, 10, nb, 0.9, k)
            cs = f"{coh:.4f}"  if coh  is not None else "   N/A"
            ps = f"{perp:.2f}" if perp is not None else "   N/A"
            print(f"  {nb:>10} {cs:>14} {ps:>12} {vocab:>8,}")
            nb_rows.append({"no_below": nb, "coherence_cv": coh,
                            "perplexity": perp, "vocab": vocab})
            if coh is not None and coh > best_coh:
                best_coh = coh
                best_nb  = nb
        if best_nb is not None:
            print(f"  ★ 최적 no_below = {best_nb}  (Coherence={best_coh:.4f})")
        rows["no_below"] = nb_rows

        # ── C: no_above 탐색
        print(f"\n  [C] no_above 탐색  (passes=10, no_below=2 고정)")
        print(f"  {'no_above':>10} {'Coherence_Cv':>14} {'Perplexity':>12} {'어휘수':>8}")
        print(f"  {'─'*50}")
        na_rows = []
        best_coh, best_na = -1, None
        for na in NO_ABOVE_LIST:
            coh, perp, vocab = run_lda(tokenized, 10, 2, na, k)
            cs = f"{coh:.4f}"  if coh  is not None else "   N/A"
            ps = f"{perp:.2f}" if perp is not None else "   N/A"
            print(f"  {na:>10.2f} {cs:>14} {ps:>12} {vocab:>8,}")
            na_rows.append({"no_above": na, "coherence_cv": coh,
                            "perplexity": perp, "vocab": vocab})
            if coh is not None and coh > best_coh:
                best_coh = coh
                best_na  = na
        if best_na is not None:
            print(f"  ★ 최적 no_above = {best_na:.2f}  (Coherence={best_coh:.4f})")
        rows["no_above"] = na_rows

        all_results[dsname] = rows

    return all_results


# =================================================================
# 10. N_RUNS 안정성 탐색 — 반복 횟수별 Coherence 분산
# =================================================================

def nruns_stability_search(datasets, k=4):
    """
    N_RUNS(반복 횟수)별 Coherence C_v의 mean ± std 계산
    std가 충분히 작아지는 N 선택 → 재현성 확보 최소 반복 수
    """
    from gensim.models import LdaModel

    N_RUNS_LIST = [3, 5, 10, 15, 20]

    SEP = "=" * 75
    print(f"\n{SEP}")
    print(f"  STEP 5 / 6   N_RUNS 안정성 탐색  (LDA, K={k})")
    print(f"  탐색: N_RUNS ∈ {{3, 5, 10, 15, 20}}")
    print(f"  평가: Coherence C_v  mean ± std")
    print(f"        std가 수렴(≤ 0.01)하는 최소 N 선택")
    print(f"{SEP}")

    all_results = {}

    for dsname, tokenized in datasets.items():
        print(f"\n  ── {dsname}  ({len(tokenized):,}건) ──")
        print(f"  {'N_RUNS':>8} {'Coh_mean':>10} {'Coh_std':>10} {'안정여부':>10}")
        print(f"  {'─'*44}")

        dictionary = corpora.Dictionary(tokenized)
        dictionary.filter_extremes(no_below=2, no_above=0.9)
        corpus = [dictionary.doc2bow(t) for t in tokenized]
        corpus = [c for c in corpus if c]

        rows = []
        all_cohs = []
        seed_pool = [42, 0, 1, 7, 13, 21, 37, 55, 77, 99,
                     100, 200, 300, 400, 500, 11, 22, 33, 44, 55]

        for seed in seed_pool[:20]:
            try:
                lda = LdaModel(
                    corpus=corpus, id2word=dictionary,
                    num_topics=k, passes=10,
                    alpha='auto', eta='auto',
                    random_state=seed, per_word_topics=False
                )
                tw = [[w for w, _ in lda.show_topic(i, topn=10)]
                      for i in range(k)]
                coh = compute_coherence(tw, tokenized)
                if coh is not None:
                    all_cohs.append(coh)
            except Exception:
                pass

        for n in N_RUNS_LIST:
            subset = all_cohs[:n]
            if len(subset) < 2:
                print(f"  {n:>8} {'N/A':>10} {'N/A':>10} {'─':>10}")
                rows.append({"n_runs": n, "mean": None, "std": None})
                continue
            m   = round(float(np.mean(subset)), 4)
            std = round(float(np.std(subset)),  4)
            stable = "✅ 수렴" if std <= 0.01 else "─"
            print(f"  {n:>8} {m:>10.4f} {std:>10.4f} {stable:>10}")
            rows.append({"n_runs": n, "mean": m, "std": std,
                         "stable": std <= 0.01})

        opt_n = next(
            (r["n_runs"] for r in rows
             if r.get("std") is not None and r["std"] <= 0.01), None
        )
        if opt_n:
            print(f"  ★ std ≤ 0.01 달성 최소 N_RUNS = {opt_n}")
        else:
            print(f"  ★ N_RUNS=20에서도 std > 0.01 → 데이터 다양성 높음")

        all_results[dsname] = rows

    return all_results


# =================================================================
# 11. [신규] VAD 전용 K 탐색 — LDA + KMeans 동시 평가
#     ★ 핵심 논리 ★
#     voc_en : Top-Down  → K=4 고정 (페르소나 먼저 설계)
#     vad_en : Bottom-Up → K를 데이터에서 귀납적으로 탐색
# =================================================================

def run_lda_for_k(tokenized, k, passes=30, no_below=2, no_above=0.9, n_seeds=5):
    """
    단일 k에 대해 LDA 실행 → Coherence C_v + Perplexity 반환.
    STEP 6 전용 헬퍼 함수.

    Parameters
    ----------
    k        : 탐색 대상 토픽 수
    passes   : STEP 4에서 결정된 최적 passes 사용 (기본값 30)
    no_below : STEP 4에서 결정된 최적값 사용 (기본값 2)
    no_above : STEP 4에서 결정된 최적값 사용 (기본값 0.9)
    n_seeds  : 시드 반복 수 (기본값 5, 안정성 확보)
    """
    from gensim.models import LdaModel

    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)

    vocab_size = len(dictionary)
    if vocab_size < k * 2:
        return None, None, vocab_size

    corpus = [dictionary.doc2bow(t) for t in tokenized]
    corpus = [c for c in corpus if c]

    if len(corpus) < k:
        return None, None, vocab_size

    seeds = [42, 0, 1, 7, 13][:n_seeds]
    cohs, perps = [], []

    for seed in seeds:
        try:
            lda = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=k,
                passes=passes,
                alpha='auto', eta='auto',
                random_state=seed,
                per_word_topics=False
            )
            perp = lda.log_perplexity(corpus)
            perps.append(perp)

            topic_words = [
                [w for w, _ in lda.show_topic(i, topn=10)]
                for i in range(k)
            ]
            coh = compute_coherence(topic_words, tokenized)
            if coh is not None:
                cohs.append(coh)
        except Exception:
            pass

    mean_coh  = round(float(np.mean(cohs)),  4) if cohs  else None
    mean_perp = round(float(np.mean(perps)), 4) if perps else None
    return mean_coh, mean_perp, vocab_size


def vad_k_search(vad_tokens,
                 k_range=None,
                 lda_passes=30,
                 lda_no_below=2,
                 lda_no_above=0.9,
                 tfidf_feats=1000,
                 n_seeds=5):
    """
    ★ vad_en 전용 Bottom-Up K 탐색 ★

    왜 vad_en만 K를 탐색하는가?
    ─────────────────────────────────────────────────────────────
    voc_en (Top-Down):
      연구자가 페르소나 4개를 먼저 설계하고 데이터를 수집.
      K=4는 도메인 설계 결정 → 탐색 불필요.
      근거: Salminen et al. (2022), M=4.51

    vad_en (Bottom-Up):
      레이블 없는 원시 리뷰 데이터.
      데이터가 몇 개의 군집으로 나뉘는지 사전에 알 수 없음.
      → K를 데이터에서 귀납적으로 결정해야 함.
      → LDA Coherence C_v + KMeans Silhouette Score 동시 평가.
    ─────────────────────────────────────────────────────────────

    평가 지표:
      LDA  : Coherence C_v (높을수록 토픽 의미 일관성 좋음)
              Perplexity    (낮을수록 — 덜 음수 — 모델 적합도 좋음)
      KMeans: Silhouette    (높을수록 군집 분리도 좋음, -1~+1)

    최적 K 판단 기준:
      1) LDA Coherence가 더 이상 증가하지 않는 엘보우(Elbow) 지점
      2) KMeans Silhouette이 최대인 지점
      3) 두 지표가 동시에 좋은 K 선택 (트레이드오프 고려)

    Parameters
    ----------
    vad_tokens   : vad_en 토큰 리스트
    k_range      : 탐색할 K 후보 리스트 (기본 2~10)
    lda_passes   : STEP 4에서 결정된 최적 passes
    lda_no_below : STEP 4에서 결정된 최적 no_below
    lda_no_above : STEP 4에서 결정된 최적 no_above
    tfidf_feats  : STEP 2에서 결정된 최적 FEATS
    n_seeds      : 시드 반복 수
    """
    if k_range is None:
        k_range = list(range(2, 11))   # K = 2, 3, 4, 5, 6, 7, 8, 9, 10

    SEP = "=" * 75
    print(f"\n{SEP}")
    print(f"  STEP 6 / 6   ★ VAD 전용 K 탐색 (Bottom-Up) ★")
    print(f"  대상: vad_en ({len(vad_tokens):,}건) 단독")
    print(f"  ※ voc_en은 Top-Down K=4 고정 → 이 탐색 대상 아님")
    print(f"  탐색: K ∈ {{{', '.join(map(str, k_range))}}}")
    print(f"  LDA  파라미터: passes={lda_passes}, "
          f"no_below={lda_no_below}, no_above={lda_no_above}")
    print(f"  KMeans 파라미터: TFIDF_FEATS={tfidf_feats:,}")
    print(f"  평가 지표 (LDA): Coherence C_v ↑  +  Perplexity ↑ (덜 음수)")
    print(f"  평가 지표 (KMeans): Silhouette Score ↑")
    print(f"  최적 K = 두 지표 엘보우 기반 동시 최적점")
    print(f"{SEP}")

    lda_rows    = []
    kmeans_rows = []

    # ── LDA K 탐색
    print(f"\n  [LDA] K별 Coherence C_v + Perplexity")
    print(f"  {'K':>5} {'Coherence_Cv':>14} {'Perplexity':>14} {'어휘수':>8}  {'비고':>10}")
    print(f"  {'─'*58}")

    best_lda_coh  = -1
    best_lda_k    = None
    prev_coh      = None
    elbow_lda_k   = None   # Coherence 증가율 급감 지점

    for k in k_range:
        coh, perp, vocab = run_lda_for_k(
            vad_tokens, k,
            passes=lda_passes,
            no_below=lda_no_below,
            no_above=lda_no_above,
            n_seeds=n_seeds
        )
        cs = f"{coh:.4f}"  if coh  is not None else "    N/A"
        ps = f"{perp:.4f}" if perp is not None else "    N/A"

        # 엘보우 감지: 이전 대비 Coherence 증가분이 0.005 미만으로 꺾이는 첫 K
        note = ""
        if coh is not None and prev_coh is not None:
            delta = coh - prev_coh
            if elbow_lda_k is None and delta < 0.005:
                elbow_lda_k = k - 1   # 직전 K가 엘보우
                note = "← 엘보우"

        print(f"  {k:>5} {cs:>14} {ps:>14} {vocab:>8}  {note}")

        lda_rows.append({
            "k"           : k,
            "coherence_cv": coh,
            "perplexity"  : perp,
            "vocab"       : vocab,
        })

        if coh is not None and coh > best_lda_coh:
            best_lda_coh = coh
            best_lda_k   = k
        prev_coh = coh

    if best_lda_k is not None:
        print(f"\n  ★ LDA Coherence 최고 K = {best_lda_k}"
              f"  (C_v={best_lda_coh:.4f})")
    if elbow_lda_k is not None:
        print(f"  ★ LDA Coherence 엘보우 K = {elbow_lda_k}"
              f"  (증가율 급감 직전 지점)")

    # ── KMeans K 탐색
    print(f"\n  [KMeans] K별 Silhouette Score  (FEATS={tfidf_feats:,})")
    print(f"  {'K':>5} {'Silhouette':>12} {'Std':>8}  {'비고':>10}")
    print(f"  {'─'*42}")

    best_sil_score = -1
    best_sil_k     = None

    texts = [" ".join(t) for t in vad_tokens]
    vec   = TfidfVectorizer(
        max_features=tfidf_feats,
        sublinear_tf=True, min_df=2,
        analyzer="word", token_pattern=r"\S+"
    )
    try:
        mat = normalize(vec.fit_transform(texts), norm="l2")
        X   = mat.toarray().astype(np.float32)
    except Exception:
        print("  [오류] TF-IDF 벡터화 실패 → KMeans K 탐색 스킵")
        X = None

    seeds = [42, 0, 1, 7, 13][:n_seeds]

    for k in k_range:
        if X is None or k >= X.shape[0]:
            kmeans_rows.append({"k": k, "silhouette": None, "silhouette_std": None})
            print(f"  {k:>5} {'N/A':>12} {'':>8}")
            continue

        sils = []
        for seed in seeds:
            try:
                km  = KMeans(n_clusters=k, random_state=seed, n_init=10)
                lbl = km.fit_predict(X)
                sil = float(silhouette_score(X, lbl, metric='cosine'))
                sils.append(sil)
            except Exception:
                pass

        mean_sil = round(float(np.mean(sils)), 4) if sils else None
        std_sil  = round(float(np.std(sils)),  4) if len(sils) > 1 else 0.0

        note = ""
        sil_str = f"{mean_sil:.4f}" if mean_sil is not None else "   N/A"
        std_str = f"±{std_sil:.4f}" if mean_sil is not None else ""

        if mean_sil is not None and mean_sil > best_sil_score:
            best_sil_score = mean_sil
            best_sil_k     = k
            note = "← 현재 최고"

        print(f"  {k:>5} {sil_str:>12} {std_str:>8}  {note}")
        kmeans_rows.append({
            "k"            : k,
            "silhouette"   : mean_sil,
            "silhouette_std": std_sil,
        })

    if best_sil_k is not None:
        print(f"\n  ★ KMeans Silhouette 최고 K = {best_sil_k}"
              f"  (Silhouette={best_sil_score:.4f})")

    # ── 최종 K 권고
    print(f"\n  {'─'*58}")
    print(f"  [최종 K 권고 — vad_en Bottom-Up]")
    candidates = set()
    if best_lda_k   is not None: candidates.add(best_lda_k)
    if elbow_lda_k  is not None: candidates.add(elbow_lda_k)
    if best_sil_k   is not None: candidates.add(best_sil_k)

    if len(candidates) == 1:
        final_k = candidates.pop()
        print(f"  → LDA Coherence / 엘보우 / KMeans Silhouette 모두 K={final_k} 지지")
        print(f"  → ✅ 권고 K = {final_k}  (세 지표 일치)")
    else:
        print(f"  → LDA Coherence 최고 K  = {best_lda_k}")
        print(f"  → LDA 엘보우 K          = {elbow_lda_k}")
        print(f"  → KMeans Silhouette 최고 K = {best_sil_k}")
        # 최빈값 또는 중앙값으로 권고
        cands = [x for x in [best_lda_k, elbow_lda_k, best_sil_k]
                 if x is not None]
        from collections import Counter
        most_common_k = Counter(cands).most_common(1)[0][0]
        print(f"  → ⚠️  지표 간 불일치 — 최빈값 기준 권고 K = {most_common_k}")
        print(f"       (논문 방법론: 두 지표 동시 고려 + 도메인 해석 병행 권장)")

    print(f"  {'─'*58}")
    print(f"  [연구 논리 요약]")
    print(f"    voc_en (Top-Down)  : K=4 고정 (사전 설계 페르소나 수)")
    print(f"    vad_en (Bottom-Up) : K=? 탐색 완료 → 위 권고값 사용")
    print(f"    → 두 방식의 K 결정 근거가 명확히 구분됨 (방법론적 일관성)")

    return {
        "target_dataset"  : "vad_en",
        "k_range"         : k_range,
        "lda_params_used" : {
            "passes"  : lda_passes,
            "no_below": lda_no_below,
            "no_above": lda_no_above,
            "n_seeds" : n_seeds,
        },
        "kmeans_params_used": {
            "tfidf_feats": tfidf_feats,
            "n_seeds"    : n_seeds,
        },
        "lda_k_search"   : lda_rows,
        "kmeans_k_search": kmeans_rows,
        "summary": {
            "best_lda_coherence_k" : best_lda_k,
            "best_lda_coherence_cv": best_lda_coh if best_lda_k else None,
            "elbow_lda_k"          : elbow_lda_k,
            "best_kmeans_sil_k"    : best_sil_k,
            "best_kmeans_sil"      : best_sil_score if best_sil_k else None,
        },
        "rationale": (
            "vad_en은 Bottom-Up 데이터로, 사전 K 가정 없이 데이터에서 "
            "귀납적으로 최적 K를 결정. "
            "LDA Coherence C_v 엘보우 + KMeans Silhouette 동시 평가."
        ),
    }


# =================================================================
# 12. 결과 요약 출력 (전체)
# =================================================================

def print_summary():
    SEP = "=" * 75
    print(f"\n{SEP}")
    print("  ★ 파라미터 결정 근거 요약 (논문 방법론 기재용)")
    print(f"{SEP}")
    print("""
  [K=4 — voc_en 페르소나·세그먼트 수]
    알고리즘 최적화 문제가 아닌 도메인 설계 결정 (Top-Down).
    Salminen et al. (2022): 358개 연구 평균 M=4.51.
    K=4는 문헌 평균에 가장 근접한 정수값.

  [K=? — vad_en 최적 토픽 수]
    Bottom-Up 귀납 탐색 (STEP 6).
    LDA Coherence C_v 엘보우 + KMeans Silhouette 동시 평가.
    → 두 지표 동시 최적인 K를 vad_en 최종 K로 결정.
    ※ voc_en(Top-Down)과 vad_en(Bottom-Up)은 K 결정 논리가 명확히 구분됨.

  [WINDOW, MIN_FREQ — SNA 공출현 파라미터]
    Coherence C_v + 네트워크 잔존엣지(밀도) 동시 고려.
    잔존엣지 < 10이면 커뮤니티 탐지 불안정.
    → 두 데이터셋 모두 충분한 밀도 유지 + 안정적 Coherence 조합 선택.
    참고: Mihalcea & Tarau (2004) 권장 범위 (W=2~10)

  [TFIDF_FEATS — KMeans 어휘 크기]
    Silhouette은 FEATS 증가에 따라 단조 감소 (차원의 저주).
    실제 어휘 포화 이후 결과 동일.
    → 전체 어휘 포괄 + 과적합 방지 균형점 선택.

  [TOP_PAGERANK_N — PageRank 상위 키워드 수]
    Coverage 95% 달성 후 한계 효용 급감.
    → 95% 이상 달성 최소 N 선택 (압축률-커버리지 최적점).

  [LDA passes — 학습 수렴]
    Perplexity 수렴 + Coherence C_v 안정화 기준.
    → 두 지표가 동시에 수렴하는 최소 passes 선택.

  [LDA no_below / no_above — 어휘 필터]
    데이터 크기별 잔존 어휘 수 차이 확인.
    → Coherence C_v 최대화 + 적절한 어휘 규모 유지 조합 선택.
    ※ voc_en(965건)과 vad_en(2370건)은 규모 차이 → 같은 값이라도 효과 다름.

  [N_RUNS — 반복 횟수]
    Coherence std ≤ 0.01 수렴 기준.
    → std가 수렴하는 최소 N 선택 (재현성 확보 최소 반복 수).
""")


# =================================================================
# 13. 메인 실행
# =================================================================

if __name__ == "__main__":
    print("\n" + "=" * 75)
    print("  VOC2Persona-LLM | 파라미터 그리디 서치 (전체 데이터)")
    print("  탐색 대상: WINDOW, MIN_FREQ, TFIDF_FEATS, TOP_PAGERANK_N,")
    print("             LDA passes, no_below, no_above, N_RUNS")
    print("  ★ 신규: STEP 6 — vad_en 전용 K 탐색 (Bottom-Up)")
    print("=" * 75)

    # ── 데이터 로드
    voc_tokens, vad_tokens = load_data()
    datasets = {
        "voc_en": voc_tokens,
        "vad_en": vad_tokens,
    }

    # ── STEP 1~5: 기존 탐색 (전체 데이터셋 대상)
    sna_res   = sna_grid_search(datasets)
    km_res    = kmeans_grid_search(datasets)
    cov_res   = coverage_search(datasets)
    lda_res   = lda_grid_search(datasets)
    nruns_res = nruns_stability_search(datasets)

    # ── STEP 6: vad_en 전용 K 탐색 (Bottom-Up)
    # ※ STEP 4 결과에서 최적 LDA 파라미터를 확인 후 아래 값 수정 권장
    #    현재는 STEP 4 결과 기준 기본값 사용
    vad_k_res = vad_k_search(
        vad_tokens,
        k_range     = list(range(2, 11)),  # K = 2~10 탐색
        lda_passes  = 30,                  # STEP 4-A 최적값 반영
        lda_no_below= 2,                   # STEP 4-B 최적값 반영
        lda_no_above= 0.9,                 # STEP 4-C 최적값 반영
        tfidf_feats = 1000,                # STEP 2 최적값 반영
        n_seeds     = 5,
    )

    # ── 요약 출력
    print_summary()

    # ── JSON 저장
    final = {
        "meta": {
            "voc_en_n_docs" : len(voc_tokens),
            "vad_en_n_docs" : len(vad_tokens),
            "k_voc"         : 4,
            "k_vad"         : vad_k_res["summary"],
            "note_voc"      : "K=4 — Top-Down 도메인 설계 결정 (Salminen et al., 2022)",
            "note_vad"      : "K=? — Bottom-Up 귀납 탐색 (LDA Coherence + KMeans Silhouette)",
        },
        "step1_sna_grid"       : sna_res,
        "step2_kmeans_grid"    : km_res,
        "step3_coverage"       : cov_res,
        "step4_lda_grid"       : lda_res,
        "step5_nruns_stability": nruns_res,
        "step6_vad_k_search"   : vad_k_res,   # ★ 신규
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print(f"\n  결과 저장 완료: {OUT_PATH}")
    print("=" * 75)