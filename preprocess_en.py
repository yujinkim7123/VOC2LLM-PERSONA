"""
=================================================================
전처리 파이프라인 (수정 버전 — voc_en + vad_en 두 개만)
=================================================================

입력 파일:
  1. merged_voc.csv        → voc_en_docs  (EN 리뷰만 전처리)
                             ※ KO 번역본은 이번 실험에서 제외
  2. VAD_dataset.xlsx      → vad_en_docs  (EN 전처리)

출력 (2개 독립 JSON):
  preprocessed_voc_en.json
  preprocessed_vad_en.json

전처리 로직:
  EN → spaCy lemmatization + NLTK stopwords (sna_only.py 동일)

사용법:
  python preprocess_pipeline.py
=================================================================
"""

import re
import os
import json
import warnings
import pandas as pd
from collections import Counter

warnings.filterwarnings("ignore")


# =================================================================
# 0. 경로 설정  ← 여기만 수정하세요
# =================================================================
VOC_PATH = r""
VAD_PATH = r""
OUT_DIR  = r""

os.makedirs(OUT_DIR, exist_ok=True)


# =================================================================
# 1. 영어 불용어 설정
# =================================================================
def get_en_stopwords():
    """NLTK 불용어 로드. 미설치 시 내장 fallback 사용."""
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
# 2. spaCy 싱글톤 로더
# =================================================================
_spacy_nlp    = None
_spacy_loaded = False

def get_spacy_nlp():
    global _spacy_nlp, _spacy_loaded
    if not _spacy_loaded:
        _spacy_loaded = True
        try:
            import spacy
            print("  [EN] spaCy 모델 로딩 중 (en_core_web_sm)...", flush=True)
            _spacy_nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            print("  [EN] spaCy 로딩 완료", flush=True)
        except (OSError, ImportError):
            print("  [경고] en_core_web_sm 미설치 → fallback 사용")
            print("         설치: python -m spacy download en_core_web_sm")
            _spacy_nlp = None
    return _spacy_nlp


# =================================================================
# 3. 영어 토크나이저
# =================================================================
def tokenize_en(text: str) -> list[str]:
    """
    영어 전처리 파이프라인.
    spaCy lemmatization → 불용어/구두점/공백/숫자 제거 → len > 2 필터.
    spaCy 미설치 시 소문자 + 불용어 제거 fallback.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    nlp = get_spacy_nlp()

    if nlp is not None:
        doc = nlp(text.lower())
        tokens = [
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
        tokens = [
            t for t in text_clean.split()
            if len(t) > 2 and t not in EN_STOPS
        ]

    return tokens


# =================================================================
# 4. 공통 빌드 · 저장 함수
# =================================================================
def build_docs(texts: list[str], metas: list[dict],
               label: str) -> list[dict]:
    """
    텍스트 리스트 → {"tokens": [...], "meta": {...}} 리스트 변환.
    유효 토큰 2개 미만 문서는 제외.
    """
    docs = []
    skip = 0
    print(f"\n  [{label}] 토크나이징 시작 ({len(texts)}건)...")

    for idx, (text, meta) in enumerate(zip(texts, metas)):
        if idx % 500 == 0 and idx > 0:
            print(f"    처리 중: {idx}/{len(texts)}건...", flush=True)

        tokens = tokenize_en(text)

        if len(tokens) < 2:
            skip += 1
            continue

        docs.append({"tokens": tokens, "meta": meta})

    avg_len = sum(len(d["tokens"]) for d in docs) / len(docs) if docs else 0
    print(f"  [{label}] 완료: 유효 {len(docs)}건 / 제외 {skip}건 / 평균 {avg_len:.1f} 토큰")
    return docs


def save_docs(docs: list[dict], filename: str):
    """전처리 결과를 JSON으로 저장."""
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"total": len(docs), "data": docs}, f, ensure_ascii=False, indent=2)
    print(f"  저장 완료: {path}  ({len(docs)}건)")


def print_stats(docs: list[dict], label: str):
    """토큰 통계 출력."""
    if not docs:
        print(f"  [{label}] 데이터 없음")
        return
    lengths = [len(d["tokens"]) for d in docs]
    print(f"\n  [{label}] 통계")
    print(f"    건수       : {len(docs):,}건")
    print(f"    평균 토큰  : {sum(lengths)/len(lengths):.1f}개")
    print(f"    최소 토큰  : {min(lengths)}개")
    print(f"    최대 토큰  : {max(lengths)}개")
    all_tokens = [t for d in docs for t in d["tokens"]]
    top10 = Counter(all_tokens).most_common(10)
    print(f"    상위 10 토큰: {[w for w, _ in top10]}")


# =================================================================
# 5. [A] merged_voc.csv → voc_en 전처리
# =================================================================
def load_voc_en(path: str) -> list[dict]:
    """
    merged_voc.csv 로드 → 영어 리뷰(EN)만 전처리.

    컬럼 매핑 규칙 (sna_only.py 동일):
      S1  : review = EN 원문   ← 이것만 사용
      S2~S4: review_translation = EN 번역  ← 이것만 사용

    KO 컬럼(review_translation/review)은 이번 실험에서 완전 제외.
    """
    print("\n[A] merged_voc.csv 로드 중 (EN 전용)...")
    df = pd.read_csv(path, encoding="utf-8")
    print(f"  전체 {len(df)}건 로드 완료")
    print(f"  null 확인 → review: {df['review'].isna().sum()} / "
          f"review_translation: {df['review_translation'].isna().sum()}")

    en_texts, metas = [], []

    for _, row in df.iterrows():
        meta = {
            "source":       "merged_voc",
            "id":           row.get("id", ""),
            "segment":      row.get("segment", ""),
            "segment_name": row.get("segment_name", ""),
            "polarity":     row.get("polarity", ""),
            "topic_id":     row.get("topic_id", ""),
            "topic_name":   row.get("topic_name", ""),
            "product":      row.get("product", ""),
            "rating":       float(row.get("rating", 0) or 0),
            "sentiment":    row.get("sentiment", ""),
        }

        # S1: review=EN 원문 사용
        # S2~S4: review_translation=EN 번역 사용
        if row["segment"] == "S1":
            en_texts.append(str(row["review"]))
        else:
            en_texts.append(str(row["review_translation"]))

        metas.append(meta)

    docs = build_docs(en_texts, metas, label="VOC-EN")
    return docs


# =================================================================
# 6. [B] VAD_dataset.xlsx → vad_en 전처리
# =================================================================
def load_vad_en(path: str) -> list[dict]:
    """
    VAD_dataset.xlsx 로드 → EN 전처리.

    리뷰 텍스트: Description(제목) + Unnamed:4(본문) 합치기.
    메타: Device Name, Company, Source, Sentiment, Aspect(s).
    Device Name/Company/Source는 병합셀 구조 → forward fill 적용.
    """
    print("\n[B] VAD_dataset.xlsx 로드 중...")
    df = pd.read_excel(path)
    print(f"  전체 {len(df)}건 로드 완료")
    print(f"  컬럼 목록: {list(df.columns)}")
    print(f"  null 확인 → Description: {df['Description'].isna().sum()} / "
          f"Unnamed:4: {df['Unnamed: 4'].isna().sum()}")

    # Description(제목) + Unnamed:4(본문) 합치기
    df["full_text"] = (
        df["Description"].fillna("").astype(str).str.strip()
        + " "
        + df["Unnamed: 4"].fillna("").astype(str).str.strip()
    ).str.strip()

    # 병합셀 구조 → forward fill
    df["Device Name"] = df["Device Name"].ffill()
    df["Company"]     = df["Company"].ffill()
    df["Source"]      = df["Source"].ffill()

    texts, metas = [], []

    for _, row in df.iterrows():
        meta = {
            "source":      "vad_dataset",
            "device_name": str(row.get("Device Name", "")),
            "company":     str(row.get("Company", "")),
            "data_source": str(row.get("Source", "")),
            "sentiment":   str(row.get("Sentiment", "")),
            "aspect":      str(row.get("Aspect(s)", "")),
        }
        texts.append(str(row["full_text"]))
        metas.append(meta)

    docs = build_docs(texts, metas, label="VAD-EN")
    return docs


# =================================================================
# 7. 메인 실행
# =================================================================
if __name__ == "__main__":
    SEP = "=" * 60
    print(f"\n{SEP}")
    print("  전처리 파이프라인 시작 (voc_en + vad_en)")
    print(f"{SEP}\n")

    # ── [A] VOC-EN ───────────────────────────────────────────────
    print("[A] merged_voc.csv → EN 전처리")
    voc_en_docs = load_voc_en(VOC_PATH)
    print_stats(voc_en_docs, "VOC-EN")
    save_docs(voc_en_docs, "preprocessed_voc_en.json")

    # ── [B] VAD-EN ───────────────────────────────────────────────
    print("\n[B] VAD_dataset.xlsx → EN 전처리")
    vad_en_docs = load_vad_en(VAD_PATH)
    print_stats(vad_en_docs, "VAD-EN")
    save_docs(vad_en_docs, "preprocessed_vad_en.json")

    # ── 최종 요약 ─────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  전처리 완료 · 2개 독립 객체 생성")
    print(f"{SEP}")
    print(f"  voc_en_docs  : {len(voc_en_docs):>6,}건  →  preprocessed_voc_en.json")
    print(f"  vad_en_docs  : {len(vad_en_docs):>6,}건  →  preprocessed_vad_en.json")
    print(f"\n  출력 디렉토리: {os.path.abspath(OUT_DIR)}")
    print(f"{SEP}\n")

    print("  [다음 단계] data_analysis.py 실행")
    print("    → EXP1~7 (SNA / LDA / KMeans / 조합) 분석 자동 수행")