# voc2persona-llm

> **Does Structured Analysis Improve LLM-Generated Consumer Personas?**  
> A Comparative Study of SNA, LDA, and KMeans Prompt Injection  
> *(Anonymous submission — under review)*

---

## Overview

This repository contains the full experimental code, result files, and statistical analysis for a study comparing **nine prompt injection strategies** for LLM-based consumer persona generation.

The core question: *Does injecting structured text analysis results (SNA / LDA / KMeans) into LLM prompts improve persona quality compared to no-data baselines or raw review injection?*

---

## Repository Structure

```
voc2persona-llm/
│
├── src/
│   ├── exp_raw_pipeline.py       # EXP_RAW: raw review injection pipeline
│   ├── lda_only.py               # EXP2: LDA-only analysis
│   ├── segmataion_lda.py         # EXP4/5: SNA+LDA / LDA+KMeans pipelines
│   └── ttest_significance.py     # Statistical significance testing (Welch's t-test)
│
├── results/
│   ├── EXP1_SNA.json             # EXP1: SNA (Louvain) results
│   ├── EXP2_LDA.json             # EXP2: LDA results
│   ├── EXP3_KMeans.json          # EXP3: KMeans (TF-IDF) results
│   ├── EXP4_SNA_LDA.json         # EXP4: SNA → LDA results
│   ├── EXP5_LDA_KMeans.json      # EXP5: LDA → KMeans results
│   ├── EXP6_SNA_KMeans.json      # EXP6: SNA → KMeans results
│   ├── EXP7_Full.json            # EXP7: SNA → LDA → KMeans (full pipeline)
│   ├── EXP_RAW_stats.json        # EXP_RAW: raw review injection stats
│   ├── final_table_3.json        # All conditions: mean ± std (N=10)
│   ├── param_gridsearch_result.json  # Grid search results (SNA / LDA / KMeans)
│   ├── summary.json              # Summary of analysis method metrics
│   ├── ttest_results.json        # Welch's t-test results (H1 / H2 / H3)
│   └── ttest_results.txt         # Human-readable t-test output
│
├── prompts/
│   └── prompting_ABSA_.json      # ABSA-based prompt templates
│
├── data/
│   ├── preprocessed_voc_en.json  # Preprocessed voc_en tokens + metadata
│   ├── preprocessed_vad_en.json  # Preprocessed vad_en tokens + metadata
│   ├── synthetic_gt_en.json      # Ground truth personas (voc_en, expert-synthesized)
│   ├── en_external_gt.json       # Ground truth personas (vad_en, ABSA-derived)
│   ├── vad_consumer_personas.json    # vad_en consumer personas
│   └── kr_appliance_personas_en.json # voc_en consumer personas (EN)
│
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Experimental Conditions

| ID | Method | Description | Data Coverage |
|----|--------|-------------|--------------|
| BASELINE | GPT-4o only | No data injection | — |
| EXP_RAW | Raw review injection | Up to 200 sampled reviews | Partial (sampled) |
| EXP1 | SNA | Louvain community keywords | Full dataset |
| EXP2 | LDA | Topic-word distributions | Full dataset |
| EXP3 | KMeans | TF-IDF cluster centroids | Full dataset |
| EXP4 | SNA → LDA | SNA filtering + LDA | Full dataset |
| EXP5 | LDA → KMeans | LDA labeling + KMeans | Full dataset |
| EXP6 | SNA → KMeans | SNA filtering + KMeans | Full dataset |
| EXP7 | SNA → LDA → KMeans | Full 3-method pipeline | Full dataset |

---

## Datasets

| Dataset | Domain | Size | K | Type |
|---------|--------|------|---|------|
| `voc_en` | Korean home appliances (VOC, EN) | 965 reviews | 4 | Top-Down |
| `vad_en` | Virtual assistant devices (EN) | 2,370 reviews | 6 | Bottom-Up |

- **voc_en**: Consumer reviews collected from public e-commerce platforms. Pre-labeled with 4 expert-defined segments (S1–S4). K=4 fixed by domain design (Salminen et al., 2022).
- **vad_en**: English product reviews from major U.S. retailers (Walmart, Target, BestBuy). K=6 selected via LDA Coherence C_v + KMeans Silhouette grid search.

---

## Evaluation Metrics

Three complementary metrics are used to evaluate generated personas against ground-truth (GT) personas:

**Matching Score** — GPT-4o (LLM-as-Judge) scores 5 alignment dimensions per persona pair:
- `keyword_alignment` — topic keyword overlap
- `value_alignment` — core value direction
- `pain_alignment` — pain point similarity
- `behavioral_alignment` — description + motivation match
- `sentiment_alignment` — sentiment ratio direction

Hungarian algorithm optimal matching is applied before averaging.

**Validity Score** — GPT-4o assesses whether persona content is grounded in domain topics (0 = fully fabricated, 1 = fully grounded).

**Topic Similarity** — Cosine similarity between `top_aspects` of generated and GT personas using `text-embedding-3-small` (dim=1536).

---

## Key Results (N=10 runs, mean ± std)

| Method | voc Matching | voc Validity | vad Matching | vad Validity |
|--------|-------------|-------------|-------------|-------------|
| BASELINE | .238 ± .032 | .507 ± .071 | .332 ± .039 | .518 ± .108 |
| EXP_RAW | .525 ± .052† | .775 ± .055† | .421 ± .059† | .680 ± .059† |
| EXP3 (KMeans) ★ | **.572 ± .038†** | **.865 ± .045†** | **.505 ± .054†** | **.733 ± .056†** |
| EXP7 (Full) | .452 ± .045† | .763 ± .065† | .424 ± .073† | .655 ± .084† |

† p < 0.001 vs. BASELINE (Welch's t-test). ★ Best performance.

**Key Finding (H3):** EXP3 (KMeans alone) significantly outperforms EXP7 (full SNA+LDA+KMeans pipeline) on Matching and Validity across both datasets — demonstrating an **Information Overload Effect** in structured prompt engineering.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up API key and paths

Edit the config section at the top of each script:

```python
# In exp_raw_pipeline.py
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"
BASE_DIR     = "./output"
VOC_CSV_PATH = "./data/merged_voc.csv"
VAD_XLSX_PATH = "./data/Virtual_Assistant_Devices.xlsx"
```

### 3. Run EXP_RAW experiment

```bash
# Both datasets, 10 runs
python src/exp_raw_pipeline.py --runs 10

# Single dataset
python src/exp_raw_pipeline.py --files voc_en --runs 10
python src/exp_raw_pipeline.py --files vad_en --runs 10

# Statistics only (if runs already completed)
python src/exp_raw_pipeline.py --stats-only
```

### 4. Run statistical significance tests

```bash
python src/ttest_significance.py
```

Output files: `ttest_results.json`, `ttest_results.txt`

---

## Hyperparameter Grid Search Results

Optimal parameters selected by grid search (see `results/param_gridsearch_result.json`):

| Dataset | Method | Best Params | Metric |
|---------|--------|-------------|--------|
| voc_en | SNA | window=2, min_freq=7 | Coherence C_v = 0.5706 |
| voc_en | LDA | K=4, passes=30 | Top-Down design |
| voc_en | KMeans | K=4, max_features=200 | Silhouette = 0.0486 |
| vad_en | LDA | K=6, passes=30 | Coherence C_v = 0.5323 |
| vad_en | KMeans | K=6, max_features=200 | Silhouette = 0.0474 |

---

## Prompt Schema

All conditions share a unified JSON output schema for persona generation:

```json
{
  "personas": [{
    "persona_id": "P1",
    "persona_name": "<consumer type name>",
    "keywords": ["5-7 keywords from reviews"],
    "core_value": "<1 sentence>",
    "description": "<2-3 sentences>",
    "motivation": "<1-2 sentences>",
    "pain_points": "<1-2 sentences>",
    "quote": "<representative review phrase>",
    "sentiment_ratio": {
      "Positive": 0.0, "Neutral": 0.0, "Negative": 0.0,
      "_inference_note": "<reasoning>"
    },
    "top_aspects": ["5 key aspects"],
    "co_aspects": {"aspect": ["co-occurring aspects"]}
  }]
}
```

---

## Statistical Testing

Welch's two-sided t-test (α = 0.05) with Cohen's d effect size.

- **H1**: All data-injection conditions vs. BASELINE → All significant (p < 0.001)
- **H2**: EXP3 vs. EXP_RAW → Significant on Matching (both datasets), Validity (voc_en)
- **H3**: EXP3 vs. EXP7 → Significant on Matching and Validity (both datasets)

Full results: `results/ttest_results.json`

---

## Requirements

```
openai>=1.0.0
gensim>=4.0.0
scikit-learn>=1.0.0
networkx>=3.0
python-louvain>=0.16
scipy>=1.9.0
numpy>=1.23.0
pandas>=1.5.0
openpyxl>=3.0.0
```

---

## License

This repository is made available for anonymous peer review purposes only.  
The code and data will be released under MIT License upon acceptance.

---

## Citation

Anonymous (under review). *Does Structured Analysis Improve LLM-Generated Consumer Personas? A Comparative Study of SNA, LDA, and KMeans Prompt Injection.*
