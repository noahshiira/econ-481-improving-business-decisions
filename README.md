# Replication: Smart Green Nudging (von Zahn et al., 2024)

A replication of **"Smart Green Nudging: Reducing Product Returns Through Digital Footprints and Causal Machine Learning"** published in *Marketing Science*.

> von Zahn, M., Feuerriegel, S., & Kuehl, N. (2024). https://doi.org/10.1287/mksc.2022.0393

---

## Overview

This notebook replicates all four core components of the paper using **simulated data**:

| Section | Method |
|---|---|
| Field Experiment & ATE | OLS (naive + covariate-adjusted) + IPW |
| Digital Footprint Features | Behavioral feature engineering |
| Heterogeneous Treatment Effects | `econml` CausalForestDML |
| Targeting Policy & Profit | Policy curve, value of personalization |

Each section includes an **LLM interpretation** via `litellm` + Llama3.2 running locally via Ollama.

---

## Setup

### 1. Install dependencies

```bash
pip install econml litellm pandas numpy matplotlib seaborn scikit-learn statsmodels
```

### 2. Start Ollama with Llama3.2

```bash
# Install Ollama: https://ollama.ai
ollama pull llama3.2
ollama serve   # runs on localhost:11434 by default
```

### 3. Add your data

Place your simulation CSVs in a `data/` folder:

```
data/
  train.csv    # Customer features + treatment + return outcome
  test.csv
```

Expected columns (at minimum):
- `treatment` — binary (1 = nudge shown, 0 = control)
- `returned` — binary outcome (1 = item returned)
- Any numeric customer/order feature columns

Update `TREATMENT_COL`, `OUTCOME_COL`, and `FEATURE_COLS` at the top of Section 1 if your column names differ.

### 4. Run the notebook

```bash
jupyter notebook replication_smart_green_nudging.ipynb
```

---

## Structure

```
.
├── replication_smart_green_nudging.ipynb   # Main replication notebook
├── data/
│   ├── train.csv                           # Simulated training data (not committed)
│   └── test.csv                            # Simulated test data (not committed)
├── README.md
└── requirements.txt
```

---

## LLM Integration

The notebook uses `litellm` to call a local **Llama3.2** model (via Ollama) at the end of each section. If Ollama is not running, the notebook still executes fully — LLM cells print a fallback message instead.

```python
import litellm
response = litellm.completion(
    model="ollama_chat/llama3.2",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0,
)
```

---

## Key Results (example with simulated data)

- **ATE**: Green nudge reduces return probability by ~X pp (OLS + controls)
- **CATE heterogeneity**: ~Y% of customers show beneficial nudge effects
- **Value of personalization**: Smart targeting outperforms universal nudging by €Z per cohort

*Actual values depend on your simulation parameters.*

---

## References

- von Zahn et al. (2024) — *Marketing Science* — https://doi.org/10.1287/mksc.2022.0393
- Wager & Athey (2018) — Causal Forests — *JASA*
- EconML library — https://econml.azurewebsites.net/
- LiteLLM — https://docs.litellm.ai/
