# 🪸 Predicting Coral Bleaching Severity with Multilayer Perceptrons

> A machine learning tutorial exploring how network depth and width affect MLP performance — and what happens when the data has no signal to give.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](coral_bleaching_mlp_tutorial.ipynb)

---

## 📖 About This Tutorial

This project is a machine learning tutorial built for academic coursework. It uses a synthetic ocean climate dataset to teach the fundamentals of **Multilayer Perceptrons (MLPs)** with a focus on a question that matters to every practitioner:

> **How does the depth (number of layers) and width (neurons per layer) of an MLP affect what it can learn?**

The tutorial covers:
- What an MLP is and how forward/backpropagation works
- Feature preprocessing for neural networks (why scaling matters)
- Running controlled experiments varying depth and width
- Using 5-fold cross-validation for reliable evaluation
- Recognising when a dataset lacks predictive signal — and what to do about it
- Ethical considerations for ML in environmental monitoring

---

## 📁 Repository Structure

```
.
├── coral_bleaching_mlp_tutorial.ipynb   # Main Jupyter notebook (full code + commentary)
├── tutorial.html                        # Webpage tutorial (~2000 words)
├── realistic_ocean_climate_dataset.csv  # Dataset (500 ocean climate records)
├── README.md                            # This file
└── LICENSE                              # MIT Licence
```

---

## 🗂️ Dataset

**File:** `realistic_ocean_climate_dataset.csv`  
**Rows:** 500 weekly observations (350 with labelled bleaching severity)  
**Locations:** Red Sea · Great Barrier Reef · Caribbean Sea · Galápagos · South China Sea · Maldives · Hawaiian Islands

| Column | Type | Description |
|--------|------|-------------|
| `Date` | String | Observation date (weekly) |
| `Location` | String | Reef system name |
| `Latitude` / `Longitude` | Float | Geographic coordinates |
| `SST (°C)` | Float | Sea surface temperature (23.6–33.2°C) |
| `pH Level` | Float | Ocean pH (7.87–8.20) |
| `Bleaching Severity` | String | **Target:** Low / Medium / High (150 missing) |
| `Species Observed` | Integer | Count of marine species (54–171) |
| `Marine Heatwave` | Boolean | Whether a heatwave event was active |

> ⚠️ **Note:** Statistical analysis (ANOVA) reveals that the bleaching labels in this dataset are **not significantly related** to the measured features (all p > 0.85). This is a synthetic dataset used for teaching purposes, and the tutorial explicitly addresses this finding as a lesson in data quality assessment.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/[your-username]/[your-repo-name].git
   cd [your-repo-name]
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn scipy jupyter
   ```

3. **Launch the notebook**
   ```bash
   jupyter notebook coral_bleaching_mlp_tutorial.ipynb
   ```

4. **Run all cells** — the notebook is self-contained and will reproduce all figures.

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pandas` | ≥1.3 | Data loading and manipulation |
| `numpy` | ≥1.21 | Numerical computation |
| `scikit-learn` | ≥1.0 | MLP model, preprocessing, evaluation |
| `matplotlib` | ≥3.4 | Figures and plots |
| `seaborn` | ≥0.11 | Correlation heatmap |
| `scipy` | ≥1.7 | ANOVA statistical tests |

---

## 📊 What the Notebook Covers

| Part | Title | Key Content |
|------|-------|-------------|
| 1 | Background | MLP architecture, ReLU, backpropagation, Universal Approximation Theorem |
| 2 | Setup | Imports and configuration |
| 3 | EDA | Class distributions, scatter plots, correlation heatmap, ANOVA signal test |
| 4 | Preprocessing | Label encoding, StandardScaler, train/test split |
| 5 | Baseline | DummyClassifier — the floor all models must beat |
| 6 | Depth Experiment | 5-fold CV across depths 1–5 (width fixed at 64) |
| 7 | Width Experiment | 5-fold CV across widths 8–256 (depth fixed at 2) |
| 8 | Comparison | Side-by-side depth vs. width figure |
| 9 | Best Model | Confusion matrix, classification report, loss curve |
| 10 | Interpretation | Why results are modest; what real-data results would look like |
| 11 | Summary & Ethics | Takeaways, ethical considerations, further reading |

---

## 📈 Key Results

All experiments used **5-fold cross-validation** for reliable estimates.

**Depth experiment** (width = 64 neurons per layer):

| Depth | CV Accuracy | ±Std |
|-------|-------------|------|
| 1 | 0.354 | 0.061 |
| 2 | 0.351 | 0.060 |
| 3 | 0.323 | 0.058 |
| 4 | 0.329 | 0.033 |
| 5 | 0.351 | 0.041 |
| **Baseline** | **0.377** | — |

**Width experiment** (depth = 2 hidden layers):

| Width | CV Accuracy | ±Std |
|-------|-------------|------|
| 8 | 0.374 | 0.038 |
| 16 | 0.357 | 0.009 |
| 32 | 0.363 | 0.066 |
| 64 | 0.351 | 0.060 |
| 128 | 0.320 | 0.033 |
| 256 | 0.349 | 0.032 |
| **Baseline** | **0.377** | — |

No configuration meaningfully outperforms the baseline — a direct consequence of the synthetic, signal-free dataset. This is explicitly taught as a lesson in data quality.

---

## 🔑 Key Lessons

1. **Always check your data before modelling** — a quick ANOVA revealed the core problem before any model was trained
2. **Feature scaling is non-negotiable** — fit StandardScaler on train only, never on test
3. **Establish a baseline first** — the majority-class classifier is your floor
4. **Cross-validation > single split** — especially with small datasets (<500 samples)
5. **Data quality > architecture** — depth and width cannot compensate for label-feature independence
6. **The Universal Approximation Theorem assumes signal exists** — approximating noise is still noise

---

## 📚 References

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. https://www.deeplearningbook.org/
2. Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals and Systems*, 2(4), 303–314.
3. Hughes, T. P. et al. (2017). Global warming and recurrent mass bleaching of corals. *Nature*, 543, 373–377. https://doi.org/10.1038/nature21707
4. Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830. https://scikit-learn.org/
5. Universal Approximation Theorem. Wikipedia. https://en.wikipedia.org/wiki/Universal_approximation_theorem

---

## ⚖️ Ethical Note

This tutorial uses a synthetic dataset and is for educational purposes only. Any real-world application of ML for coral reef monitoring should:
- Be validated on genuine field data from multiple reef systems
- Communicate model uncertainty to decision-makers, not just point predictions
- Follow ethical marine research protocols during data collection
- Account for geographic bias in training data

---

## 📄 Licence

This project is licensed under the **MIT Licence** — see [LICENSE](LICENSE) for details.

You are free to use, modify, and distribute this code and tutorial with attribution.

---

## 👤 Author

**[Tahsan Mahmud]**  
[University of Hertfordshire] · [MSc Data Science ]  
[https://github.com/Tahsan-Md]

---

*Built as part of a machine learning course assignment. GitHub repository: [your-repo-url]*

