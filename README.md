# Stock Returns: Chronos vs. GARCH

**Advanced Simulation Methods | Rutgers University | Fall 2026**

## 1. Project Overview

This project investigates whether a time-series language model can compete with traditional statistical models in probabilistic financial return forecasting, with particular emphasis on volatility prediction and distributional accuracy.

## 2. Research Questions

- Can Chronos produce predictive distributions comparable to GARCH(1,1)?
- Does fine-tuning Chronos on financial returns improve forecasting performance?
- Can Chronos successfully capture volatility clustering?
- How does forecasting performance differ between value-weighted and equal-weighted portfolios?
- Are differences in model performance statistically significant?

## 3. Data

### 3.1 Data Source
- Kenneth French Data Library — 49 Industry Portfolios (Daily)
- Value-weighted returns
- Equal-weighted returns

### 3.2 Data Preprocessing
- Missing-value handling
- Time-series separation
- Return scaling
- Exploratory data analysis

### 3.3 Data Partitioning

| Dataset | Period |
|---|---|
| Training | 1926–2010 |
| Validation | 2011–2015 |
| Testing | 2016 onward |

## 4. Forecasting Models

### 4.1 Random Walk
- Historical volatility estimation
- Zero-mean predictive distribution

### 4.2 GARCH(1,1)
- Student-t innovations
- Rolling estimation window
- Monthly parameter refitting
- Daily volatility forecasting

### 4.3 Zero-Shot Chronos
- Pretrained Chronos-T5-Mini
- Forecasting without additional training

### 4.4 Fine-Tuned Chronos
- Model architecture
- Scaling and tokenization
- Training objective
- Hyperparameter selection
- Training and validation loss
- Checkpoint selection

## 5. Experimental Methodology

### 5.1 Rolling Forecasting Framework
### 5.2 One-Day-Ahead Predictive Distributions
### 5.3 Prevention of Data Leakage
### 5.4 Computational Resources
### 5.5 Training Experiments

## 6. Evaluation Metrics

- Continuous Ranked Probability Score (CRPS)
- 90% prediction interval coverage
- 95% prediction interval coverage
- Average prediction interval width
- QLIKE volatility loss
- Directional accuracy

## 7. Statistical Inference

- Paired forecasting loss comparisons
- Diebold–Mariano test and/or block bootstrap
- Temporal and cross-industry dependence
- Statistical significance of performance differences

## 8. Results

### 8.1 Overall Model Comparison

| Model | CRPS | 90% Coverage | 95% Coverage | QLIKE |
|---|---|---|---|---|
| Random Walk | TBD | TBD | TBD | TBD |
| GARCH(1,1) | TBD | TBD | TBD | TBD |
| Zero-Shot Chronos | TBD | TBD | TBD | TBD |
| Fine-Tuned Chronos | TBD | TBD | TBD | TBD |

### 8.2 Value-Weighted vs. Equal-Weighted Returns
### 8.3 Industry-Level Performance
### 8.4 Performance Across Volatility Regimes
### 8.5 Volatility Forecasting Through 2020
### 8.6 Directional Predictability
### 8.7 Statistical Significance

## 9. Discussion

- Interpretation of experimental results
- Strengths and weaknesses of each model
- Benefits of financial fine-tuning
- Training data limitations
- Computational limitations
- Potential sources of bias

## 10. Conclusion and Future Work

## 11. Repository Structure

```text
Project-4/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── preprocess.py
│   ├── random_walk.py
│   ├── garch.py
│   ├── chronos_zero_shot.py
│   ├── chronos_finetune.py
│   ├── rolling_evaluation.py
│   ├── metrics.py
│   ├── statistical_tests.py
│   └── visualizations.py
├── notebooks/
├── checkpoints/
├── results/
│   ├── forecasts/
│   ├── tables/
│   ├── figures/
│   └── training_logs/
├── report/
├── presentation/
├── config.yaml
├── requirements.txt
├── .gitignore
└── README.md
```

## 12. Installation and Reproducibility

### 12.1 Environment Setup
### 12.2 Dependencies
### 12.3 Data Preparation
### 12.4 Model Training
### 12.5 Running the Evaluation
### 12.6 Reproducing Results

## 13. References

- Kenneth French Data Library
- Chronos: Learning the Language of Time Series (Ansari et al., 2024)
- GARCH (Bollerslev, 1986)
- Comparing Predictive Accuracy (Diebold and Mariano, 1995)
- Additional references

## 14. Contributors

*To be completed.*
