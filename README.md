#  Weibull Reliability Analyzer

Statistical reliability analysis of NASA CMAPSS turbofan engine degradation data using Weibull distribution fitting, survival analysis, and interactive visualization.

** Live App:** [https://weibull-reliability-analyzer-l2lfgrgjrdznc8nd55dgu9.streamlit.app/]

## Overview

This project applies classical reliability engineering methods to real-world run-to-failure sensor data from 100 turbofan engines (NASA CMAPSS FD001 dataset). It estimates failure time distributions, computes industry-standard reliability metrics, and visualizes the complete reliability profile of the fleet.

## Key Results

| Metric | Value |
|---|---|
| Shape parameter (β) | 4.41 — confirms wear-out failure mode |
| Scale parameter (η) | 225.0 cycles |
| MTTF | 205.1 cycles |
| B10 Life | 135.1 cycles (maintenance threshold) |
| B50 Life | 207.1 cycles |
| 2-parameter Weibull fit | R² = 0.894 |
| 3-parameter Weibull fit | R² = 0.974 (improved with threshold parameter γ) |

## Methodology

1. Extracted failure time (max operational cycle) per engine from sensor time-series
2. Fit 2-parameter and 3-parameter Weibull distributions via Maximum Likelihood Estimation
3. Validated fit using Weibull probability plots (linearized CDF) with median rank regression
4. Computed PDF, CDF, Reliability function R(t), and Hazard function h(t)
5. Derived MTTF and B-life (B10/B50/B90) metrics for maintenance planning

## Key Finding

The 2-parameter Weibull fit showed mild S-curvature at both tails (R²=0.89), suggesting population heterogeneity. Adding a location/threshold parameter γ (3-parameter Weibull) improved fit to R²=0.97 — consistent with a burn-in period before wear-out degradation begins, a physically meaningful result for turbofan engines.

## Tech Stack

Python · NumPy · SciPy · Pandas · Matplotlib · Streamlit · Plotly

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Project Structure

```
weibull_reliability/
├── data/              # NASA CMAPSS FD001 dataset
├── notebooks/         # Full analysis notebook
├── app/app.py         # Interactive Streamlit dashboard
└── requirements.txt
```
