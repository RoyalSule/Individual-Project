# Evaluating Machine Learning Algorithms for Stock Market Prediction

## Overview
This project develops and evaluates machine learning models to predict the daily direction of the S&P 500 index — whether the market will close higher or lower the next trading day. Five classification algorithms are trained on historical price data and technical indicators, then evaluated using a walk-forward backtesting system. The models are compared against each other and against a traditional moving average crossover strategy.

## Research Question
*"Do machine learning models outperform traditional trading strategies for S&P 500 trend prediction?"*

## Objectives
- Download and clean S&P 500 historical data via Yahoo Finance
- Engineer features using price/volume ratios and technical indicators (EMA, MACD, ATR)
- Build and train five ML models using a consistent pipeline
- Backtest each model using walk-forward validation
- Compare models across precision, Sharpe ratio, total return and equity curve

## Project Structure
    notebooks/
    src/
        data_loader.py
        evaluate.py
    models/
    requirements.txt
    README.md

## Installation
```bash
git clone <repo-url>
cd Individual-Project
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## How to Run
Run the notebooks in order:
1. `01_logistic_regression.ipynb`
2. `02_mlp.ipynb`
3. `03_knn.ipynb`
4. `04_random_forest.ipynb`
5. `05_svm.ipynb`
6. `06_comparison.ipynb` — loads all saved predictions and produces the final comparison

Each notebook handles its own data loading, backtesting and evaluation. Run all cells top to bottom using `Kernel > Restart & Run All`.

## Models
| Model | Notebook |
|-------|----------|
| Logistic Regression | `01_logistic_regression.ipynb` |
| MLP | `02_mlp.ipynb` |
| KNN | `03_knn.ipynb` |
| Random Forest | `04_random_forest.ipynb` |
| SVM | `05_svm.ipynb` |

## Ethical Considerations
This project is conducted for academic purposes only and does not constitute financial or investment advice. All data used is publicly available via Yahoo Finance.

---
Royal Sule · University of Leeds · 2026