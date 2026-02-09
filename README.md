# Stock Market Prediction Using Machine Learning

**Overview**

This project develops and evaluates machine learning models to predict short and long term trends in the S&P 500 index. The models are trained, using historical price data, technical indicators (e.g. EMA, MACD, RSI) and news sentiment data. I implement a acktesting system to evaluate the accuracy and performance of each model.

**Objectives**
* Download and clean data
* Feature engineering using technical indicators
* Build and train the models
* Backtest and evaluate performance
* Compare models

**Structure**
data/  
    processed/
    raw/
src/        
notebooks/  
results/    

**Installation**
git clone <repo-url>
cd INDIVIDUAL-PROJECT
python -m venv venv
source venv/bin/activate # On Linux
pip install -r requirements.txt

**How to run**
python src/data_loader.py
python src/models.py
python src/backtesting.py

**Models**
* Logistic Regression
* Random Forest
* Support Vector Machine (SVM)
* K Nearest Neighbours (KNN)
* Multi Layer Perceptron (MLP)

**Ethical considerations**
This project is conducted for academic purposes only and does not consitute financial advice. All data used is publicly available.

Royal Sule
Univeristy of Leeds
2026


