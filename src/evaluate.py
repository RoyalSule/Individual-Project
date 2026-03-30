import pandas as pd
import numpy as np
from sklearn.metrics import precision_score

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis= 1)
    return combined

def backtest(data, model, predictors, start=2500, step=250):
   all_predictions = []
   for i in range(start, data.shape[0], step):
       train = data.iloc[0:i].copy()
       test = data.iloc[i:(i+step)].copy()
       predictions = predict(train, test, predictors, model)
       all_predictions.append(predictions)
   return pd.concat(all_predictions)

