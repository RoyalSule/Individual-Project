import pandas as pd
import numpy as np
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:, 1]
    preds = (preds >= 0.60).astype(int)
    preds = pd.Series(preds, index=test.index, name="Predictions")
    return pd.concat([test["Target"], preds], axis=1)


def backtest(data, model, predictors, start=2500, step=250):
    # Walk-forward validation — never trains on future data
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()
        all_predictions.append(predict(train, test, predictors, model))
    return pd.concat(all_predictions)

def add_returns(predictions, data):
    # In on predicted 1, out on predicted 0
    predictions = predictions.copy()
    predictions["Returns"] = data["Close"].pct_change()
    predictions["Strategy"] = predictions["Returns"] * predictions["Predictions"].shift(1)
    return predictions


def sharpe_ratio(predictions):
    daily_returns = predictions["Strategy"]
    if daily_returns.std() == 0:
        return 0.0
    return round((daily_returns.mean() / daily_returns.std()) * np.sqrt(252), 4)

def ma_crossover_returns(predictions, data):
    ma50 = data['Close'].rolling(50).mean()
    ma200 = data['Close'].rolling(200).mean()
    ma_signal = (ma50 > ma200).astype(int)
    return predictions['Returns'] * ma_signal.shift(1).reindex(predictions.index)

def summary_metrics(predictions, model_name, data):
    ma_returns = ma_crossover_returns(predictions, data)
    return {
        "Model": model_name,
        "Precision": round(precision_score(predictions["Target"], predictions["Predictions"]), 4),
        "Sharpe Ratio": float(sharpe_ratio(predictions)),
        "Strategy Return": float(round(predictions["Strategy"].cumsum().iloc[-1], 4)),
        "Buy & Hold Return": float(round(predictions["Returns"].cumsum().iloc[-1], 4)),
        "MA Crossover Return": float(round(ma_returns.cumsum().iloc[-1], 4)),
        "Number of Trades": int(predictions["Predictions"].sum())
    }


def plot_equity_curve(predictions, model_name, data):
    ma_returns = ma_crossover_returns(predictions, data)
    ax = predictions[['Returns', 'Strategy']].cumsum().plot(
        figsize=(14, 5),
        title=f'Equity Curve — {model_name}',
        color=['blue', 'orange']
    )
    ma_returns.cumsum().plot(ax=ax, color='green', label='MA Crossover')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.legend(['Buy & Hold', model_name, 'MA Crossover'])
    plt.tight_layout()
    plt.show()

def plot_prediction_distribution(predictions, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    predictions['Predictions'].value_counts().sort_index().plot.bar(
        ax=axes[0], color=['red', 'green'],
        title='Predicted', rot=0
    )
    predictions['Target'].value_counts().sort_index().plot.bar(
        ax=axes[1], color=['red', 'green'],
        title='Actual', rot=0
    )
    fig.suptitle(f'Prediction Distribution — {model_name}')
    plt.tight_layout()
    plt.show()