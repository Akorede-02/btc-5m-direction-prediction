# BTC 5-Minute Direction Prediction

A time-series classification study of whether recent BTC price and return features can predict short-horizon price direction. The project compares Logistic Regression and XGBoost, then evaluates XGBoost with a forecast-horizon gap and expanding walk-forward validation.

**Key result:** the saved five-fold walk-forward run achieved **49.46% accuracy**, below the **53.48% validation-majority benchmark**. The tested features did not demonstrate a robust predictive signal.

[View the notebook](btc_5m_direction_prediction.ipynb)

## Data and approach

- **Data:** BTC-USD OHLCV downloaded with `yfinance`, requesting five days of one-minute observations. The saved run contains 6,847 rows before feature preparation.
- **Target:** `1` when the close five observations ahead exceeds the current close; `0` otherwise. Rows without a future close are removed, and the future close is excluded from the features.
- **Features:** current and lagged closes, one-period price change, returns and lagged returns, and five-observation rolling means.
- **Models:** scaled Logistic Regression and XGBoost with 100 estimators, depth 3, learning rate 0.05, and random seed 42. Scaling is fitted only on the training portion.
- **Evaluation:** exploratory chronological 80/20 splits, a split with a five-row gap, and five expanding walk-forward folds using `TimeSeriesSplit(n_splits=5, gap=5)`. The gap keeps training labels from reaching into the evaluation window.

## Recorded results

| Experiment | Accuracy |
|---|---:|
| Logistic Regression — price features | 53.14% |
| Logistic Regression — return features | 53.33% |
| XGBoost — return features | 53.04% |
| XGBoost — combined features, single split | 53.99% |
| XGBoost — combined features, five-row gap | 53.62% |
| **XGBoost — mean across five walk-forward folds** | **49.46%** |
| **Validation-majority benchmark — mean across folds** | **53.48%** |

The exploratory splits do not include a gap; the final walk-forward evaluation provides the stronger evidence. The notebook computes its benchmark from the larger class proportion **within each validation fold**. This is a retrospective class-balance reference, not a deployable baseline selected using training data. A future evaluation should also compare against a classifier that learns the majority class from each training fold.

## Run locally

Use Python 3.11 or newer in a virtual environment:

```bash
git clone https://github.com/Akorede-02/btc-5m-direction-prediction.git
cd btc-5m-direction-prediction
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the dependencies and open the notebook:

```bash
python -m pip install -r requirements.txt
jupyter lab btc_5m_direction_prediction.ipynb
```

Select the environment's Python kernel and run the cells from top to bottom. Internet access is required to download market data. On macOS, XGBoost may require the OpenMP runtime (`brew install libomp`).

The notebook and saved outputs are preserved from the original experiment. The historical data snapshot and original package lockfile are not included. `period="5d"` downloads a rolling recent window, so reruns will use different data and will not reproduce the recorded scores exactly. Dependencies are unpinned because the original versions were not recorded.

## Limitations and next steps

- Five days of data cover few market conditions and do not establish a stable trading edge.
- The target uses five **rows** ahead. Missing minute bars can make the elapsed forecast horizon longer than five minutes; timestamp continuity should be checked before treating it as an exact five-minute horizon.
- Features include the current close, so predictions assume that bar has finished.
- Accuracy alone does not measure economic value. Transaction costs, slippage, latency, and trade execution are not modeled.
- Useful extensions include an archived dataset, a training-derived baseline, longer evaluation periods, and volume or order-book features.

The main contribution is a transparent modeling workflow and an honest negative result under stricter time-series evaluation.
