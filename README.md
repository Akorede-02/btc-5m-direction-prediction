# BTC 5-Minute Direction Prediction

A time-series classification study of whether recent BTC price and return features can predict short-horizon price direction. The project compares Logistic Regression and XGBoost, then evaluates XGBoost with a five-row forecast-horizon gap and expanding walk-forward validation.

**Key result (original run):** the saved five-fold walk-forward run achieved **49.46% accuracy**, below the **53.48% retrospective validation-majority reference**. This did not establish a robust predictive signal. The revised notebook calculates a training-derived majority baseline; its results have not been reproduced on the original snapshot.

[View the notebook](btc_5m_direction_prediction.ipynb)

## Data and approach

- **Data:** BTC-USD OHLCV downloaded with `yfinance`, requesting five days of one-minute observations. The saved run contains 6,847 rows before feature preparation.
- **Target:** `1` when the close five observations ahead exceeds the current close; `0` otherwise. Rows without a future close are removed. The future close and timestamp are excluded from model inputs. The notebook now checks timestamp continuity and reports how many targets span exactly five minutes.
- **Features:** current and lagged closes, one-period price change, returns and lagged returns, and five-observation rolling means.
- **Models:** scaled Logistic Regression and XGBoost with 100 estimators, depth 3, learning rate 0.05, and random seed 42. Scaling is fitted only on the training portion.
- **Evaluation:** exploratory chronological 80/20 splits, a split with a five-row gap, and five expanding walk-forward folds using `TimeSeriesSplit(n_splits=5, gap=5)`. The gap separates training and evaluation observations. The gap-based split and walk-forward folds now also assert that the last training target timestamp is strictly before the first evaluation observation.

## Historical results (original run)

| Experiment | Accuracy |
|---|---:|
| Logistic Regression — price features | 53.14% |
| Logistic Regression — return features | 53.33% |
| XGBoost — return features | 53.04% |
| XGBoost — combined features, single split | 53.99% |
| XGBoost — combined features, five-row gap | 53.62% |
| **XGBoost — mean across five walk-forward folds** | **49.46%** |
| Validation-majority reference — mean across folds | 53.48% |
| Training-derived majority baseline (corrected code) | Not rerun on original data |

The exploratory splits do not include a gap and may contain target overlap. In the revised walk-forward code, the majority class is learned from each training fold and predicted on the corresponding validation fold. The validation-majority proportion is retained separately as a retrospective class-balance reference, not a deployable baseline. The notebook produces a per-fold comparison and the mean difference between model and training-derived baseline. The original scores above are historical, not results from a fresh run of the corrected code.

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

The historical data snapshot and original package lockfile are not included. The unchanged saved outputs and historical results table refer to the original experiment. The corrected evaluation and timestamp checks have not been rerun on that snapshot. `period="5d"` downloads a rolling recent window, so reruns will use different data and will not reproduce the recorded scores exactly. Dependencies are unpinned because the original versions were not recorded.

## Limitations and next steps

- Five days of data cover few market conditions and do not establish a stable trading edge.
- The target uses five **rows** ahead. Missing minute bars can make the elapsed forecast horizon longer than five minutes; the revised notebook reports non-one-minute intervals and actual elapsed target horizons. The original row-based target is intentionally retained.
- Features include the current close, so predictions assume that bar has finished.
- Accuracy alone does not measure economic value. Transaction costs, slippage, latency, and trade execution are not modeled.
- Useful extensions include archiving a fixed dataset and dependency versions, defining an exact-time target, testing on a later untouched period, and exploring volume or order-book features.

The main contribution is a transparent modeling workflow and an honest inconclusive historical result under stricter time-series evaluation. A fresh run of the corrected evaluation is required to determine performance relative to the training-derived baseline.
