# StockDB.AI — Bullish or Bearish AI

A CNN that reads a candlestick chart as an **image** and predicts whether the next day will be **bullish or bearish** for ES and NQ futures.

Instead of hand-coding indicators, the model learns straight from 50-candle chart images, so it picks up visual price-action patterns the way a trader does.

## How it works

| Script | What it does |
| --- | --- |
| `data.py` | Generates and labels the chart images (bullish/bearish by the next day's move) |
| `train.py` | Builds and trains the CNN, then saves the model |
| `predict.py` | Loads the model and predicts on a new chart |

## Tech

Python · PyTorch · NumPy · Pillow

## Setup

```bash
pip install torch torchvision numpy pillow
```

## Usage

```bash
python data.py        # build the dataset
python train.py       # train the model
python predict.py     # predict on a chart
```

## Results

- 90.84% training accuracy over 50 epochs
- Correctly called an unseen TradingView screenshot with ~90% confidence

## Notes

My first model overfit to how I generated the images, so I rebuilt the dataset with cleaner charts and retrained. The 90.84% is a training number — the next step is a proper held-out test set with no overlapping windows.

This is a personal ML project, **not financial advice**.

---

**Bryce Lombardo** · [@Bryce-L2](https://github.com/Bryce-L2)
