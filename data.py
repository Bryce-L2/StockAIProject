import yfinance as yf
import mplfinance as mpf
import os
import matplotlib

# Creates folders to store chart images
os.makedirs("dataset/bullish", exist_ok=True)
os.makedirs("dataset/bearish", exist_ok=True)

# Downloads 2 years of daily price data for ES and NQ futures
print("Downloading data...")
es_data = yf.download("ES=F", period="2y", interval="1d", auto_adjust=True)
nq_data = yf.download("NQ=F", period="2y", interval="1d", auto_adjust=True)

# Fix column structure from newer yfinance versions
if isinstance(es_data.columns, type(es_data.columns)) and hasattr(es_data.columns, 'levels'):
    es_data.columns = es_data.columns.get_level_values(0)
    nq_data.columns = nq_data.columns.get_level_values(0)

# Keeps only data I need and forces them to be numbers
es_data = es_data[["Open", "High", "Low", "Close", "Volume"]].astype(float)
nq_data = nq_data[["Open", "High", "Low", "Close", "Volume"]].astype(float)

print(f"ES candles: {len(es_data)}")
print(f"NQ candles: {len(nq_data)}")

# Uses green and red candles for clearer visual signal
custom_style = mpf.make_mpf_style(
    base_mpf_style='charles',
    marketcolors=mpf.make_marketcolors(
        up='green',
        down='red',
        edge='inherit',
        wick='inherit'
    )
)

# Tracks how many images generated for each label
count = {"bullish": 0, "bearish": 0}

def generate_charts(data, symbol):
    window = 50

    for i in range(window, len(data) - 1):
        # Grabs 50 candles at a time
        chunk = data.iloc[i - window:i]

        # Looks at what price did the next day to label the chart
        next_close = float(data.iloc[i]["Close"])
        current_close = float(data.iloc[i - 1]["Close"])

        # Labels
        label = "bullish" if next_close > current_close else "bearish"

        # Builds the file path for saving the image
        filename = f"dataset/{label}/{symbol}_{i}.png"

        try:
            # Draws and save the candlestick chart as a PNG
            mpf.plot(chunk, type='candle', style=custom_style,
                     savefig=filename, figsize=(6, 6))
            count[label] += 1
        except Exception as e:
            # Error traps 
            print(f"Error at {i}: {e}")
            continue

# Runs
print("Generating ES charts...")
generate_charts(es_data, "ES")

print("Generating NQ charts...")
generate_charts(nq_data, "NQ")

print(f"Done! Generated {count['bullish']} bullish and {count['bearish']} bearish charts")
