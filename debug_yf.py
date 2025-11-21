import yfinance as yf
import pandas as pd
import lgb_forecast_v2
from lgb_forecast_v2 import load_and_process_data, create_features_and_targets

print("lgb_forecast file:", lgb_forecast_v2.__file__)

# print("Testing yfinance download...")
# try:
#     # Try downloading a small chunk first
#     sp500 = yf.download("^GSPC", period="1mo", progress=False)
#     print("Download result shape:", sp500.shape)
#     print(sp500.head())
# except Exception as e:
#     print("Download failed:", e)

print("\nTesting load_and_process_data...")
df = load_and_process_data("Electric_Production.csv")
print("Columns:", df.columns)
print("SP500 stats:\n", df['SP500'].describe())

print("\nTesting feature generation...")
X, y, dates, feature_names = create_features_and_targets(df)
print("Feature names:", feature_names)
print("X shape:", X.shape)
