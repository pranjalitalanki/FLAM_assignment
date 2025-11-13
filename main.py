import numpy as np
import pandas as pd

print("=== FLAM Assignment: Loading Data ===")

# Load the CSV using default settings for comma-separated with header
try:
    data = pd.read_csv('xy_data.csv')
    print("✓ Successfully loaded CSV data")
    print(f"Data columns: {list(data.columns)}")
    print(f"Data types: {data.dtypes}")
except Exception as e:
    print(f"Error loading CSV: {e}")
    print("Make sure xy_data.csv is in the same folder as this script")
    exit()

# Verify data loaded correctly
print(f"\nDataset shape: {data.shape[0]} rows, {data.shape[1]} columns")
print(f"x range: {data['x'].min():.2f} to {data['x'].max():.2f}")
print(f"y range: {data['y'].min():.2f} to {data['y'].max():.2f}")

print("\nFirst 5 data points:")
print(data.head())

# Generate t values for 6 < t < 60
N = len(data)
t = np.linspace(6.1, 59.9, N)
print(f"\nGenerated {N} t values from {t.min():.2f} to {t.max():.2f}")

x_obs = data['x'].values
y_obs = data['y'].values

print(f"\nSample data points with t:")
for i in range(min(5, N)):
    print(f"t={t[i]:.2f}: x={x_obs[i]:.2f}, y={y_obs[i]:.2f}")

print("\nData loading complete! Ready for curve implementation.")
