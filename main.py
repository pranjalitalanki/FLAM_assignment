import numpy as np
import pandas as pd



# Load the CSV 
try:
    data = pd.read_csv('xy_data.csv')
    
    print(f"Data columns: {list(data.columns)}")
    print(f"Data types: {data.dtypes}")
except Exception as e:
    print(f"Error loading CSV: {e}")
    
    exit()


print(f"\nDataset shape: {data.shape[0]} rows, {data.shape[1]} columns")
print(f"x range: {data['x'].min():.2f} to {data['x'].max():.2f}")
print(f"y range: {data['y'].min():.2f} to {data['y'].max():.2f}")


print(data.head())


N = len(data)
t = np.linspace(6.1, 59.9, N)
print(f"\nGenerated {N} t values from {t.min():.2f} to {t.max():.2f}")

x_obs = data['x'].values
y_obs = data['y'].values


for i in range(min(5, N)):
    print(f"t={t[i]:.2f}: x={x_obs[i]:.2f}, y={y_obs[i]:.2f}")


