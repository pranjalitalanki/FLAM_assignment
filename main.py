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

def parametric_curve(t, theta, M, X):
    
    
   
    theta_rad = np.deg2rad(theta)
    
    
    abs_t = np.abs(t)
    exp_term = np.exp(M * abs_t)  
    sin_03t = np.sin(0.3 * t)     
    
    
    x = (t * np.cos(theta_rad) - 
         exp_term * sin_03t * np.sin(theta_rad) + X)
    
    
    y = (42 + t * np.sin(theta_rad) + 
         exp_term * sin_03t * np.cos(theta_rad))
    
    return x, y


test_theta = 25.0   
test_M = 0.0        
test_X = 50.0        

print(f"Testing with θ={test_theta}°, M={test_M}, X={test_X}")


x_test, y_test = parametric_curve(t[:10], test_theta, test_M, test_X)

print("\nFirst 5 predicted points from curve (t, x_pred, y_pred):")
for i in range(5):
    print(f"t={t[i]:.2f}: x={x_test[i]:.2f}, y={y_test[i]:.2f}")


print("\nComparison with actual data points:")
for i in range(min(5, N)):
    x_diff = abs(x_test[i] - x_obs[i])
    y_diff = abs(y_test[i] - y_obs[i])
    print(f"t={t[i]:.2f}: actual(x={x_obs[i]:.2f}, y={y_obs[i]:.2f}) "
          f"vs pred(x={x_test[i]:.2f}, y={y_test[i]:.2f}) "
          f"[dx={x_diff:.2f}, dy={y_diff:.2f}]")

