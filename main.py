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

def l1_loss(params, t, x_obs, y_obs):
 
    theta, M, X = params
    
   
    x_pred, y_pred = parametric_curve(t, theta, M, X)
    
   
    x_errors = np.abs(x_obs - x_pred)
    y_errors = np.abs(y_obs - y_pred)
    
    
    total_loss = np.sum(x_errors) + np.sum(y_errors)
    
    return total_loss


initial_params = [25.0, 0.0, 50.0]
test_loss = l1_loss(initial_params, t, x_obs, y_obs)

print(f"L1 loss with initial guess (θ=25°, M=0, X=50): {test_loss:.2f}")
print(f"Average error per data point: {test_loss/(2*N):.4f} (x + y components)")


print("\nTesting parameter sensitivity:")
print("- Original loss:", test_loss)


loss_theta_10 = l1_loss([10.0, 0.0, 50.0], t, x_obs, y_obs)
loss_theta_40 = l1_loss([40.0, 0.0, 50.0], t, x_obs, y_obs)
print(f"- θ=10° loss: {loss_theta_10:.2f} (change of {abs(loss_theta_10 - test_loss):.2f})")
print(f"- θ=40° loss: {loss_theta_40:.2f} (change of {abs(loss_theta_40 - test_loss):.2f})")



bounds = [
    (0.1, 49.9),   
    (-0.049, 0.049), 
    (0.1, 99.9)   
]


initial_guess = [25.0, 0.0, 50.0] 

print(f"Optimization bounds:")
print(f"- θ (degrees): {bounds[0][0]:.1f} to {bounds[0][1]:.1f}")
print(f"- M: {bounds[1][0]:.4f} to {bounds[1][1]:.4f}")
print(f"- X: {bounds[2][0]:.1f} to {bounds[2][1]:.1f}")

print(f"\nInitial parameter guess: θ={initial_guess[0]:.1f}°, M={initial_guess[1]:.4f}, X={initial_guess[2]:.1f}")
print(f"Initial loss: {test_loss:.2f}")
