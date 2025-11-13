import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt


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





result = minimize(
    l1_loss,                   
    initial_guess,              
    args=(t, x_obs, y_obs),     
    bounds=bounds,              
    method='L-BFGS-B',          
    options={
        'maxiter': 1000,        
        'ftol': 1e-6            
    }
)


theta_opt, M_opt, X_opt = result.x
final_loss = result.fun

print(f"Optimization {'successful!' if result.success else 'failed :('}")
print(f"Message: {result.message}")
print(f"\nOptimized Parameters:")
print(f"  θ (theta) = {theta_opt:.3f}°")
print(f"  M         = {M_opt:.6f}")
print(f"  X         = {X_opt:.3f}")
print(f"\nFinal L1 Loss: {final_loss:.2f}")
print(f"Average error per point: {final_loss/(2*N):.4f}")


x_fitted, y_fitted = parametric_curve(t, theta_opt, M_opt, X_opt)


x_rmse = np.sqrt(np.mean((x_obs - x_fitted)**2))
y_rmse = np.sqrt(np.mean((y_obs - y_fitted)**2))
total_rmse = np.sqrt(x_rmse**2 + y_rmse**2)

print(f"\n Quality Metrics:")
print(f"  x RMSE: {x_rmse:.3f}")
print(f"  y RMSE: {y_rmse:.3f}")
print(f"  Combined RMSE: {total_rmse:.3f}")


print("\n visualization...")

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.scatter(x_obs, y_obs, s=10, alpha=0.6, color='blue', label='Observed Data (1501 points)')
plt.plot(x_fitted, y_fitted, 'r-', linewidth=2, label=f'Fitted Curve\nθ={theta_opt:.1f}°, M={M_opt:.4f}, X={X_opt:.1f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Parametric Curve Fit in (x,y) Space')
plt.legend()
plt.grid(True, alpha=0.3)


plt.subplot(2, 2, 2)
plt.scatter(t, x_obs, s=10, alpha=0.6, color='blue', label='Observed x(t)')
plt.plot(t, x_fitted, 'r-', linewidth=2, label='Fitted x(t)')
plt.xlabel('t')
plt.ylabel('x')
plt.title('x-component Fit vs Time t')
plt.legend()
plt.grid(True, alpha=0.3)


plt.subplot(2, 2, 3)
plt.scatter(t, y_obs, s=10, alpha=0.6, color='blue', label='Observed y(t)')
plt.plot(t, y_fitted, 'r-', linewidth=2, label='Fitted y(t)')
plt.xlabel('t')
plt.ylabel('y')
plt.title('y-component Fit vs Time t')
plt.legend()
plt.grid(True, alpha=0.3)


residuals_x = x_obs - x_fitted
residuals_y = y_obs - y_fitted
plt.subplot(2, 2, 4)
plt.plot(t, residuals_x, 'b-', alpha=0.7, label='x residuals')
plt.plot(t, residuals_y, 'r-', alpha=0.7, label='y residuals')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.xlabel('t')
plt.ylabel('Residuals (observed - fitted)')
plt.title('Fit Residuals Over Time')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('curve_fitting_results.png', dpi=300, bbox_inches='tight')
plt.show()


print("\nSaving results for assignment submission...")


results = {
    'theta_degrees': theta_opt,
    'M': M_opt,
    'X': X_opt,
    'final_l1_loss': final_loss,
    'x_rmse': x_rmse,
    'y_rmse': y_rmse,
    'total_rmse': total_rmse,
    'success': result.success,
    'iterations': result.nit if hasattr(result, 'nit') else 'unknown'
}


print("\n" + "="*60)
print("=== FINAL PARAMETRIC EQUATIONS FOR SUBMISSION ===")
print("\nCopy these equations for your LaTeX submission:")
print(r"""
x(t) = t \cos({theta_opt:.3f}^\circ) - e^{{{M_opt:.6f}|t|}} \sin(0.3t) \sin({theta_opt:.3f}^\circ) + {X_opt:.3f}

y(t) = 42 + t \sin({theta_opt:.3f}^\circ) + e^{{{M_opt:.6f}|t|}} \sin(0.3t) \cos({theta_opt:.3f}^\circ)

Where: θ = {theta_opt:.3f}°, M = {M_opt:.6f}, X = {X_opt:.3f}
Final L1 Loss: {final_loss:.2f}
""".format(**results, theta_opt=theta_opt, M_opt=M_opt, X_opt=X_opt, final_loss=final_loss))


with open('final_results_fixed.txt', 'w', encoding='utf-8') as f:
    f.write("Optimized Parameters:\n")
    f.write(f"theta = {theta_opt:.6f} degrees\n")
    f.write(f"M = {M_opt:.12f}\n")
    f.write(f"X = {X_opt:.6f}\n")
    f.write(f"Final L1 Loss = {final_loss:.6f}\n")
    f.write("\nLaTeX Equations:\n")
    f.write(f"x(t) = t * cos({theta_opt:.3f}°) - exp({M_opt:.6f} * |t|) * sin(0.3*t) * sin({theta_opt:.3f}°) + {X_opt:.3f}\n")
    f.write(f"y(t) = 42 + t * sin({theta_opt:.3f}°) + exp({M_opt:.6f} * |t|) * sin(0.3*t) * cos({theta_opt:.3f}°)\n")
