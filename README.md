# FLAM Assignment: Parametric Curve Fitting

## Solution

**Optimized Parameters:**
- θ = 27.534°
- M = 0.018858
- X = 55.076

**Final L1 Loss:** 37807.75

The parametric equations are:

$$
\left(
t \cos(27.534^\circ) - e^{0.018858 |t|} \sin(0.3t) \sin(27.534^\circ) + 55.076,
42 + t \sin(27.534^\circ) + e^{0.018858 |t|} \sin(0.3t) \cos(27.534^\circ)
\right)
$$

## Assignment Submission String

Copy this exact string for submission:

`\left(t*\cos(27.534)-e^{0.018858\left|t\right|}\cdot\sin(0.3t)\sin(27.534)+55.076,42+ t*\sin(27.534)+e^{0.018858\left|t\right|}\cdot\sin(0.3t)\cos(27.534)\right)`

## Desmos Visualization

Paste the submission string above into the Desmos calculator:  
https://www.desmos.com/calculator/rfj91yrxob

## Implementation Overview

The solution is implemented in `main.py`:

1. Data Loading: Loads `xy_data.csv` (1500 points) and generates t values (6.1 to 59.9)  
2. Parametric Curve: Implements the mathematical model with θ, M, X parameters  
3. L1 Optimization: Uses scipy.optimize.minimize for L1 norm minimization  
4. Visualization: Creates 4-panel plots showing fit quality  
5. Results: Generates submission-ready LaTeX equations  

## Files

- `main.py` - Complete implementation and analysis  
- `xy_data.csv` - Original dataset  
- `final_results_fixed.txt` - Raw optimization output  
- `curve_fitting_results.png` - Visualization of results  

## Running the Solution

