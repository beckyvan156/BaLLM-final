# -*- coding: utf-8 -*-
"""
Created on Thu May 29 12:41:41 2025

@author: becky
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# === Step 1: Define simulation parameters ===
n_samples = 5000  


true_beta = {
    'intercept': -0.2,
    'X': 5,
    'Y': -4,
    'Z': 7,
    'W': 3
}



# Marginal probabilities for binary predictors
p_X = 0.3
p_Y = 0.5
p_Z = 0.4
p_W = 0.6

# === Step 2: Simulate predictor values ===
np.random.seed(42)  # For reproducibility
X = np.random.binomial(1, p_X, n_samples)
Y = np.random.binomial(1, p_Y, n_samples)
Z = np.random.binomial(1, p_Z, n_samples)
W = np.random.binomial(1, p_W, n_samples)

# === Step 3: Compute probabilities and simulate A ===
log_odds = (true_beta['intercept']
            + true_beta['X'] * X
            + true_beta['Y'] * Y
            + true_beta['Z'] * Z
            + true_beta['W'] * W)
p_A = expit(log_odds)  # Convert log-odds to probability
A = np.random.binomial(1, p_A)

# === Step 4: Create DataFrame ===
df = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z, 'W': W, 'A': A})

# === Step 5: Fit logistic regression ===
X_design = sm.add_constant(df[['X', 'Y', 'Z', 'W']])
model = sm.Logit(df['A'], X_design).fit(disp=0)  # Set disp=0 to suppress fitting output

# === Step 6: Display results ===
print("True Coefficients:")
for name in ['intercept', 'X', 'Y', 'Z', 'W']:
    print(f"  {name:>9}: {true_beta[name]:.3f}")

print("\nEstimated Coefficients:")
print(model.params)


print("\nSummary:")
print(model.summary())




# === Predict probabilities and classify ===
pred_probs = model.predict(X_design)                   # P(A=1 | X,Y,Z,W)
pred_labels = (pred_probs >= 0.5).astype(int)          

# === Compute accuracy ===
accuracy = accuracy_score(df['A'], pred_labels)
conf_matrix = confusion_matrix(df['A'], pred_labels)

# === Output results ===
print(f"Prediction Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)


# === Compute ROC and AUC ===
fpr, tpr, thresholds = roc_curve(df['A'], pred_probs)
auc = roc_auc_score(df['A'], pred_probs)

# === Plot ROC Curve ===
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Logistic Regression")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

####################################################


import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# === Simulation Parameters ===
n_reps = 500
n_samples = 5000

true_beta = {
    'intercept': -0.2,
    'X': 5,
    'Y': -4,
    'Z': 7,
    'W': 3
}

# Marginal probabilities for binary predictors
p_X = 0.3
p_Y = 0.5
p_Z = 0.4
p_W = 0.6

# Parameter names in order (for display)
param_names = ['β₀ (intercept)', 'β₁', 'β₂', 'β₃', 'β₄']
param_keys = ['intercept', 'X', 'Y', 'Z', 'W']  # Keys for true_beta dictionary

# Storage for results
results = {
    'beta_hat': [],      # Estimated coefficients
    'se_hat': [],        # Standard errors
    'auc': []            # AUC values
}

# === Run Simulation ===
np.random.seed(42)

print(f"Running {n_reps} simulations...")
print("-" * 60)

for rep in range(n_reps):
    if (rep + 1) % 50 == 0:
        print(f"Completed {rep + 1}/{n_reps} repetitions...")
    
    # Simulate predictor values
    X = np.random.binomial(1, p_X, n_samples)
    Y = np.random.binomial(1, p_Y, n_samples)
    Z = np.random.binomial(1, p_Z, n_samples)
    W = np.random.binomial(1, p_W, n_samples)
    
    # Compute probabilities and simulate outcome
    log_odds = (true_beta['intercept'] +
                true_beta['X'] * X +
                true_beta['Y'] * Y +
                true_beta['Z'] * Z +
                true_beta['W'] * W)
    p_A = expit(log_odds)
    A = np.random.binomial(1, p_A)
    
    # Create design matrix
    df = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z, 'W': W, 'A': A})
    X_design = sm.add_constant(df[['X', 'Y', 'Z', 'W']])
    
    # Fit logistic regression
    try:
        model = sm.Logit(df['A'], X_design).fit(disp=0)
        
        # Store estimated coefficients and standard errors
        results['beta_hat'].append(model.params.values)
        results['se_hat'].append(model.bse.values)
        
        # Calculate AUC
        pred_probs = model.predict(X_design)
        auc = roc_auc_score(df['A'], pred_probs)
        results['auc'].append(auc)
        
    except:
        # In case of convergence issues
        results['beta_hat'].append([np.nan] * 5)
        results['se_hat'].append([np.nan] * 5)
        results['auc'].append(np.nan)

print(f"Simulation complete!")
print("=" * 60)

# === Convert to arrays ===
beta_hat_array = np.array(results['beta_hat'])
se_hat_array = np.array(results['se_hat'])
auc_array = np.array(results['auc'])

# Remove any failed simulations
valid_idx = ~np.isnan(beta_hat_array[:, 0])
beta_hat_array = beta_hat_array[valid_idx]
se_hat_array = se_hat_array[valid_idx]
auc_array = auc_array[valid_idx]

print(f"\nValid simulations: {len(beta_hat_array)}/{n_reps}")
print("=" * 60)

# === Calculate Performance Metrics ===
true_beta_array = np.array([true_beta[name] for name in param_keys])

# 1. Bias: E[β̂] - β
bias = np.mean(beta_hat_array, axis=0) - true_beta_array

# 2. Empirical SE: SD(β̂)
empirical_se = np.std(beta_hat_array, axis=0, ddof=1)

# 3. Mean of estimated SEs
mean_estimated_se = np.mean(se_hat_array, axis=0)

# 4. Coverage: Proportion where β ∈ [β̂ ± 1.96*SE]
coverage = np.zeros(len(param_names))
for i in range(len(param_names)):
    lower = beta_hat_array[:, i] - 1.96 * se_hat_array[:, i]
    upper = beta_hat_array[:, i] + 1.96 * se_hat_array[:, i]
    coverage[i] = np.mean((true_beta_array[i] >= lower) & 
                          (true_beta_array[i] <= upper))

# === Display Results ===
print("\n" + "=" * 80)
print("SIMULATION RESULTS SUMMARY")
print("=" * 80)

results_df = pd.DataFrame({
    'Parameter': param_names,
    'True β': true_beta_array,
    'Mean β̂': np.mean(beta_hat_array, axis=0),
    'Bias': bias,
    'Empirical SE(β̂)': empirical_se,
    'Mean Est. SE': mean_estimated_se,
    'Coverage (95%)': coverage
})

print("\n", results_df.to_string(index=False))

print("\n" + "-" * 80)
print(f"Mean AUC across simulations: {np.mean(auc_array):.4f}")
print(f"SD of AUC: {np.std(auc_array, ddof=1):.4f}")
print(f"Min AUC: {np.min(auc_array):.4f}")
print(f"Max AUC: {np.max(auc_array):.4f}")
print("=" * 80)

# === Visualization ===
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(f'Monte Carlo Simulation Results (n={n_reps} repetitions, N={n_samples} samples/rep)', 
             fontsize=16, fontweight='bold', y=0.995)
axes = axes.flatten()

for i, param in enumerate(param_names):
    ax = axes[i]
    
    # Histogram of estimated coefficients
    ax.hist(beta_hat_array[:, i], bins=30, alpha=0.7, 
            color='steelblue', edgecolor='black')
    ax.axvline(true_beta_array[i], color='red', linestyle='--', 
               linewidth=2, label=f'True β = {true_beta_array[i]:.2f}')
    ax.axvline(np.mean(beta_hat_array[:, i]), color='green', 
               linestyle='-', linewidth=2, 
               label=f'Mean β̂ = {np.mean(beta_hat_array[:, i]):.2f}')
    
    ax.set_xlabel(f'{param}')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{param}\nBias = {bias[i]:.4f}, Coverage = {coverage[i]:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

# AUC distribution
ax = axes[5]
ax.hist(auc_array, bins=30, alpha=0.7, color='coral', edgecolor='black')
ax.axvline(np.mean(auc_array), color='darkred', linestyle='-', 
           linewidth=2, label=f'Mean = {np.mean(auc_array):.4f}')
ax.set_xlabel('AUC')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of AUC')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = r'C:\Users\becky\OneDrive - University of Texas Southwestern\LLM\manuscript\figures\simulation_results.pdf'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"\nPlot saved to: {save_path}")

# === Additional Analysis: Coverage by Parameter ===
print("\n" + "=" * 80)
print("COVERAGE ANALYSIS")
print("=" * 80)
print("\nInterpretation:")
print("- Bias close to 0 indicates unbiased estimation")
print("- Empirical SE measures variability of estimates across simulations")
print("- Mean Est. SE should be close to Empirical SE (model SE estimates are accurate)")
print("- Coverage should be close to 0.95 for 95% confidence intervals")
print("=" * 80)