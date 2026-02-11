# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 15:47:40 2025

@author: becky
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import pymc as pm
import pytensor.tensor as at
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import pickle
import os
import networkx as nx
from sklearn.metrics import roc_curve, auc

### input dataset ###
final_kg=pd.read_csv('C:/Users/becky/OneDrive/Desktop/2023 summer intern/literature/CARD/results/final_kg.csv')
antibiogram_gene=pd.read_csv('C:/Users/becky/OneDrive/Desktop/2023 summer intern/Data/purified_data/antibiogram_VAMP.csv')
antibiogram_pheno=pd.read_csv('C:/Users/becky/OneDrive/Desktop/2023 summer intern/Data/purified_data/antibiogram_phenotype.csv')

###  format column   ###
antibiogram_pheno['phenotype'] = antibiogram_pheno['phenotype'].map({'resistant': 1, 'susceptible': 0})
antibiogram_gene['variants'] = antibiogram_gene['variants'].str.split('.').str[0]

### Merge the dataframes on `sample_id` ###
merged_df = pd.merge(antibiogram_pheno, antibiogram_gene, on='sample_id', how='left')

###  Split by antibiotics ###
antibiogram_dict = {}

for abx in merged_df['antibiotics'].unique():
    key = f'antibiogram_{abx.lower()}'
    antibiogram_dict[key] = merged_df[merged_df['antibiotics'] == abx].copy()

# Example: antibiogram_dict['antibiogram_ertapenem']


# Dictionary to hold wide-format bayes dataframes
bayes_df_dict = {}

# Loop through each antibiotic in final_kg
for abx in final_kg['antibiotic'].unique():
    # Subset final_kg for this antibiotic
    abx_genes = final_kg[final_kg['antibiotic'] == abx]['gene'].unique()
    
    # Subset the long dataframe for this antibiotic
    abx_long_df = antibiogram_dict[f'antibiogram_{abx.lower()}']

    # Get unique sample_id, bacteria, and phenotype (first occurrence per sample)
    sample_info = (
        abx_long_df
        .drop_duplicates(subset=['sample_id'])[['sample_id', 'bacteria', 'phenotype']]
        .reset_index(drop=True)
    )

    # Create the base bayes dataframe
    abx_bayes = sample_info.copy()

    # Add gene columns, initialized to 0
    for gene in abx_genes:
        abx_bayes[gene] = 0

    # Fill gene values based on presence in variants
    for i, row in abx_bayes.iterrows():
        sample_id = row['sample_id']
        
        # Get all variants for this sample_id
        sample_variants = abx_long_df[abx_long_df['sample_id'] == sample_id]['variants'].unique()
        
        for gene in abx_genes:
            if gene in sample_variants:
                abx_bayes.at[i, gene] = 1

    # Save to dictionary
    bayes_df_dict[f'{abx.lower()}_bayes'] = abx_bayes





##########################     Bayesian modeling    #############################

# Step 1: Extract amikacin data
df = bayes_df_dict['trimethoprim-sulfamethoxazole_bayes'].copy()

# Step 2: Identify genes and prepare X and y
non_gene_cols = ['sample_id', 'bacteria', 'phenotype']
gene_cols = [col for col in df.columns if col not in non_gene_cols]

X = df[gene_cols].astype(float)
y = df['phenotype'].astype(int)

### check gene variability across samples in X_train ###
constant_columns = [col for col in X.columns if X[col].nunique() == 1]
print("Constant columns:", constant_columns)
# Drop them
X = X.drop(columns=constant_columns)


# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 4: Build mu_dict from final_kg
# Filter to antibiotic = "amikacin"
final_kg_ami = final_kg[final_kg['antibiotic'] == 'trimethoprim-sulfamethoxazole']
mu_dict = dict(zip(final_kg_ami['gene'], final_kg_ami['Score']))

# Only keep mu for genes in the training set
valid_genes = [g for g in gene_cols if g in mu_dict]

# Get the intersection of valid_genes and X_train columns
valid_genes_filtered = [g for g in valid_genes if g in X_train.columns]

# Subset X_train using the filtered gene list
X_train_valid = X_train[valid_genes_filtered]

# Update X and indices to valid genes
X_test_valid = X_test[valid_genes_filtered]

# update mu 
mu = np.array([mu_dict[g] for g in valid_genes_filtered if g in mu_dict])
group_idx = pd.factorize(mu)[0]  # Handles ties
unique_mu = np.unique(mu)




# # Step 5: PyMC model
# tau = 1.0
# delta = 1.0  # loosen delta

# with pm.Model() as model:
#     beta0 = pm.Normal("beta0", mu=0, sigma=5)

#     # Step 1: Sample unordered values
#     theta_group_unordered = pm.Normal("theta_group_unordered", mu=0, sigma=1, shape=len(unique_mu))

#     # Step 2: Apply sort as deterministic
#     ordered_theta = pm.Deterministic("ordered_theta", at.sort(theta_group_unordered))

#     # Step 3: Noise + hierarchical indexing
#     eps = pm.Normal("eps", mu=0, sigma=delta, shape=len(valid_genes_filtered))
#     beta = pm.Deterministic("beta", mu + tau * (ordered_theta[group_idx] + eps))

#     logits = beta0 + pm.math.dot(X_train_valid.values, beta)
#     A_obs = pm.Bernoulli("A_obs", logit_p=logits, observed=y_train.values)

#     trace = pm.sample(1000, tune=1000, target_accept=0.95, return_inferencedata=True)


# # Step 6: Posterior Prediction
# posterior_beta = trace.posterior["beta"].stack(draws=("chain", "draw")).values
# posterior_beta0 = trace.posterior["beta0"].stack(draws=("chain", "draw")).values

# logits_test = posterior_beta0 + X_test_valid.values @ posterior_beta
# probs_test = 1 / (1 + np.exp(-logits_test))
# P_test_mean = probs_test.mean(axis=1)

# y_pred = (P_test_mean > 0.5).astype(int)

# print("AUC:", roc_auc_score(y_test, P_test_mean))
# print("Accuracy:", accuracy_score(y_test, y_pred))






### just fit a logistic regression using train data ###
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_valid, y_train)
y_pred_test = clf.predict(X_test_valid)
accuracy = accuracy_score(y_test, y_pred_test)
print("Test Accuracy:", accuracy)
coef = clf.coef_[0]       
intercept = clf.intercept_[0]


### bayesian modeling tuning ###

# 1. no constraint at all for beta #
with pm.Model() as model:
    beta0 = pm.Normal("beta0", mu=0, sigma=5)
    
    # Set each beta_j ~ N(0, 1)
    beta = pm.Normal("beta", mu=0, sigma=1, shape=X_train_valid.shape[1])

    # Logistic regression
    logits = beta0 + pm.math.dot(X_train_valid.values, beta)
    A_obs = pm.Bernoulli("A_obs", logit_p=logits, observed=y_train.values)

    # Sample
    trace = pm.sample(1000, tune=1000, target_accept=0.95, return_inferencedata=True)


posterior_beta = trace.posterior["beta"].stack(draws=("chain", "draw")).values
posterior_beta0 = trace.posterior["beta0"].stack(draws=("chain", "draw")).values

logits_test = posterior_beta0 + X_test_valid.values @ posterior_beta
probs_test = 1 / (1 + np.exp(-logits_test))
P_test_mean = probs_test.mean(axis=1)

y_pred = (P_test_mean > 0.5).astype(int)

print("AUC:", roc_auc_score(y_test, P_test_mean))
print("Accuracy:", accuracy_score(y_test, y_pred))



# 2. beta~N(mu,1)
with pm.Model() as model:
    beta0 = pm.Normal("beta0", mu=0, sigma=5)
    
    # Set each beta_j ~ N(mu, 1)
    beta = pm.Normal("beta", mu=mu, sigma=1, shape=X_train_valid.shape[1])

    # Logistic regression
    logits = beta0 + pm.math.dot(X_train_valid.values, beta)
    A_obs = pm.Bernoulli("A_obs", logit_p=logits, observed=y_train.values)

    # Sample
    trace = pm.sample(1000, tune=1000, target_accept=0.95, return_inferencedata=True)


posterior_beta = trace.posterior["beta"].stack(draws=("chain", "draw")).values
posterior_beta0 = trace.posterior["beta0"].stack(draws=("chain", "draw")).values

logits_test = posterior_beta0 + X_test_valid.values @ posterior_beta
probs_test = 1 / (1 + np.exp(-logits_test))
P_test_mean = probs_test.mean(axis=1)

y_pred = (P_test_mean > 0.5).astype(int)

print("AUC:", roc_auc_score(y_test, P_test_mean))
print("Accuracy:", accuracy_score(y_test, y_pred))



# 3. try other samplers with beta~N(mu,1)
with pm.Model() as model:
    # Intercept
    beta0 = pm.Normal("beta0", mu=0, sigma=5)

    # Coefficients: beta_j ~ N(mu_j, 1)
    beta = pm.Normal("beta", mu=mu, sigma=1, shape=X_train_valid.shape[1])

    # Logistic regression
    logits = beta0 + pm.math.dot(X_train_valid.values, beta)
    A_obs = pm.Bernoulli("A_obs", logit_p=logits, observed=y_train.values)

    # Use Metropolis sampler
    step = pm.Metropolis()
    trace = pm.sample(1000, tune=1000, step=step, return_inferencedata=True)

# Posterior prediction
posterior_beta = trace.posterior["beta"].stack(draws=("chain", "draw")).values
posterior_beta0 = trace.posterior["beta0"].stack(draws=("chain", "draw")).values

# Predict on test set
logits_test = posterior_beta0 + X_test_valid.values @ posterior_beta
probs_test = 1 / (1 + np.exp(-logits_test))
P_test_mean = probs_test.mean(axis=1)
y_pred = (P_test_mean > 0.5).astype(int)

print("AUC:", roc_auc_score(y_test, P_test_mean))
print("Accuracy:", accuracy_score(y_test, y_pred))



# 4. try other samplers with PyMC model
tau = 1.0
delta = 1.0 

with pm.Model() as model:
    beta0 = pm.Normal("beta0", mu=0, sigma=5)

    # Step 1: Sample unordered values
    theta_group_unordered = pm.Normal("theta_group_unordered", mu=0, sigma=1, shape=len(unique_mu))

    # Step 2: Apply sort as deterministic
    ordered_theta = pm.Deterministic("ordered_theta", at.sort(theta_group_unordered))

    # Step 3: Noise + hierarchical indexing
    eps = pm.Normal("eps", mu=0, sigma=delta, shape=len(valid_genes_filtered))
    beta = pm.Deterministic("beta", mu + tau * (ordered_theta[group_idx] + eps))

    logits = beta0 + pm.math.dot(X_train_valid.values, beta)
    A_obs = pm.Bernoulli("A_obs", logit_p=logits, observed=y_train.values)

    # Use Metropolis sampler
    step = pm.Metropolis()
    trace = pm.sample(1000, tune=1000, step=step, return_inferencedata=True)


# Posterior prediction
posterior_beta = trace.posterior["beta"].stack(draws=("chain", "draw")).values
posterior_beta0 = trace.posterior["beta0"].stack(draws=("chain", "draw")).values

# Predict on test set
logits_test = posterior_beta0 + X_test_valid.values @ posterior_beta
probs_test = 1 / (1 + np.exp(-logits_test))
P_test_mean = probs_test.mean(axis=1)
y_pred = (P_test_mean > 0.5).astype(int)

print("AUC:", roc_auc_score(y_test, P_test_mean))
print("Accuracy:", accuracy_score(y_test, y_pred))




# Plot beta from pymc
posterior_beta = np.array(posterior_beta)  # Ensure it's an ndarray
mu_flat = mu.flatten()

# Create a long DataFrame for plotting
n_betas, n_samples = posterior_beta.shape
df = pd.DataFrame({
    'beta_idx': np.repeat(np.arange(n_betas), n_samples),
    'sampled_beta': posterior_beta.flatten(),
    'true_mu': np.repeat(mu_flat, n_samples)
})

#Violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x='true_mu', y='sampled_beta', data=df, inner='quartile', palette='Set2')
plt.axhline(0, color='gray', linestyle='--')
plt.title('trimethoprim-sulfamethoxazole:posterior beta distributions grouped by true value')
plt.xlabel('True β Value (mu)')
plt.ylabel('Sampled β')
plt.grid(True)
plt.show()



######################   wrap up MH code as a function and save model parameters #############################

# Step 1: Extract data
df = bayes_df_dict['trimethoprim-sulfamethoxazole_bayes'].copy()

# Step 2: Identify genes and prepare X and y
non_gene_cols = ['sample_id', 'bacteria', 'phenotype']
gene_cols = [col for col in df.columns if col not in non_gene_cols]

X = df[gene_cols].astype(float)
y = df['phenotype'].astype(int)

### check gene variability across samples in X_train ###
constant_columns = [col for col in X.columns if X[col].nunique() == 1]
# Drop them
X = X.drop(columns=constant_columns)


# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 4: Build mu_dict from final_kg
# Filter to antibiotic = "amikacin"
final_kg_ami = final_kg[final_kg['antibiotic'] == 'trimethoprim-sulfamethoxazole']
mu_dict = dict(zip(final_kg_ami['gene'], final_kg_ami['Score']))

# Only keep mu for genes in the training set
valid_genes = [g for g in gene_cols if g in mu_dict]

# Get the intersection of valid_genes and X_train columns
valid_genes_filtered = [g for g in valid_genes if g in X_train.columns]

# Subset X_train using the filtered gene list
X_train_valid = X_train[valid_genes_filtered]

# Update X and indices to valid genes
X_test_valid = X_test[valid_genes_filtered]

# update mu 
mu = np.array([mu_dict[g] for g in valid_genes_filtered if g in mu_dict])
group_idx = pd.factorize(mu)[0]  # Handles ties
unique_mu = np.unique(mu)

mu_flat = mu.flatten()
gene_strength = pd.DataFrame({            
    'gene': valid_genes_filtered,
    'mu': mu_flat
})           

gene_strength.to_csv("trimethoprim-sulfamethoxazole_gene_strength.csv", index=False)    # save gene_strength

# # Load the parameter CSV
# gene_strength = pd.read_csv("trimethoprim-sulfamethoxazole_gene_strength.csv")


###  logistic regression  ###
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_valid, y_train)
coef = clf.coef_[0]       
intercept = clf.intercept_[0]

with open("trimethoprim-sulfamethoxazole_logistic_model.pkl", "wb") as f:           # save logistic model
    pickle.dump(clf, f)


# # load the logistic model
# with open("trimethoprim-sulfamethoxazole_logistic_model.pkl", "rb") as f:
#     clf_loaded = pickle.load(f)
    
# y_pred_test = clf.predict(X_test_valid)
# accuracy = accuracy_score(y_test, y_pred_test)
# print("Test Accuracy:", accuracy)




# RWMH samplers with PyMC model
tau = 1.0
delta = 1.0 

with pm.Model() as model:
    beta0 = pm.Normal("beta0", mu=0, sigma=5)

    # Step 1: Sample unordered values
    theta_group_unordered = pm.Normal("theta_group_unordered", mu=0, sigma=1, shape=len(unique_mu))

    # Step 2: Apply sort as deterministic
    ordered_theta = pm.Deterministic("ordered_theta", at.sort(theta_group_unordered))

    # Step 3: Noise + hierarchical indexing
    eps = pm.Normal("eps", mu=0, sigma=delta, shape=len(valid_genes_filtered))
    beta = pm.Deterministic("beta", mu + tau * (ordered_theta[group_idx] + eps))

    logits = beta0 + pm.math.dot(X_train_valid.values, beta)
    A_obs = pm.Bernoulli("A_obs", logit_p=logits, observed=y_train.values)

    # Use Metropolis sampler
    step = pm.Metropolis()
    trace = pm.sample(1000, tune=1000, step=step, return_inferencedata=True)


# Posterior prediction
posterior_beta = trace.posterior["beta"].stack(draws=("chain", "draw")).values
posterior_beta0 = trace.posterior["beta0"].stack(draws=("chain", "draw")).values

# Save both arrays into one .npz file
np.savez("trimethoprim-sulfamethoxazole_posterior_params.npz", posterior_beta=posterior_beta, posterior_beta0=posterior_beta0)


# # load parameters
# data = np.load("trimethoprim-sulfamethoxazole_posterior_params.npz")
# posterior_beta = data["posterior_beta"]
# posterior_beta0 = data["posterior_beta0"]


# # Predict on test set
# logits_test = posterior_beta0 + X_test_valid.values @ posterior_beta
# probs_test = 1 / (1 + np.exp(-logits_test))
# P_test_mean = probs_test.mean(axis=1)
# y_pred = (P_test_mean > 0.5).astype(int)
# print("AUC:", roc_auc_score(y_test, P_test_mean))
# print("Accuracy:", accuracy_score(y_test, y_pred))




def run_model_for_all_antibiotics(
    bayes_df_dict, final_kg, output_dir
):
    os.makedirs(output_dir, exist_ok=True)

    for key in bayes_df_dict.keys():
        if not key.endswith("_bayes"):
            continue  # skip keys that are not antibiotic_bayes format

        abx = key.replace("_bayes", "")
        print(f"Processing antibiotic: {abx}")

        # Step 1: Extract data
        df = bayes_df_dict[key].copy()

        # Step 2: Identify genes and prepare X and y
        non_gene_cols = ['sample_id', 'bacteria', 'phenotype']
        gene_cols = [col for col in df.columns if col not in non_gene_cols]

        X = df[gene_cols].astype(float)
        y = df['phenotype'].astype(int)

        # Remove constant columns
        constant_columns = [col for col in X.columns if X[col].nunique() == 1]
        X = X.drop(columns=constant_columns)

        # Step 3: Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Step 4: Build mu_dict from final_kg
        final_kg_filtered = final_kg[final_kg['antibiotic'] == abx]
        mu_dict = dict(zip(final_kg_filtered['gene'], final_kg_filtered['Score']))
        valid_genes = [g for g in gene_cols if g in mu_dict]
        valid_genes_filtered = [g for g in valid_genes if g in X_train.columns]

        X_train_valid = X_train[valid_genes_filtered]
        X_test_valid = X_test[valid_genes_filtered]

        mu = np.array([mu_dict[g] for g in valid_genes_filtered])
        group_idx = pd.factorize(mu)[0]
        unique_mu = np.unique(mu)

        # Save gene strength
        gene_strength = pd.DataFrame({'gene': valid_genes_filtered, 'mu': mu})
        # gene_strength.to_csv(os.path.join(output_dir, f"{abx}_gene_strength.csv"), index=False)

        # # Logistic regression
        # clf = LogisticRegression(max_iter=1000)
        # clf.fit(X_train_valid, y_train)

        # with open(os.path.join(output_dir, f"{abx}_logistic_model.pkl"), "wb") as f:
        #     pickle.dump(clf, f)

        # Bayesian model
        tau = 1.0
        # delta = 1.0
        delta = 2.5

        with pm.Model() as model:
            beta0 = pm.Normal("beta0", mu=0, sigma=5)
            theta_group_unordered = pm.Normal("theta_group_unordered", mu=0, sigma=1, shape=len(unique_mu))
            ordered_theta = pm.Deterministic("ordered_theta", at.sort(theta_group_unordered))
            eps = pm.Normal("eps", mu=0, sigma=delta, shape=len(valid_genes_filtered))
            beta = pm.Deterministic("beta", mu + tau * (ordered_theta[group_idx] + eps))
            logits = beta0 + pm.math.dot(X_train_valid.values, beta)
            A_obs = pm.Bernoulli("A_obs", logit_p=logits, observed=y_train.values)

            step = pm.Metropolis()
            trace = pm.sample(1000, tune=1000, step=step, return_inferencedata=True, progressbar=False)

        posterior_beta = trace.posterior["beta"].stack(draws=("chain", "draw")).values
        posterior_beta0 = trace.posterior["beta0"].stack(draws=("chain", "draw")).values
        
        # Predict on test set
        logits_test = posterior_beta0 + X_test_valid.values @ posterior_beta
        probs_test = 1 / (1 + np.exp(-logits_test))
        P_test_mean = probs_test.mean(axis=1)
        y_pred = (P_test_mean > 0.5).astype(int)
        print("Accuracy:", accuracy_score(y_test, y_pred))


        np.savez(
            # os.path.join(output_dir, f"{abx}_posterior_params.npz"),
            os.path.join(output_dir, f"{abx}_posterior_params_2.5.npz"),

            posterior_beta=posterior_beta,
            posterior_beta0=posterior_beta0
        )

    print("Processing complete for all antibiotics.")


run_model_for_all_antibiotics(
    bayes_df_dict,
    final_kg,
    output_dir=r"C:\Users\becky\OneDrive\Desktop\2023 summer intern\literature\CARD\results\KG model parameters"
)




### check parameters for some antibiotics ###
# List of antibiotics of interest
antibiotics = ["amikacin","ciprofloxacin", "imipenem", "levofloxacin"]

# Directory where your models are saved
model_dir = r"C:\Users\becky\OneDrive\Desktop\2023 summer intern\literature\CARD\results\KG model parameters"

# Output dictionaries
logistic_param_df_dict = {}
bayesian_param_df_dict = {}

for abx in antibiotics:
    try:
        # Load gene names from gene_strength file
        gene_strength_path = os.path.join(model_dir, f"{abx}_gene_strength.csv")
        gene_strength_df = pd.read_csv(gene_strength_path)
        gene_names = gene_strength_df["gene"].tolist()

        # --- Logistic Model ---
        with open(os.path.join(model_dir, f"{abx}_logistic_model.pkl"), "rb") as f:
            clf = pickle.load(f)

        coef = clf.coef_.flatten()
        intercept = clf.intercept_[0]

        logistic_df = pd.DataFrame({
            "gene": gene_names,
            "logistic_coef": coef
        })
        logistic_df["intercept"] = intercept  # optional: repeated per row

        logistic_param_df_dict[abx] = logistic_df

        # --- Bayesian Model ---
        bayes = np.load(os.path.join(model_dir, f"{abx}_posterior_params.npz"))
        posterior_beta = bayes["posterior_beta"]  # shape: (n_samples, n_genes)

        bayes_beta_df = pd.DataFrame(posterior_beta)
        bayesian_param_df_dict[abx] = bayes_beta_df

        print(f"✅ Loaded: {abx} — Logistic + Bayesian β samples ({posterior_beta.shape[0]} genes)")

    except FileNotFoundError as e:
        print(f"❌ Missing model or data for {abx}: {e}")






logistic_df = logistic_param_df_dict["imipenem"]
gene_names = logistic_df["gene"].values

bayes_df = bayesian_param_df_dict["imipenem"]
bayes_df.index = gene_names


bayes_long_df = bayes_df.reset_index().melt(
    id_vars="index", var_name="sample", value_name="posterior_beta"
)
bayes_long_df = bayes_long_df.rename(columns={"index": "gene"})

#  Map logistic regression beta to each gene
logistic_coef_map = dict(zip(logistic_df["gene"], logistic_df["logistic_coef"]))
bayes_long_df["logistic_coef"] = bayes_long_df["gene"].map(logistic_coef_map)

gene_order = logistic_df["gene"].tolist()

# Apply categorical order to both DataFrames
bayes_long_df["gene"] = pd.Categorical(bayes_long_df["gene"], categories=gene_order, ordered=True)
logistic_df["gene"] = pd.Categorical(logistic_df["gene"], categories=gene_order, ordered=True)

gene_strength_df = pd.read_csv('C:\\Users\\becky\\OneDrive\\Desktop\\2023 summer intern\\literature\\CARD\\results\\KG model parameters\\imipenem_gene_strength.csv')



# Plot
plt.figure(figsize=(18, 6))

# Boxplot (Bayesian posterior samples)
sns.boxplot(
    data=bayes_long_df,
    x="gene",
    y="posterior_beta",
    color='lightblue',
    showfliers=False,
    zorder=1  # lower z-order (background)
)

# Scatterplot (Logistic regression β)
sns.scatterplot(
    data=logistic_df,
    x="gene",
    y="logistic_coef",
    color='red',
    s=40,
    label="Logistic β",
    zorder=2  # higher z-order (foreground)
)

plt.xticks(rotation=90)
plt.xlabel("Gene")
plt.ylabel("Beta Coefficient")
plt.title("Imipenem: Bayesian Posterior β vs Logistic Regression β")
plt.legend()
plt.tight_layout()
plt.show()


### doripenem posterior distribution boxplot ###
file_path = r"C:\Users\becky\OneDrive\Desktop\2023 summer intern\literature\CARD\results\KG model parameters\doripenem_posterior_params_2.5.npz"
bayes = np.load(file_path)
posterior_beta = bayes["posterior_beta"]
posterior_beta0 = bayes["posterior_beta0"]

file_path = r"C:\Users\becky\OneDrive\Desktop\2023 summer intern\literature\CARD\results\KG model parameters\doripenem_gene_strength.csv"
gene_strength = pd.read_csv(file_path)
ordered_genes = gene_strength['gene'].tolist()

logistic_path = r"C:\Users\becky\OneDrive\Desktop\2023 summer intern\literature\CARD\results\KG model parameters\doripenem_logistic_model.pkl"
with open(logistic_path, "rb") as f:
    logistic_model = pickle.load(f)

logistic_beta = logistic_model.coef_[0]  # Shape should match ordered_genes




def load_gene_name_mapping(kegg_file_path):
    """
    Load KEGG orthology file and create mapping from K00001 format to gene names
    """
    gene_mapping = {}
    
    with open(kegg_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('K'):
                # Split by tab or spaces to get K number and description
                parts = line.split('\t' if '\t' in line else ' ', 1)
                if len(parts) >= 2:
                    k_number = parts[0].strip()
                    description = parts[1].strip()
                    
                    # Extract gene name before the first semicolon
                    if ';' in description:
                        gene_name = description.split(';')[0].strip()
                    else:
                        gene_name = description.strip()
                    
                    gene_mapping[k_number] = gene_name
    
    print(f"Loaded {len(gene_mapping)} gene name mappings")
    return gene_mapping


def create_gene_strength_boxplot(posterior_beta, ordered_genes, logistic_beta, kegg_file_path=None):
    """
    Create boxplot of gene strength ordered by median values with logistic regression coefficients
    
    Parameters:
    posterior_beta: numpy array of shape [n_genes, n_iterations]
    ordered_genes: list of gene names corresponding to rows
    logistic_beta: numpy array of logistic regression coefficients
    kegg_file_path: path to KEGG orthology file for gene name mapping
    """
    
    # Load gene name mapping if provided
    if kegg_file_path:
        gene_mapping = load_gene_name_mapping(kegg_file_path)
        # Convert K numbers to readable names
        display_genes = []
        for gene in ordered_genes:
            if gene in gene_mapping:
                display_name = gene_mapping[gene]
                # Truncate very long names
                if len(display_name) > 50:
                    display_name = display_name[:47] + "..."
                display_genes.append(display_name)
            else:
                display_genes.append(gene)  # Keep original if not found
        print(f"Mapped {sum(1 for i, gene in enumerate(ordered_genes) if gene in gene_mapping)} out of {len(ordered_genes)} genes")
    else:
        display_genes = ordered_genes
    
    # Calculate median for each gene (each row)
    medians = np.median(posterior_beta, axis=1)
    
    # Create a DataFrame for easier handling
    gene_data = []
    for i, (gene, display_name) in enumerate(zip(ordered_genes, display_genes)):
        median_val = medians[i]
        for value in posterior_beta[i, :]:
            gene_data.append({
                'gene': gene, 
                'display_name': display_name,
                'value': value, 
                'median': median_val
            })
    
    df = pd.DataFrame(gene_data)
    
    # Sort genes by median value
    gene_order_df = df.groupby(['gene', 'display_name'])['median'].first().reset_index()
    gene_order_df = gene_order_df.sort_values('median')
    display_name_order = gene_order_df['display_name'].tolist()
    
    # Create logistic beta mapping for ordered display names
    logistic_values = {}
    for i, (gene, display_name) in enumerate(zip(ordered_genes, display_genes)):
        logistic_values[display_name] = logistic_beta[i]
    
    # Create the plot
    plt.figure(figsize=(20, 10))
    
    # Create boxplot with ordered genes using display names
    sns.boxplot(data=df, x='display_name', y='value', order=display_name_order)
    
    # Add yellow dots for logistic regression coefficients
    logistic_ordered = [logistic_values[name] for name in display_name_order]
    x_positions = np.arange(len(display_name_order))
    plt.scatter(x_positions, logistic_ordered, color='yellow', s=50, zorder=5, 
                label='Logistic Regression Estimates', edgecolors='black', linewidths=0.5)
    
    # Customize the plot
    plt.title('Doripenem: Posterior Gene Strength Distribution from BaLLM Ordered by Median', fontsize=20, fontweight='bold')
    plt.xlabel('Gene', fontsize=18)
    plt.ylabel('Posterior Beta Value', fontsize=18)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90, ha='right', fontsize=14)
    
    # Add a horizontal line at y=0 for reference
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    # Add legend
    plt.legend(loc='upper left', fontsize=18)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    # Print some summary statistics using display names
    print("Posterior Gene strength summary (ordered by median):")
    print("-" * 50)
    summary_stats = df.groupby(['gene', 'display_name'])['value'].agg(['median', 'mean', 'std']).round(4)
    summary_stats = summary_stats.reset_index().set_index('display_name')
    summary_stats = summary_stats.sort_values('median')
    
    # Add logistic regression coefficients to summary
    summary_stats['logistic_beta'] = summary_stats.index.map(logistic_values)
    
    print(summary_stats[['gene', 'median', 'mean', 'std', 'logistic_beta']])
    
    # Save statistics to CSV
    output_path = r"C:\Users\becky\OneDrive\Desktop\2023 summer intern\literature\CARD\results\doripenem_gene_strength_stats.csv"
    summary_stats.to_csv(output_path)
    print(f"\nStatistics saved to: {output_path}")
    
    return display_name_order, summary_stats


# Usage example:
kegg_file_path = r"C:\Users\becky\OneDrive\Desktop\2023 summer intern\Data\KEGG_orthology.txt"
gene_order, stats = create_gene_strength_boxplot(posterior_beta, ordered_genes, logistic_beta, kegg_file_path)




##### table 1 accuracy plots #####
data = {
    'antibiotic': [
        "amikacin", "amoxicillin-clavulanic acid", "amoxicillin", "ampicillin-sulbactam", 
        "ampicillin", "aztreonam", "cefazolin", "cefepime", "cefotaxime", "cefoxitin", 
        "ceftazidime", "ceftiofur", "ceftriaxone", "cefuroxime", "chloramphenicol", 
        "ciprofloxacin", "clindamycin", "doripenem", "ertapenem", "erythromycin", 
        "gentamicin", "imipenem", "kanamycin", "levofloxacin", "meropenem", 
        "piperacillin-tazobactam", "tetracycline", "tobramycin", "trimethoprim-sulfamethoxazole"
    ],
    'logistic_accuracy': [
        0.930, 0.806, 0.971, 0.957, 0.976, 0.800, 0.880, 0.772, 0.987, 0.808,
        0.842, 0.806, 0.837, 0.917, 0.996, 0.794, 0.971, 0.854, 0.874, 0.938,
        0.941, 0.920, 0.987, 0.859, 0.912, 0.879, 0.951, 0.908, 0.973
    ],
    'nuts_full_constraint': [
        0.135, 0.252, 0.971, 0.443, 0.453, 0.733, 0.880, 0.662, 0.907, 0.265,
        0.835, 0.197, 0.311, 0.917, 0.071, 0.720, 0.971, 0.528, 0.583, 0.953,
        0.212, 0.670, 0.094, 0.709, 0.473, 0.589, 0.450, 0.417, 0.210
    ],
    'rwmh_full_constraint': [
        0.926, 0.810, 0.971, 0.964, 0.988, 0.800, 0.880, 0.772, 0.987, 0.811,
        0.850, 0.803, 0.843, 0.917, 0.996, 0.799, 0.971, 0.927, 0.858, 0.953,
        0.941, 0.920, 0.991, 0.844, 0.912, 0.860, 0.963, 0.936, 0.975
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Sort by logistic regression accuracy for better visualization
df = df.sort_values('logistic_accuracy').reset_index(drop=True)

def create_individual_plots(df, save_dir=None):
    """Create individual plots separately"""
    
    x_pos = np.arange(len(df))
    
    # Plot 1: BaLLM (RWMH) vs Logistic Regression
    plt.figure(figsize=(15, 6))
    plt.plot(x_pos, df['logistic_accuracy'], 'b-', linewidth=2, marker='o', 
             markersize=6, label='Logistic Regression', color='#1f77b4')
    plt.scatter(x_pos, df['rwmh_full_constraint'], s=60, color='#ff7f0e', 
                label='BaLLM (RWMH)', alpha=0.8, zorder=5)
    
    plt.title('BaLLM (RWMH) vs Logistic Regression Performance', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Antibiotics', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(x_pos, df['antibiotic'], rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f"{save_dir}/ballm_rwmh_vs_logistic.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: BaLLM (NUTS) vs BaLLM (RWMH)
    plt.figure(figsize=(15, 6))
    plt.plot(x_pos, df['nuts_full_constraint'], 'g-', linewidth=2, marker='o', 
             markersize=6, label='BaLLM (NUTS)', color='#2ca02c')
    plt.scatter(x_pos, df['rwmh_full_constraint'], s=60, color='#ff7f0e', 
                label='BaLLM (RWMH)', alpha=0.8, zorder=5)
    
    plt.title('BaLLM (NUTS) vs BaLLM (RWMH) Performance', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Antibiotics', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(x_pos, df['antibiotic'], rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f"{save_dir}/ballm_nuts_vs_rwmh.png", dpi=300, bbox_inches='tight')
    plt.show()



create_individual_plots(df, r"C:\Users\becky\OneDrive\Desktop\2023 summer intern\literature\CARD\results")























###############    test models on Shelburne dataset  ##################
shelburne_gene=pd.read_csv('C:/Users/becky/OneDrive/Desktop/2023 summer intern/Data/purified_data/Shelburne_VAMP.csv')
shelburne_pheno=pd.read_csv('C:/Users/becky/OneDrive/Desktop/2023 summer intern/Data/purified_data/Shelburne_phenotype.csv')

###  format column   ###
shelburne_pheno['phenotype'] = shelburne_pheno['phenotype'].map({'resistant': 1, 'susceptible': 0})
shelburne_gene['variants'] = shelburne_gene['variants'].str.split('.').str[0]


### Merge the dataframes on `sample_id` ###
merged_df = pd.merge(shelburne_pheno, shelburne_gene, on='sample_id', how='left')

###  Split by antibiotics ###
shelburne_dict = {}

for abx in merged_df['antibiotics'].unique():
    key = f'shelburne_{abx.lower()}'
    shelburne_dict[key] = merged_df[merged_df['antibiotics'] == abx].copy()

# Example: shelburne_dict['shelburne_ertapenem']


# Dictionary to hold wide-format bayes dataframes
shelburne_bayes_df_dict = {}

# Loop through each antibiotic in final_kg
for abx in final_kg['antibiotic'].unique():
    abx_key = f'shelburne_{abx.lower()}'
    
    # Skip if antibiotic is not found in shelburne_dict
    if abx_key not in shelburne_dict:
        print(f"Warning: {abx_key} not found in shelburne_dict. Skipping...")
        continue

    # Subset final_kg for this antibiotic
    abx_genes = final_kg[final_kg['antibiotic'] == abx]['gene'].unique()
    
    # Subset the long dataframe for this antibiotic
    abx_long_df = shelburne_dict[abx_key]

    # Get unique sample_id, bacteria, and phenotype (first occurrence per sample)
    sample_info = (
        abx_long_df
        .drop_duplicates(subset=['sample_id'])[['sample_id', 'bacteria', 'phenotype']]
        .reset_index(drop=True)
    )

    # Create the base bayes dataframe
    abx_bayes = sample_info.copy()

    # Add gene columns, initialized to 0
    for gene in abx_genes:
        abx_bayes[gene] = 0

    # Fill gene values based on presence in variants
    for i, row in abx_bayes.iterrows():
        sample_id = row['sample_id']
        
        # Get all variants for this sample_id
        sample_variants = abx_long_df[abx_long_df['sample_id'] == sample_id]['variants'].unique()
        
        for gene in abx_genes:
            if gene in sample_variants:
                abx_bayes.at[i, gene] = 1

    # Save to dictionary
    shelburne_bayes_df_dict[f'{abx.lower()}_bayes'] = abx_bayes





def evaluate_all_antibiotics(
    test_df_dict, model_dir
):
    results = []

    for key in test_df_dict.keys():
        if not key.endswith("_bayes"):
            continue

        abx = key.replace("_bayes", "")
        print(f"Evaluating antibiotic: {abx}")

        # Step 1: Extract data
        df = test_df_dict[key].copy()
        non_gene_cols = ['sample_id', 'bacteria', 'phenotype']
        gene_cols = [col for col in df.columns if col not in non_gene_cols]

        X = df[gene_cols].astype(float)
        y = df['phenotype'].astype(int)

        # Load models and gene strength
        try:
            with open(os.path.join(model_dir, f"{abx}_logistic_model.pkl"), "rb") as f:
                clf = pickle.load(f)

            # bayes = np.load(os.path.join(model_dir, f"{abx}_posterior_params.npz"))
            bayes = np.load(os.path.join(model_dir, f"{abx}_posterior_params_2.5.npz"))
            posterior_beta = bayes["posterior_beta"]
            posterior_beta0 = bayes["posterior_beta0"]

            gene_strength = pd.read_csv(os.path.join(model_dir, f"{abx}_gene_strength.csv"))
            ordered_genes = gene_strength['gene'].tolist()
        except FileNotFoundError:
            print(f"Model or data missing for {abx}, skipping.")
            continue

        # Fill missing genes with 0
        X_filled = pd.DataFrame(0, index=X.index, columns=ordered_genes)
        for gene in ordered_genes:
            if gene in X.columns:
                X_filled[gene] = X[gene].astype(float)
        
        # count how many columns are all 0:
        num_all_zero_columns = (X_filled == 0).all(axis=0).sum()  
        percentage = num_all_zero_columns / X_filled.shape[1] * 100
        print(f"All-zero columns: {num_all_zero_columns} / {X_filled.shape[1]} ({percentage:.2f}%)")


        X_valid = X_filled.values

        # Logistic model prediction
        y_pred_logistic = clf.predict(X_valid)
        acc_logistic = accuracy_score(y, y_pred_logistic)

        # Bayesian model prediction
        
        logits = posterior_beta0 + X_valid @ posterior_beta
        probs = 1 / (1 + np.exp(-logits))
        
        row_mean = np.mean(probs, axis=1)
        row_std = np.std(probs, axis=1)
        df_stats = pd.DataFrame({
            'mean': row_mean,
            'std': row_std})
        
        P_mean = probs.mean(axis=1)
        y_pred_bayes = (P_mean > 0.5).astype(int)

        # P_median = np.median(probs, axis=1)
        # y_pred_bayes = (P_median > 0.5).astype(int)

        acc_bayes = accuracy_score(y, y_pred_bayes)

        # Class balance info
        percent_1 = y.mean() * 100
        unique, counts = np.unique(y_pred_bayes, return_counts=True)
        bayes_prediction_counts = dict(zip(unique, counts))
        always_majority_class = len(bayes_prediction_counts) == 1 and (
            list(bayes_prediction_counts.keys())[0] == int(y.mean() > 0.5)
        )
        
        print(f"  - Number of samples: {len(y)}")
        print(f"  - % of resistant (y=1): {percent_1:.2f}%")
        print(f"  - Logistic Accuracy:   {acc_logistic:.4f}")
        print(f"  - Bayesian Accuracy:   {acc_bayes:.4f}")
        if always_majority_class:
            print(f"  - Bayesian model predicted only the majority class ({list(bayes_prediction_counts.keys())[0]})!")

        results.append({
            "antibiotic": abx,
            "num_samples": len(y),
            "%_resistant": percent_1,
            "logistic_acc": acc_logistic,
            "bayesian_acc": acc_bayes,
            "bayes_majority_class": always_majority_class
        })

    return pd.DataFrame(results)



results_df = evaluate_all_antibiotics(
    test_df_dict=shelburne_bayes_df_dict,
    model_dir=r"C:\Users\becky\OneDrive\Desktop\2023 summer intern\literature\CARD\results\KG model parameters"
)



###  ROC ###
def plot_roc_curve(abx, test_df_dict, model_dir, method='bayesian'):
    key = f"{abx}_bayes"
    df = test_df_dict[key].copy()

    # Prepare features and labels
    non_gene_cols = ['sample_id', 'bacteria', 'phenotype']
    gene_cols = [col for col in df.columns if col not in non_gene_cols]
    X = df[gene_cols].astype(float)
    y = df['phenotype'].astype(int)

    # Load model components
    with open(os.path.join(model_dir, f"{abx}_logistic_model.pkl"), "rb") as f:
        clf = pickle.load(f)
    bayes = np.load(os.path.join(model_dir, f"{abx}_posterior_params.npz"))
    posterior_beta = bayes["posterior_beta"]
    posterior_beta0 = bayes["posterior_beta0"]
    gene_strength = pd.read_csv(os.path.join(model_dir, f"{abx}_gene_strength.csv"))
    ordered_genes = gene_strength['gene'].tolist()

    # Align columns
    X_filled = pd.DataFrame(0, index=X.index, columns=ordered_genes)
    for gene in ordered_genes:
        if gene in X.columns:
            X_filled[gene] = X[gene].astype(float)
    X_valid = X_filled.values

    # Get predicted probabilities
    if method == 'bayesian':
        logits = posterior_beta0 + X_valid @ posterior_beta
        probs = 1 / (1 + np.exp(-logits))
        P_mean = probs.mean(axis=1)
        y_score = P_mean
    elif method == 'logistic':
        y_score = clf.predict_proba(X_valid)[:, 1]
    else:
        raise ValueError("method must be 'bayesian' or 'logistic'")

    # Compute ROC
    fpr, tpr, _ = roc_curve(y, y_score)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc


# Plot ROC curves for two antibiotics
plt.figure(figsize=(8, 6))
for abx in ["amikacin", "ceftazidime"]:
    fpr, tpr, roc_auc = plot_roc_curve(
        abx,
        test_df_dict=shelburne_bayes_df_dict,
        model_dir=r"C:\Users\becky\OneDrive\Desktop\2023 summer intern\literature\CARD\results\KG model parameters",
        method="bayesian"
    )
    plt.plot(fpr, tpr, label=f"{abx} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Bayesian ROC Curves")
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
for abx in ["amikacin", "ceftazidime"]:
    fpr, tpr, roc_auc = plot_roc_curve(
        abx,
        test_df_dict=shelburne_bayes_df_dict,
        model_dir=r"C:\Users\becky\OneDrive\Desktop\2023 summer intern\literature\CARD\results\KG model parameters",
        method="logistic"
    )
    plt.plot(fpr, tpr, label=f"{abx} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic ROC Curves")
plt.legend()
plt.tight_layout()
plt.show()






###############    test models on ARIsolate dataset  ##################
ariosolate_gene=pd.read_csv('C:/Users/becky/OneDrive/Desktop/2023 summer intern/Data/purified_data/ARIsolateBank_VAMP.csv')
ariosolate_pheno=pd.read_csv('C:/Users/becky/OneDrive/Desktop/2023 summer intern/Data/purified_data/ARIsolateBank_phenotype.csv')

###  format column   ###
ariosolate_pheno['phenotype'] = ariosolate_pheno['phenotype'].map({'resistant': 1, 'susceptible': 0})
ariosolate_gene['variants'] = ariosolate_gene['variants'].str.split('.').str[0]


### Merge the dataframes on `sample_id` ###
merged_df = pd.merge(ariosolate_pheno, ariosolate_gene, on='sample_id', how='left')

###  Split by antibiotics ###
ariosolate_dict = {}

for abx in merged_df['antibiotics'].unique():
    key = f'ariosolate_{abx.lower()}'
    ariosolate_dict[key] = merged_df[merged_df['antibiotics'] == abx].copy()

# Example: ariosolate_dict['ariosolate_ertapenem']


# Dictionary to hold wide-format bayes dataframes
ariosolate_bayes_df_dict = {}

# Loop through each antibiotic in final_kg
for abx in final_kg['antibiotic'].unique():
    abx_key = f'ariosolate_{abx.lower()}'
    
    # Skip if antibiotic is not found in shelburne_dict
    if abx_key not in ariosolate_dict:
        print(f"Warning: {abx_key} not found in ariosolate_dict. Skipping...")
        continue

    # Subset final_kg for this antibiotic
    abx_genes = final_kg[final_kg['antibiotic'] == abx]['gene'].unique()
    
    # Subset the long dataframe for this antibiotic
    abx_long_df = ariosolate_dict[abx_key]

    # Get unique sample_id, bacteria, and phenotype (first occurrence per sample)
    sample_info = (
        abx_long_df
        .drop_duplicates(subset=['sample_id'])[['sample_id', 'bacteria', 'phenotype']]
        .reset_index(drop=True)
    )

    # Create the base bayes dataframe
    abx_bayes = sample_info.copy()

    # Add gene columns, initialized to 0
    for gene in abx_genes:
        abx_bayes[gene] = 0

    # Fill gene values based on presence in variants
    for i, row in abx_bayes.iterrows():
        sample_id = row['sample_id']
        
        # Get all variants for this sample_id
        sample_variants = abx_long_df[abx_long_df['sample_id'] == sample_id]['variants'].unique()
        
        for gene in abx_genes:
            if gene in sample_variants:
                abx_bayes.at[i, gene] = 1

    # Save to dictionary
    ariosolate_bayes_df_dict[f'{abx.lower()}_bayes'] = abx_bayes




## calculate number of Kegg genes for bacteria Citrobacter freundii: 16 samples ##
cf_pheno = ariosolate_pheno[ariosolate_pheno['bacteria'] == 'Citrobacter freundii']

gene_counts = []

for _, row in cf_pheno.iterrows():
    sample_id = row['sample_id']
    antibiotic = row['antibiotics']
    
    key = f"{antibiotic}_bayes"
    
    if key not in ariosolate_bayes_df_dict:
        gene_counts.append(None)
        continue
    
    df = ariosolate_bayes_df_dict[key]
    
    # Find the matching row
    match = df[(df['sample_id'] == sample_id) & (df['bacteria'] == 'Citrobacter freundii')]
    
    if match.empty:
        gene_counts.append(None)
        continue
    
    # Count 1s across gene columns (skip first 3: sample_id, bacteria, phenotype)
    gene_cols = df.columns[3:]
    num_genes = (match.iloc[0][gene_cols] == 1).sum()
    gene_counts.append(num_genes)

cf_pheno = cf_pheno.copy()
cf_pheno['num_genes'] = gene_counts

print(cf_pheno['num_genes'].describe())
print(f"\nMin: {cf_pheno['num_genes'].min()}")
print(f"Max: {cf_pheno['num_genes'].max()}")










def evaluate_all_antibiotics_by_bacteria(test_df_dict, model_dir):
    """
    Evaluate models and calculate accuracy by bacteria type
    Returns detailed predictions and accuracy by bacteria
    """
    all_predictions = []
    
    for key in test_df_dict.keys():
        if not key.endswith("_bayes"):
            continue
        
        abx = key.replace("_bayes", "")
        print(f"Evaluating antibiotic: {abx}")
        
        # Extract data
        df = test_df_dict[key].copy()
        non_gene_cols = ['sample_id', 'bacteria', 'phenotype']
        gene_cols = [col for col in df.columns if col not in non_gene_cols]
        
        X = df[gene_cols].astype(float)
        y = df['phenotype'].astype(int)
        
        # Load models
        try:
            with open(os.path.join(model_dir, f"{abx}_logistic_model.pkl"), "rb") as f:
                clf = pickle.load(f)
            
            bayes = np.load(os.path.join(model_dir, f"{abx}_posterior_params_2.5.npz"))
            posterior_beta = bayes["posterior_beta"]
            posterior_beta0 = bayes["posterior_beta0"]
            
            gene_strength = pd.read_csv(os.path.join(model_dir, f"{abx}_gene_strength.csv"))
            ordered_genes = gene_strength['gene'].tolist()
        except FileNotFoundError:
            print(f"Model or data missing for {abx}, skipping.")
            continue
        
        # Fill missing genes with 0
        X_filled = pd.DataFrame(0, index=X.index, columns=ordered_genes)
        for gene in ordered_genes:
            if gene in X.columns:
                X_filled[gene] = X[gene].astype(float)
        
        X_valid = X_filled.values
        
        # Logistic model prediction
        y_pred_logistic = clf.predict(X_valid)
        
        # Bayesian model prediction
        logits = posterior_beta0 + X_valid @ posterior_beta
        probs = 1 / (1 + np.exp(-logits))
        P_mean = probs.mean(axis=1)
        y_pred_bayes = (P_mean > 0.5).astype(int)
        
        # Store predictions with sample info
        for i, idx in enumerate(df.index):
            all_predictions.append({
                'sample_id': df.loc[idx, 'sample_id'],
                'bacteria': df.loc[idx, 'bacteria'],
                'antibiotic': abx,
                'true_phenotype': y.iloc[i],
                'logistic_pred': y_pred_logistic[i],
                'bayes_pred': y_pred_bayes[i]
            })
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame(all_predictions)
    
    # Calculate overall accuracy by antibiotic
    print("\n" + "="*80)
    print("OVERALL ACCURACY BY ANTIBIOTIC")
    print("="*80)
    
    overall_results = []
    for abx in predictions_df['antibiotic'].unique():
        abx_data = predictions_df[predictions_df['antibiotic'] == abx]
        
        logistic_acc = accuracy_score(abx_data['true_phenotype'], abx_data['logistic_pred'])
        bayes_acc = accuracy_score(abx_data['true_phenotype'], abx_data['bayes_pred'])
        
        overall_results.append({
            'antibiotic': abx,
            'num_samples': len(abx_data),
            'logistic_acc': logistic_acc,
            'bayes_acc': bayes_acc
        })
        
        print(f"{abx}: Logistic={logistic_acc:.4f}, Bayes={bayes_acc:.4f}, N={len(abx_data)}")
    
    overall_df = pd.DataFrame(overall_results)
    
    # Calculate accuracy by bacteria type (aggregated across all antibiotics)
    print("\n" + "="*80)
    print("ACCURACY BY BACTERIA TYPE")
    print("="*80)
    
    bacteria_results = []
    for bacteria in predictions_df['bacteria'].unique():
        bacteria_data = predictions_df[predictions_df['bacteria'] == bacteria]
        
        logistic_acc = accuracy_score(bacteria_data['true_phenotype'], 
                                      bacteria_data['logistic_pred'])
        bayes_acc = accuracy_score(bacteria_data['true_phenotype'], 
                                  bacteria_data['bayes_pred'])
        
        bacteria_results.append({
            'bacteria': bacteria,
            'num_samples': len(bacteria_data),
            'percent_resistant': bacteria_data['true_phenotype'].mean() * 100,
            'logistic_acc': logistic_acc,
            'bayes_acc': bayes_acc
        })
        
        print(f"{bacteria}: Logistic={logistic_acc:.4f}, Bayes={bayes_acc:.4f}, N={len(bacteria_data)}")
    
    bacteria_df = pd.DataFrame(bacteria_results)
    bacteria_df = bacteria_df.sort_values('bacteria')
    
    return predictions_df, overall_df, bacteria_df


# Run evaluation
predictions_df, overall_df, bacteria_df = evaluate_all_antibiotics_by_bacteria(
    test_df_dict=ariosolate_bayes_df_dict,
    model_dir=r"C:\Users\becky\OneDrive\Desktop\2023 summer intern\literature\CARD\results\KG model parameters"
)
output_path = r"C:\Users\becky\OneDrive\Desktop\2023 summer intern\literature\CARD\results\Bayesian model test_ARIsolate_by_bacteria.csv"
bacteria_df.to_csv(output_path, index=False)




###############    test models on CF dataset  ##################
cf_gene=pd.read_csv('C:/Users/becky/OneDrive/Desktop/2023 summer intern/Data/purified_data/cf_VAMP.csv')
cf_pheno=pd.read_csv('C:/Users/becky/OneDrive/Desktop/2023 summer intern/Data/purified_data/cf_phenotype.csv')

###  format column   ###
cf_pheno['phenotype'] = cf_pheno['phenotype'].map({'resistant': 1, 'susceptible': 0})
cf_gene['variants'] = cf_gene['variants'].str.split('.').str[0]


### Merge the dataframes on `sample_id` ###
merged_df = pd.merge(cf_pheno, cf_gene, on='sample_id', how='left')

###  Split by antibiotics ###
cf_dict = {}

for abx in merged_df['antibiotics'].unique():
    key = f'cf_{abx.lower()}'
    cf_dict[key] = merged_df[merged_df['antibiotics'] == abx].copy()



# Dictionary to hold wide-format bayes dataframes
cf_bayes_df_dict = {}

# Loop through each antibiotic in final_kg
for abx in final_kg['antibiotic'].unique():
    abx_key = f'cf_{abx.lower()}'
    
    # Skip if antibiotic is not found in cf_dict
    if abx_key not in cf_dict:
        print(f"Warning: {abx_key} not found in cf_dict. Skipping...")
        continue

    # Subset final_kg for this antibiotic
    abx_genes = final_kg[final_kg['antibiotic'] == abx]['gene'].unique()
    
    # Subset the long dataframe for this antibiotic
    abx_long_df = cf_dict[abx_key]

    # Get unique sample_id, bacteria, and phenotype (first occurrence per sample)
    sample_info = (
        abx_long_df
        .drop_duplicates(subset=['sample_id'])[['sample_id', 'bacteria', 'phenotype']]
        .reset_index(drop=True)
    )

    # Create the base bayes dataframe
    abx_bayes = sample_info.copy()

    # Add gene columns, initialized to 0
    for gene in abx_genes:
        abx_bayes[gene] = 0

    # Fill gene values based on presence in variants
    for i, row in abx_bayes.iterrows():
        sample_id = row['sample_id']
        
        # Get all variants for this sample_id
        sample_variants = abx_long_df[abx_long_df['sample_id'] == sample_id]['variants'].unique()
        
        for gene in abx_genes:
            if gene in sample_variants:
                abx_bayes.at[i, gene] = 1

    # Save to dictionary
    cf_bayes_df_dict[f'{abx.lower()}_bayes'] = abx_bayes


results_df = evaluate_all_antibiotics(
    test_df_dict=cf_bayes_df_dict,
    model_dir=r"C:\Users\becky\OneDrive\Desktop\2023 summer intern\literature\CARD\results\KG model parameters"
)

output_path = r"C:\Users\becky\OneDrive\Desktop\2023 summer intern\literature\CARD\results\Bayesian model test_cf.csv"
results_df.to_csv(output_path, index=False)




###############    test models on german dataset  ##################
german_gene=pd.read_csv('C:/Users/becky/OneDrive/Desktop/2023 summer intern/Data/purified_data/german_VAMP.csv')
german_pheno=pd.read_csv('C:/Users/becky/OneDrive/Desktop/2023 summer intern/Data/purified_data/german_phenotype.csv')

###  format column   ###
german_pheno['phenotype'] = german_pheno['phenotype'].map({'resistant': 1, 'susceptible': 0})
german_gene['variants'] = german_gene['variants'].str.split('.').str[0]


### Merge the dataframes on `sample_id` ###
merged_df = pd.merge(german_pheno, german_gene, on='sample_id', how='left')

###  Split by antibiotics ###
german_dict = {}

for abx in merged_df['antibiotics'].unique():
    key = f'german_{abx.lower()}'
    german_dict[key] = merged_df[merged_df['antibiotics'] == abx].copy()



# Dictionary to hold wide-format bayes dataframes
german_bayes_df_dict = {}

# Loop through each antibiotic in final_kg
for abx in final_kg['antibiotic'].unique():
    abx_key = f'german_{abx.lower()}'
    
    # Skip if antibiotic is not found in german_dict
    if abx_key not in german_dict:
        print(f"Warning: {abx_key} not found in germane_dict. Skipping...")
        continue

    # Subset final_kg for this antibiotic
    abx_genes = final_kg[final_kg['antibiotic'] == abx]['gene'].unique()
    
    # Subset the long dataframe for this antibiotic
    abx_long_df = german_dict[abx_key]

    # Get unique sample_id, bacteria, and phenotype (first occurrence per sample)
    sample_info = (
        abx_long_df
        .drop_duplicates(subset=['sample_id'])[['sample_id', 'bacteria', 'phenotype']]
        .reset_index(drop=True)
    )

    # Create the base bayes dataframe
    abx_bayes = sample_info.copy()

    # Add gene columns, initialized to 0
    for gene in abx_genes:
        abx_bayes[gene] = 0

    # Fill gene values based on presence in variants
    for i, row in abx_bayes.iterrows():
        sample_id = row['sample_id']
        
        # Get all variants for this sample_id
        sample_variants = abx_long_df[abx_long_df['sample_id'] == sample_id]['variants'].unique()
        
        for gene in abx_genes:
            if gene in sample_variants:
                abx_bayes.at[i, gene] = 1

    # Save to dictionary
    german_bayes_df_dict[f'{abx.lower()}_bayes'] = abx_bayes


results_df = evaluate_all_antibiotics(
    test_df_dict=german_bayes_df_dict,
    model_dir=r"C:\Users\becky\OneDrive\Desktop\2023 summer intern\literature\CARD\results\KG model parameters"
)

output_path = r"C:\Users\becky\OneDrive\Desktop\2023 summer intern\literature\CARD\results\Bayesian model test_german.csv"
results_df.to_csv(output_path, index=False)



###############    test models on astrazeneca dataset  ##################
pheno_path = r"C:/Users/becky/OneDrive/Desktop/2023 summer intern/Data/purified_data/AstraZeneca_phenotype.txt"
astrazeneca_pheno = pd.read_csv(pheno_path, sep='\t', header=None, usecols=[0, 1, 2], 
                                names=['sample_id', 'antibiotics', 'phenotype'])

geno_path = r"C:/Users/becky/OneDrive/Desktop/2023 summer intern/Data/purified_data/AstraZeneca_VAMP.txt"
astrazeneca_gene = pd.read_csv(geno_path, sep='\t', header=None, 
                               names=['sample_id', 'variants'])


###  format column   ###
astrazeneca_pheno['phenotype'] = astrazeneca_pheno['phenotype'].map({'resistant': 1, 'susceptible': 0})
astrazeneca_pheno = astrazeneca_pheno.dropna(subset=['phenotype'])
astrazeneca_gene['variants'] = astrazeneca_gene['variants'].str.split('.').str[0]


### Merge the dataframes on `sample_id` ###
merged_df = pd.merge(astrazeneca_pheno, astrazeneca_gene, on='sample_id', how='left')

###  Split by antibiotics ###
astrazeneca_dict = {}

for abx in merged_df['antibiotics'].unique():
    key = f'astrazeneca_{abx.lower()}'
    astrazeneca_dict[key] = merged_df[merged_df['antibiotics'] == abx].copy()



# Dictionary to hold wide-format bayes dataframes
astrazeneca_bayes_df_dict = {}

# Loop through each antibiotic in final_kg
for abx in final_kg['antibiotic'].unique():
    abx_key = f'astrazeneca_{abx.lower()}'
    
    # Skip if antibiotic is not found in astrazeneca_dict
    if abx_key not in astrazeneca_dict:
        print(f"Warning: {abx_key} not found in astrazeneca_dict. Skipping...")
        continue

    # Subset final_kg for this antibiotic
    abx_genes = final_kg[final_kg['antibiotic'] == abx]['gene'].unique()
    
    # Subset the long dataframe for this antibiotic
    abx_long_df = astrazeneca_dict[abx_key]

    # Get unique sample_id, bacteria, and phenotype (first occurrence per sample)
    sample_info = (
        abx_long_df
        .drop_duplicates(subset=['sample_id'])[['sample_id','phenotype']]
        .reset_index(drop=True)
    )

    # Create the base bayes dataframe
    abx_bayes = sample_info.copy()

    # Add gene columns, initialized to 0
    for gene in abx_genes:
        abx_bayes[gene] = 0

    # Fill gene values based on presence in variants
    for i, row in abx_bayes.iterrows():
        sample_id = row['sample_id']
        
        # Get all variants for this sample_id
        sample_variants = abx_long_df[abx_long_df['sample_id'] == sample_id]['variants'].unique()
        
        for gene in abx_genes:
            if gene in sample_variants:
                abx_bayes.at[i, gene] = 1

    # Save to dictionary
    astrazeneca_bayes_df_dict[f'{abx.lower()}_bayes'] = abx_bayes


results_df = evaluate_all_antibiotics(
    test_df_dict=astrazeneca_bayes_df_dict,
    model_dir=r"C:\Users\becky\OneDrive\Desktop\2023 summer intern\literature\CARD\results\KG model parameters"
)

output_path = r"C:\Users\becky\OneDrive\Desktop\2023 summer intern\literature\CARD\results\Bayesian model test_astrazeneca.csv"
results_df.to_csv(output_path, index=False)




###############    test models on rabin dataset  ##################
pheno_path = r"C:/Users/becky/OneDrive/Desktop/2023 summer intern/Data/purified_data/Rabin_phenotype.txt"
rabin_pheno = pd.read_csv(pheno_path, sep='\t', header=None, 
                                names=['sample_id', 'bacteria','antibiotics', 'phenotype'])

rabin_path = r"C:/Users/becky/OneDrive/Desktop/2023 summer intern/Data/purified_data/Rabin_VAMP.txt"
rabin_gene = pd.read_csv(rabin_path, sep='\t', header=None, 
                               names=['sample_id', 'variants'])


###  format column   ###
rabin_pheno['phenotype'] = rabin_pheno['phenotype'].map({'resistant': 1, 'susceptible': 0})
rabin_pheno = rabin_pheno.dropna(subset=['phenotype'])
rabin_gene['variants'] = rabin_gene['variants'].str.split('.').str[0]


### Merge the dataframes on `sample_id` ###
merged_df = pd.merge(rabin_pheno, rabin_gene, on='sample_id', how='left')

###  Split by antibiotics ###
rabin_dict = {}

for abx in merged_df['antibiotics'].unique():
    key = f'rabin_{abx.lower()}'
    rabin_dict[key] = merged_df[merged_df['antibiotics'] == abx].copy()



# Dictionary to hold wide-format bayes dataframes
rabin_bayes_df_dict = {}

# Loop through each antibiotic in final_kg
for abx in final_kg['antibiotic'].unique():
    abx_key = f'rabin_{abx.lower()}'
    
    # Skip if antibiotic is not found in rabin_dict
    if abx_key not in rabin_dict:
        print(f"Warning: {abx_key} not found in rabin_dict. Skipping...")
        continue

    # Subset final_kg for this antibiotic
    abx_genes = final_kg[final_kg['antibiotic'] == abx]['gene'].unique()
    
    # Subset the long dataframe for this antibiotic
    abx_long_df = rabin_dict[abx_key]

    # Get unique sample_id, bacteria, and phenotype (first occurrence per sample)
    sample_info = (
        abx_long_df
        .drop_duplicates(subset=['sample_id'])[['sample_id', 'bacteria', 'phenotype']]
        .reset_index(drop=True)
    )

    # Create the base bayes dataframe
    abx_bayes = sample_info.copy()

    # Add gene columns, initialized to 0
    for gene in abx_genes:
        abx_bayes[gene] = 0

    # Fill gene values based on presence in variants
    for i, row in abx_bayes.iterrows():
        sample_id = row['sample_id']
        
        # Get all variants for this sample_id
        sample_variants = abx_long_df[abx_long_df['sample_id'] == sample_id]['variants'].unique()
        
        for gene in abx_genes:
            if gene in sample_variants:
                abx_bayes.at[i, gene] = 1

    # Save to dictionary
    rabin_bayes_df_dict[f'{abx.lower()}_bayes'] = abx_bayes


results_df = evaluate_all_antibiotics(
    test_df_dict=rabin_bayes_df_dict,
    model_dir=r"C:\Users\becky\OneDrive\Desktop\2023 summer intern\literature\CARD\results\KG model parameters"
)

output_path = r"C:\Users\becky\OneDrive\Desktop\2023 summer intern\literature\CARD\results\Bayesian model test_rabin.csv"
results_df.to_csv(output_path, index=False)











################################# visualize KG  ###########################
df = pd.read_csv("C:\\Users\\becky\\OneDrive\\Desktop\\2023 summer intern\\literature\\CARD\\results\\final_kg.csv")  
df = df.head(10)

# Create directed graph
G = nx.DiGraph()
for _, row in df.iterrows():
    G.add_edge(row["gene"], row["antibiotic"], label=row["First Word"])

# Get the central antibiotic node (assumes only 1)
central_node = df["antibiotic"].iloc[0]
gene_nodes = df["gene"].unique()

# Manually set positions: center antibiotic, genes around in circle
import math

angle_step = 2 * math.pi / len(gene_nodes)
radius = 3
pos = {}

# Place central antibiotic at center
pos[central_node] = (0, 0)

# Place each gene node on a circle
for i, gene in enumerate(gene_nodes):
    angle = i * angle_step
    x = radius * math.cos(angle)
    y = radius * math.sin(angle)
    pos[gene] = (x, y)

# Draw nodes
nx.draw_networkx_nodes(
    G, pos,
    node_color='lightblue',
    node_size=1500  # Increase this value for bigger circles
)

# Draw edges with arrows
nx.draw_networkx_edges(
    G, pos,
    edge_color='gray',
    arrows=True,
    arrowsize=30,
    width=1,
    connectionstyle="arc3,rad=0.05"  # ↓ reduced from 0.1 to 0.05
)


# Draw node labels
nx.draw_networkx_labels(G, pos, font_size=8)

# Draw edge labels (relationship text)
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='darkred')

plt.axis('off')
plt.tight_layout()
plt.show()



########################### check Shelburne CARD RGI prediction results  ##############################################
file_path = r"C:\Users\becky\OneDrive - University of Texas Southwestern\LLM\results\purified_data.clean.deduplicate.RGI\Shelburne.RGI_pheno_table.all_samples_n_198.txt"
card = pd.read_csv(file_path, sep='\t', header=None)
card.columns = ['sample_id', 'antibiotics', 'phenotype']


# Clean sample_id in shelburne_pheno by removing 'Enterobacter_' prefix
shelburne_pheno['sample_id_clean'] = shelburne_pheno['sample_id'].str.replace('Enterobacter_', '', regex=False)

# Get unique combinations of sample_id and antibiotics from shelburne_pheno
shelburne_combinations = shelburne_pheno[['sample_id_clean', 'antibiotics']].drop_duplicates()

# Filter card dataset to only include sample_id and antibiotics that exist in shelburne_pheno
card_filtered = card.merge(
    shelburne_combinations, 
    left_on=['sample_id', 'antibiotics'], 
    right_on=['sample_id_clean', 'antibiotics'],
    how='inner'
)

print(f"Original card dataset: {len(card)} rows")
print(f"Filtered card dataset: {len(card_filtered)} rows")

# Create a set of (sample_id, antibiotics) combinations that are predicted as resistant by CARD
card_resistant_set = set(zip(card_filtered['sample_id'], card_filtered['antibiotics']))

# Add card_predicted_pheno column to shelburne_pheno
def get_card_prediction(row):
    if (row['sample_id_clean'], row['antibiotics']) in card_resistant_set:
        return 'resistant'
    else:
        return 'susceptible'

shelburne_pheno['card_predicted_pheno'] = shelburne_pheno.apply(get_card_prediction, axis=1)

# Calculate accuracy for each antibiotic
accuracy_results = []

for antibiotic in shelburne_pheno['antibiotics'].unique():
    # Filter data for this antibiotic
    antibiotic_data = shelburne_pheno[shelburne_pheno['antibiotics'] == antibiotic].copy()
    
    # Convert phenotypes to binary (1 for resistant, 0 for susceptible)
    y_true = (antibiotic_data['phenotype'] == 'resistant').astype(int)
    y_pred = (antibiotic_data['card_predicted_pheno'] == 'resistant').astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
   
    accuracy_results.append({
        'antibiotic': antibiotic,
        'accuracy': accuracy
      
    })

# Create results DataFrame
results_df = pd.DataFrame(accuracy_results)
results_df = results_df.sort_values('accuracy', ascending=False)
print("\nAccuracy by Antibiotic:")
print("=" * 80)
print(results_df.to_string(index=False, float_format='%.3f'))




########################### check ARIsolate CARD RGI prediction results  ##############################################
ariosolate_pheno=pd.read_csv('C:/Users/becky/OneDrive/Desktop/2023 summer intern/Data/purified_data/ARIsolateBank_phenotype.csv')

file_path = r"C:\Users\becky\OneDrive - University of Texas Southwestern\LLM\results\purified_data.clean.deduplicate.RGI\ARIsolateBank.RGI_pheno_table.all_samples_n_299.txt"
card = pd.read_csv(file_path, sep='\t', header=None)
card.columns = ['sample_id', 'antibiotics', 'phenotype']

# Get unique combinations of sample_id and antibiotics from ariosolate_pheno
ariosolate_combinations = ariosolate_pheno[['sample_id','antibiotics']].drop_duplicates()

# Filter card dataset to only include sample_id and antibiotics that exist in shelburne_pheno
card_filtered = card.merge(
    ariosolate_combinations, 
    left_on=['sample_id', 'antibiotics'], 
    right_on=['sample_id', 'antibiotics'],
    how='inner'
)


# Create a set of (sample_id, antibiotics) combinations that are predicted as resistant by CARD
card_resistant_set = set(zip(card_filtered['sample_id'], card_filtered['antibiotics']))

# Add card_predicted_pheno column to ariosolate_pheno
def get_card_prediction(row):
    if (row['sample_id'], row['antibiotics']) in card_resistant_set:
        return 'resistant'
    else:
        return 'susceptible'

ariosolate_pheno['card_predicted_pheno'] = ariosolate_pheno.apply(get_card_prediction, axis=1)

# Calculate accuracy for each bacteria
accuracy_results = []

for bacteria in ariosolate_pheno['bacteria'].unique():
    # Filter data for this bacteria
    bacteria_data = ariosolate_pheno[ariosolate_pheno['bacteria'] == bacteria].copy()
    
    # Convert phenotypes to binary (1 for resistant, 0 for susceptible)
    y_true = (bacteria_data['phenotype'] == 'resistant').astype(int)
    y_pred = (bacteria_data['card_predicted_pheno'] == 'resistant').astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
   
    accuracy_results.append({
        'bacteria': bacteria,
        'accuracy': accuracy
      
    })

# Create results DataFrame
results_df = pd.DataFrame(accuracy_results)
results_df = results_df.sort_values('accuracy', ascending=False)
print("\nAccuracy by Antibiotic:")
print("=" * 80)
print(results_df.to_string(index=False, float_format='%.3f'))

























