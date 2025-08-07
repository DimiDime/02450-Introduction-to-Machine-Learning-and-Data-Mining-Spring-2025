#%% Imports
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import torch
import warnings
warnings.filterwarnings("ignore")
#Reg a
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from dtuimldmtools import rlr_validate
#Reg b
from scipy import stats
from scipy.stats import ttest_rel
from dtuimldmtools import draw_neural_net, train_neural_net
from dtuimldmtools.statistics.statistics import correlated_ttest
# Class
from dtuimldmtools import categoric2numeric
from scipy.linalg import svd 
import sklearn.linear_model as lm
from dtuimldmtools import correlated_ttest, mcnemar, categoric2numeric, visualize_decision_boundary, rlr_validate, draw_neural_net, train_neural_net
from sklearn import model_selection
import torch
from sklearn.model_selection import train_test_split
import seaborn as sns
#%% Preset
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Use full width of the display
pd.set_option('display.max_colwidth', None)  # Show full content of each column
pd.set_option('display.expand_frame_repr', False)  # Don't wrap to multiple lines

colors = ["royalblue", "darkorange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]
font1 = {'family':'arial','color':'black','fontsize':16}
font2 = {'family':'arial','color':'black','fontsize':15}
font3 = {'family':'arial','color':'black','fontsize':12}

df = pd.read_parquet("DKHousingPrices.parquet")

#%%##########
### Extra ###
#############

# Remove properties built before 1850
df_filtered = df[df["year_build"] >= 1850]

# Remove properties with purchase_price above the 99th percentile
price_threshold = df_filtered["purchase_price"].quantile(0.99)
df_filtered = df_filtered[df_filtered["purchase_price"] <= price_threshold]

# rooms
rooms_treshold = df_filtered["no_rooms"].quantile(0.99)
df_filtered = df_filtered[df_filtered["no_rooms"] <= rooms_treshold]

# sqm
sqm_treshold = df_filtered["sqm"].quantile(0.99)
df_filtered = df_filtered[df_filtered["sqm"] <= sqm_treshold]

# Remove extreme values for % change between offer and purchase
change_lower = df_filtered["%_change_between_offer_and_purchase"].quantile(0.01)
change_upper = df_filtered["%_change_between_offer_and_purchase"].quantile(0.99)
df_filtered = df_filtered[(df_filtered["%_change_between_offer_and_purchase"] >= change_lower) & 
                          (df_filtered["%_change_between_offer_and_purchase"] <= change_upper)]

# Remove extreme values for sqm price
sqm_price_threshold = df_filtered["sqm_price"].quantile(0.99)
df_filtered = df_filtered[df_filtered["sqm_price"] <= sqm_price_threshold]

# Display how many rows were removed
print(f"Rows removed: {df.shape[0] - df_filtered.shape[0]}")

# Define attributes
numeric_variables = ["year_build", "purchase_price", "%_change_between_offer_and_purchase", 
                   "no_rooms", "sqm", "sqm_price", "zip_code", 
                   "nom_interest_rate%", "dk_ann_infl_rate%", "yield_on_mortgage_credit_bonds%"]

df_numeric = df_filtered[numeric_variables]

# Convert into float for math calculations
df_numeric = df_numeric.astype(np.float64)
print(df_numeric.dtypes) 


#%%######################## 
### Data Manipulation #####
###########################

df[df['sqm_price'].isnull()]
df = df[df['sqm'].notna()]
#%% Remove unique house id
df = df.drop('house_id', axis=1)
# Impute missing values
df['dk_ann_infl_rate%'] = df['dk_ann_infl_rate%'].fillna(1.6)
df['yield_on_mortgage_credit_bonds%'] = df['yield_on_mortgage_credit_bonds%'].fillna(4.1)
df.columns = df.columns.str.strip()
df["sales_type"] = df["sales_type"].replace("-", "other_sale")


#%%#######################
### REGRESSION, Part a ###
##########################
# X
attribute_names = ["year_build", 
                   "%_change_between_offer_and_purchase", 
                   "no_rooms", 
                   "sqm", 
                   #"sqm_price", 
                   "zip_code", 
                   "nom_interest_rate%", 
                   "dk_ann_infl_rate%", 
                   #"yield_on_mortgage_credit_bonds%"
                   ]
# y
class_name = ["purchase_price"]

#X = df_numeric[attribute_names].values
#y = df_numeric["purchase_price"].values.ravel()

X = df[attribute_names].values
y = df["purchase_price"].values.ravel()

# Take subset
np.random.seed(0)
idx = np.random.choice(len(X), 1000, replace=False)
X, y = X[idx], y[idx]

# Add offset/bias term
X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
attribute_names = ["Offset"] + attribute_names
M = X.shape[1]

''' Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = X_scaled
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
'''

# CV setup
K = 10
CV = model_selection.KFold(K, shuffle=True)
# Regularization
#lambdas = np.power(50.0, range(-1, 7))
lambda_interval = np.logspace(-1, 6, 50)

# Initialize variables for storing results
Error_train = np.empty(K)
Error_test = np.empty(K)
Error_train_rlr = np.empty(K)
Error_test_rlr = np.empty(K)
Error_train_nofeatures = np.empty(K)
Error_test_nofeatures = np.empty(K)
w_rlr = np.empty((M, K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))

#Check:
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

#%% Outer cross-validation loop
for k, (train_index, test_index) in enumerate(CV.split(X, y)):
    #print(f'Crossvalidation fold: {k+1}/{K}')
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Standardize features (except offset)
    mu[k, :] = np.mean(X_train[:, 1:], axis=0)
    sigma[k, :] = np.std(X_train[:, 1:], axis=0)
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :]) / sigma[k, :]
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :]) / sigma[k, :]
    
    # Standardize y
    y_mu = np.mean(y_train)
    y_sigma = np.std(y_train)
    y_train = (y_train - y_mu) / y_sigma
    y_test = (y_test - y_mu) / y_sigma
    
    # Inner cross-validation to find optimal lambda
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = \
        rlr_validate(X_train, y_train, lambda_interval, 10)
        
    # Get minimum val error
    min_error = np.min(test_err_vs_lambda)    
    
    # Compute baseline errors (predicting mean)
    Error_train_nofeatures[k] = np.square(y_train - y_train.mean()).mean()
    Error_test_nofeatures[k] = np.square(y_test - y_test.mean()).mean()
    
    # Estimate weights with optimal lambda
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0, 0] = 0  # Don't regularize bias term
    
    w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty)
    
    # Compute errors
    Error_train_rlr[k] = np.square(y_train - X_train @ w_rlr[:, k]).mean()
    Error_test_rlr[k] = np.square(y_test - X_test @ w_rlr[:, k]).mean()
    
    new_names = {
      'Offset': 'Bias',
      '%_change_between_offer_and_purchase': '%_change',
      'yield_on_mortgage_credit_bonds%': 'yield_%',
      'nom_interest_rate%': 'int_rate%',
      'dk_ann_infl_rate%': 'infl_rate%',
      'gross_income': 'gross_inc',
      'net_income': 'net_inc',
      'num_persons_in_household': 'household_size',
      'savings_account_balance': 'savings',
      'housing_expenses': 'housing_costs',
      'purchase_price': 'price'
    }
    
    attribute_names_short = [new_names.get(name, name) for name in attribute_names]
    
    # Plot for last fold
    if k == K-1:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.semilogx(lambda_interval, mean_w_vs_lambda.T[:, 1:], '.-')
        plt.xlabel('Regularization factor (λ)', fontdict=font1)
        plt.ylabel('Coefficient values', fontdict=font1)
        plt.legend(attribute_names_short[1:], loc='lower right', prop={'size': 12})
        #plt.title('Coefficients')
        plt.grid()
        
        plt.subplot(1, 2, 2)
        # Plot training and validation errors
        plt.loglog(lambda_interval, train_err_vs_lambda.T, 'b.-', label='Training error')
        plt.loglog(lambda_interval, test_err_vs_lambda.T, 'r.-', label='Validation error')

        # Highlight optimal lambda with green dot and value label
        plt.scatter(opt_lambda, min_error, color='green', s=100, zorder=5, 
                  label=f'Optimal λ = {opt_lambda:.2e}')
        
        plt.xlabel('Regularization factor (λ)', fontdict=font1)
        plt.xlim(min(lambda_interval), max(lambda_interval))  # Auto-scale to λ range
        plt.ylabel('Mean Squared Error', fontdict=font1)
        plt.ylim(0, 1.01)  # Slightly above 1 to prevent cutoff

        plt.legend(loc='upper left', framealpha=1, prop={'size': 14})  # Make legend visible
        plt.grid(True, linestyle='--', alpha=0.7)
        #plt.title('Error vs. Regularization Strength', fontdict=font1)
        plt.tight_layout()
        
        '''
        plt.subplot(1, 2, 2)
        plt.loglog(lambda_interval, train_err_vs_lambda.T, 'b.-', 
                  lambda_interval, test_err_vs_lambda.T, 'r.-')
        plt.ylim(0, 1.1)  # 0 to 1.0
        plt.xlabel('Regularization factor (λ)', fontdict=font1)
        plt.xlim(0,10000)
        plt.ylabel('Squared error', fontdict=font1)
        plt.ylim(0, 1)
        plt.scatter(opt_lambda, min_error, color='green', s=100, zorder=5, label='Optimal λ')
        plt.grid()
        plt.tight_layout()
        #plt.title(f'Optimal λ: {opt_lambda:.2e}', fontdict=font1)
        #plt.legend()
        #plt.legend(['Train error', 'Validation error'], loc='upper left', prop={'size': 12})
        #plt.yticks([0.1, 0.2, 0.5, 1.0], ["0.1", "0.2", "0.5", "1.0"])  # Custom ticks
        '''

# Print results
print("\n=== Regression Results ===")
print("Baseline (mean prediction):")
print(f"- Training MSE: {Error_train_nofeatures.mean():.2f}")
print(f"- Test MSE:     {Error_test_nofeatures.mean():.2f}")

print("\nRegularized linear regression:")
print(f"- Training MSE: {Error_train_rlr.mean():.2f}")
print(f"- Test MSE:     {Error_test_rlr.mean():.2f}")
print(f"- Optimal λ:    {opt_lambda:.2e}")

print("\nFeature coefficients (last fold):")
for name, coef in zip(attribute_names, w_rlr[:, -1]):
    print(f"{name:>30}: {coef:.4f}")





#%%#######################
### REGRESSION, Part b ###
##########################

K1, K2 = 10, 10  # Outer/Inner
hidden_units_range = [1,6,8,10]
#lambda_interval = np.logspace(-1, 7, 10)
lambda_interval = np.logspace(-1, 6, 50)
n_replicates = 1
max_iter = 1000

# X
attribute_names = ["year_build", 
                   "%_change_between_offer_and_purchase", 
                   "no_rooms", 
                   "sqm", 
                   #"sqm_price", 
                   "zip_code", 
                   "nom_interest_rate%", 
                   "dk_ann_infl_rate%", 
                   #"yield_on_mortgage_credit_bonds%"
                   ]

#X = df_numeric[attribute_names].to_numpy(dtype=np.float64)
#y = df_numeric["purchase_price"].to_numpy(dtype=np.float64)

X = df[attribute_names].to_numpy(dtype=np.float64)
y = df["purchase_price"].to_numpy(dtype=np.float64)

#X = df[attribute_names].to_numpy(dtype=np.float64)
#y = df["purchase_price"].to_numpy(dtype=np.float64)


# Take subset
np.random.seed(0)
idx = np.random.choice(len(X), 1000, replace=False)
X, y = X[idx], y[idx]

# Scale featues
scaler_X, scaler_y = StandardScaler(), StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1))
y = y.reshape(-1, 1)

# Store results
results = {
    'ANN': {'test_errors': [], 'best_h': []},
    'Ridge': {'test_errors': [], 'best_lambda': []},
    'Baseline': {'test_errors': []}
}

#Check:
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

#%% Outer CV
i=0
outer_cv = KFold(n_splits=K1, shuffle=True, random_state=42)

for train_idx, test_idx in outer_cv.split(X,y):
    print(f"\nOuter Fold {i+1}/{K1}")
    
    # Split and scale
    X_par, X_test = X[train_idx], X[test_idx]
    y_par, y_test = y[train_idx], y[test_idx]
    
    scaler_X = StandardScaler().fit(X_par)
    scaler_y = StandardScaler().fit(y_par)
    
    X_par_scaled = scaler_X.transform(X_par)
    y_par_scaled = scaler_y.transform(y_par)
    
    # Inner CV - Hyperparameter tuning
    inner_cv = KFold(n_splits=K2, shuffle=True, random_state=42)
    ann_val_errors = {h: [] for h in hidden_units_range}
    ridge_val_errors = {l: [] for l in lambda_interval}
    
    for inner_train_idx, val_idx in inner_cv.split(X_par_scaled):
        X_train, X_val = X_par_scaled[inner_train_idx], X_par_scaled[val_idx]
        y_train, y_val = y_par_scaled[inner_train_idx], y_par_scaled[val_idx]
        
        # ANN Training
        for h in hidden_units_range:
            model = lambda: torch.nn.Sequential(
                torch.nn.Linear(X.shape[1], h),
                #torch.nn.Tanh(),
                torch.nn.ReLU(),
                torch.nn.Linear(h, 1)
            )
            
            net, final_loss, learning_curve = train_neural_net(
                model,
                torch.nn.MSELoss(),
                X=torch.FloatTensor(X_train),
                y=torch.FloatTensor(y_train),
                n_replicates=n_replicates,
                max_iter=max_iter
            )
            
            y_pred = net(torch.FloatTensor(X_val)).detach().numpy().flatten()
            ann_val_errors[h].append(mean_squared_error(y_val, y_pred))
        
        # Ridge Training
        for l in lambda_interval:
            ridge = Ridge(alpha=l).fit(X_train, y_train)
            y_pred = ridge.predict(X_val)
            ridge_val_errors[l].append(mean_squared_error(y_val, y_pred))
    # End Inner
    # Select best hyperparameters
    best_h = min(ann_val_errors, key=lambda h: np.mean(ann_val_errors[h]))
    best_lambda = min(ridge_val_errors, key=lambda l: np.mean(ridge_val_errors[l]))
    
    # Train final models
    # ANN
    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], best_h),
        #torch.nn.Tanh(),
        torch.nn.ReLU(),
        torch.nn.Linear(best_h, 1)
    )
    net, final_loss, learning_curve = train_neural_net(
                                                      model,
                                                      torch.nn.MSELoss(),
                                                      X=torch.FloatTensor(X_par_scaled),
                                                      y=torch.FloatTensor(y_par_scaled),
                                                      n_replicates=n_replicates,
                                                      max_iter=max_iter
                                                  )
    y_pred = scaler_y.inverse_transform(
        net(torch.FloatTensor(scaler_X.transform(X_test))).detach().numpy().reshape(-1, 1))
    results['ANN']['test_errors'].append(mean_squared_error(y[test_idx], y_pred))
    results['ANN']['best_h'].append(best_h)
    
    # Ridge
    ridge = Ridge(alpha=best_lambda).fit(X_par_scaled, y_par_scaled)
    y_pred = scaler_y.inverse_transform(
        ridge.predict(scaler_X.transform(X_test)).reshape(-1, 1))
    results['Ridge']['test_errors'].append(mean_squared_error(y[test_idx], y_pred))
    results['Ridge']['best_lambda'].append(best_lambda)
    
    # Baseline
    baseline_pred = np.full_like(y[test_idx], np.mean(y_par))
    results['Baseline']['test_errors'].append(mean_squared_error(y[test_idx], baseline_pred))



#%% Visualization
plt.figure(figsize=(10, 10))
y_est = scaler_y.inverse_transform(
    net(torch.FloatTensor(scaler_X.transform(X_test))).detach().numpy())
y_true = y[test_idx]
axis_range = [min(y_true.min(), y_est.min())-1, max(y_true.max(), y_est.max())+1]
plt.plot(axis_range, axis_range, 'k--')
plt.scatter(y_true, y_est, alpha=0.8)
plt.xlabel("True Purchase Price", fontsize=20)#fontdict=font1)
plt.ylabel("Predicted Purchase Price", fontsize=20)#fontdict=font1)
plt.legend(["Model estimations","Perfect estimation"], prop={'size': 18})
plt.grid()
plt.show()

# Create DataFrame from results
df_results = pd.DataFrame({
    'Outer Fold': range(1, K1+1),
    'ANN (h*)': results['ANN']['best_h'],
    'ANN MSE': np.round(results['ANN']['test_errors'], 2),
    'Ridge (λ*)': np.round(results['Ridge']['best_lambda'], 4),
    'Ridge MSE': np.round(results['Ridge']['test_errors'], 2),
    'Baseline MSE': np.round(results['Baseline']['test_errors'], 2)
})

# Add generalization error row
generalization_errors = pd.DataFrame({
    'Outer Fold': ['Generalized'],
    'ANN (h*)': np.mean(results['ANN']['best_h']),
    'ANN MSE': np.round(np.mean(results['ANN']['test_errors']), 2),
    'Ridge (λ*)': np.round(np.mean(results['Ridge']['best_lambda']), 4),
    'Ridge MSE': np.round(np.mean(results['Ridge']['test_errors']), 2),
    'Baseline MSE': np.round(np.mean(results['Baseline']['test_errors']), 2)
})

df_final = pd.concat([df_results, generalization_errors], ignore_index=True)

# Formatting
df_final.set_index('Outer Fold', inplace=True)
df_final.index.name = None
df_final.columns.name = 'Outer fold'

# Display with improved styling
styled_table = (df_final.style
                .format({'ANN MSE': '{:.3f}', 
                        'Ridge MSE': '{:.3f}',
                        'Baseline MSE': '{:.3f}'})
                .set_caption('Two-Level Cross-Validation Results (MSE)'))

display(styled_table)


#%%#######################################
###  Statistical Comparison (SETUP II) ###
##########################################
alpha = 0.05
K = 10 # K=J
rho = 1/K
loss_in_r_func = 2
 
# Error
r_ann_vs_ridge = []
r_ann_vs_baseline = []
r_ridge_vs_baseline = []

opt_lambd = 100 # most common in CV
opt_h = 10 # most common in CV
y_true = []
y_hat = []
CV_setup_ii = KFold(n_splits=K1,shuffle=True, random_state = 43)

k=0
for train_index, test_indes in CV_setup_ii.split(X):
    print(f"\nComputing setup II CV K Fold: {k+1}/{K1}")

    X_train, y_train = X[train_index,:], y[train_index]
    X_test, y_test = X[test_index, :], y[test_index]
    
    # for ANN
    X_test_tensor = torch.tensor(X[test_index,:], dtype=torch.float)
    y_test_tensor = torch.tensor(y[test_index], dtype=torch.uint8)
    
    # Baseline
    m_baseline = np.mean(y_train)
    y_hat_baseline  = np.ones((y_test.shape[0],1))*m_baseline.squeeze()
    
    # Ridge
    m_linear = Ridge(alpha=opt_lambda).fit(X_train,y_train.squeeze())
    y_hat_linear  = m_linear.predict(X_test).reshape(-1,1)

    # ANN
    m_ANN = lambda: torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], opt_h),
        #torch.nn.Tanh(),
        torch.nn.ReLU(),
        torch.nn.Linear(opt_h, 1)
    )
    net, final_loss, learning_curve = train_neural_net(
                                                      m_ANN,
                                                      torch.nn.MSELoss(),
                                                      X=torch.FloatTensor(X_train),
                                                      y=torch.FloatTensor(y_train),
                                                      n_replicates=n_replicates,
                                                      max_iter=max_iter
                                                  )
    y_hat_ANN = net(X_test_tensor)
    y_hat_ANN = y_hat_ANN.detach().numpy()

    # True vs Pred   
    y_true.append(y_test)
    y_hat.append(np.concatenate([y_hat_baseline, y_hat_linear,y_hat_ANN], axis=1) )
            
    # r test
    # r = (M_e - y_test)**2 - (Pred - test)**2
    r_ridge_vs_baseline.append(np.mean( np.abs( y_hat_baseline-y_test ) ** loss_in_r_func - np.abs( y_hat_linear-y_test) ** loss_in_r_func))
    r_ann_vs_baseline.append(np.mean( np.abs( y_hat_baseline-y_test ) ** loss_in_r_func - np.abs( y_hat_ANN-y_test) ** loss_in_r_func))
    r_ann_vs_ridge.append(np.mean( np.abs( y_hat_ANN-y_test ) ** loss_in_r_func - np.abs( y_hat_linear-y_test) ** loss_in_r_func))

    # End loop
    k += 1
    
    
## Ridge vs baseline    
base_vs_linear, CI_setupII_base_vs_linear = correlated_ttest(r_ridge_vs_baseline, rho, alpha=alpha)

## ANN vs baseline    
base_vs_ANN, CI_setupII_base_vs_ANN = correlated_ttest(r_ann_vs_baseline, rho, alpha=alpha)

## ANN vs Ridge  
ANN_vs_linear, CI_setupII_ANN_vs_linear = correlated_ttest(r_ann_vs_ridge, rho, alpha=alpha)

## Statistics Test Table
df_statistics = pd.DataFrame(np.ones((3,5)), columns = ['H_0',
                                                        'p-value',
                                                        'CI-lower',
                                                        'CI-upper',
                                                        'Result']
                             )

df_statistics['H_0'] = ['baseline - linear =0',
                        'baseline - ANN =0',
                        'ANN - ridge =0'
                        ]

df_statistics['p-value'] = [base_vs_linear,
                            base_vs_ANN,
                            ANN_vs_linear
                            ]

df_statistics['CI-lower'] = [CI_setupII_base_vs_linear[0],
                             CI_setupII_ANN_vs_linear[0],
                             CI_setupII_base_vs_ANN[0]]

df_statistics['CI-upper'] = [CI_setupII_base_vs_linear[1],
                             CI_setupII_ANN_vs_linear[1],
                             CI_setupII_base_vs_ANN[1]]

reject_H0 = (df_statistics.loc[:,'p-value']<alpha)

df_statistics.loc[reject_H0,'Result'] = 'H_0 rejected'
df_statistics.loc[~reject_H0,'Result'] = 'failed to rejected'
df_statistics = df_statistics.set_index('H_0')

#Print
print("\n### Statistical Comparison Results Setup II ###")
styled_df = df_statistics.style.format({
    'p-value': "{:.4f}", 
    'CI-lower': "{:.4f}", 
    'CI-upper': "{:.4f}"
})
#print(styled_df.to_string())
print(df_statistics.round(4))




#%%################### 
### Classifiaction ###
######################
filename = "dimi_set_project_1.csv"
df = pd.read_csv(filename)
raw_data = df.values

cols = range(1, 19)
X = raw_data[:, cols]
attributeNames = np.array(df.columns[cols])

cont_variables = [
    "year_build", 
    "purchase_price",
    "%_change_between_offer_and_purchase",
    "no_rooms",
    "sqm",
    "sqm_price",
    "zip_code"
    ]
X = df[cont_variables].values
attributeNames = df[cont_variables].columns
y_categorical = df["house_type"].values
# ######House type
villas = np.where(y_categorical == "Villa")[0]
summerhouses= np.where(y_categorical == "Summerhouse")[0]
farms = np.where(y_categorical == "Farm")[0]
apartments = np.where(y_categorical == "Apartment")[0]
townhouses = np.where(y_categorical == "Townhouse")[0]
np.random.seed(0)
villas_idx = villas[np.random.randint(len(villas), size= 70000)]
summerhouses_idx = summerhouses[np.random.randint(len(summerhouses), size= 70000)]
farms_idx = farms[np.random.randint(len(farms), size= 70000)]
apartments_idx = apartments[np.random.randint(len(apartments), size= 70000)]
townhouses_idx = townhouses[np.random.randint(len(townhouses), size= 70000)]
indices = np.concatenate([villas_idx,summerhouses_idx, farms_idx, apartments_idx,townhouses_idx])
X = X[indices,:]
y_categorical = y_categorical[indices]

y, classNames = categoric2numeric(y_categorical)
y = np.where(y ==1)[1]

np.random.seed(0)
idx = np.random.randint(X.shape[0], size= 1000)
X = X[idx,:]
y = y[idx]
C = len(classNames)
N, M = X.shape
mode = stats.mode(y)[0]
# print(classNames)
# print(sum(y == 0))
# print(sum(y == 1))
# print(sum(y == 2))
# print(sum(y == 3))
# print(sum(y == 4))
# print(C)
print(mode)
# %%
# K-fold cross-validation
K1 = 5
K2 = 5

CV_in = model_selection.KFold(K2, shuffle=True, random_state =0)
CV_out = model_selection.KFold(K1, shuffle = True, random_state =0)

summaries, summaries_axes = plt.subplots(1, 2, figsize=(10, 5))
# color_list = ["royalblue", "darkorange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]
color_list = [
    "tab:orange",
    "tab:green",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
    "tab:red",
    "tab:blue",
]
font1 = {'family':'arial','color':'black','fontsize':16}
font2 = {'family':'arial','color':'black','fontsize':15}
font3 = {'family':'arial','color':'black','fontsize':12}


regularization_strength = np.logspace(-5, 3, 20)
n_hidden_units = [1,5,7,9]

test_error_rate_logistic = np.zeros((K1))
test_error_rate_ann = np.zeros((K1))
err_baseline = np.zeros((K1))
optimal_hidden_units = np.zeros((K1))
i = 0
yhat = []
y_true = []
r_ann_vs_baseline = []
r_lr_vs_baseline = []
r_ann_vs_lr = []
for outer_train_index, outer_test_index in CV_out.split(X, y):
    print("\nOuter crossvalidation fold: {0}/{1}".format(i + 1, K1))
    # print('Outer Train indices: {0}'.format(outer_train_index))
    # print('Outer Test indices: {0}\n'.format(outer_test_index))
    X_par = X[outer_train_index]
    y_par = y[outer_train_index]
    X_test = X[outer_test_index]
    y_test = y[outer_test_index]

    val_error_rate_logistic = np.zeros((K2,len(regularization_strength)))
    val_error_rate_ann = np.zeros((K2,len(n_hidden_units)))
    train_error_rate_logistic = np.empty((K2, len(regularization_strength)))
    train_error_rate_ann = np.empty((K2, len(n_hidden_units)))
    j = 0
    for train_index, test_index in CV_in.split(X_par, y_par):
        print("\nInner crossvalidation fold: {0}/{1}".format(j + 1, K2))
        # print('Inner Train indices: {0}'.format(train_index))
        # print('Inner Test indices: {0}\n'.format(test_index))
        X_train = X[train_index,:]
        y_train = y[train_index]
        X_val = X[test_index,:]
        y_val = y[test_index]

        mu = np.mean(X_train, 0)
        sigma = np.std(X_train, 0)
        X_train = (X_train - mu) / sigma
        X_val = (X_val - mu) / sigma

        for l in range(0,len(regularization_strength)):
            mdl = lm.LogisticRegression(
                solver="lbfgs",
                # multi_class="multinomial",
                tol=1e-4,
                random_state=1,
                penalty="l2",
                C=1 / regularization_strength[l],
                max_iter =1000
            )
            mdl.fit(X_train, y_train)
            
            y_train_est_logistic = mdl.predict(X_train)
            y_val_est_logistic = mdl.predict(X_val)
            train_error_rate_logistic[j,l] = np.sum(y_train_est_logistic != y_train) / len(y_train)
            val_error_rate_logistic[j,l] = np.sum(y_val_est_logistic != y_val) / len(y_val)
        
        
        for s in range(len(n_hidden_units)) :
            model = lambda: torch.nn.Sequential(
                torch.nn.Linear(M, n_hidden_units[s]),  
                torch.nn.ReLU(),  
                torch.nn.Linear(n_hidden_units[s], C),
                torch.nn.Softmax(dim=1))
            loss_fn = torch.nn.CrossEntropyLoss()
            net, final_loss, learning_curve = train_neural_net(model, loss_fn,
                X=torch.tensor(X_train, dtype=torch.float),
                y=torch.tensor(y_train, dtype=torch.long),
                n_replicates=3,
                max_iter=10000)
            softmax_logits = net(torch.tensor(X_val, dtype=torch.float))
            y_val_est_ann = (torch.max(softmax_logits, dim=1)[1]).data.numpy()
            e = y_val_est_ann != y_val
            # print("Number of miss-classifications for ANN:\n\t {0} out of {1}".format(sum(e), len(e)))
            val_error_rate_ann[j,s] = np.sum(e)/len(y_val)
        j +=1
    train_error_model_logistic = np.mean(train_error_rate_logistic, axis = 0)
    gen_error_model_logistic = np.mean(val_error_rate_logistic,axis =0)       
    min_error_logistic = np.min(gen_error_model_logistic)
    opt_lambda_idx = np.argmin(gen_error_model_logistic)
    opt_lambda = regularization_strength[opt_lambda_idx]

    if i == K1-1:
        plt.figure(figsize=(8, 8))
        plt.semilogx(regularization_strength, train_error_model_logistic * 100)
        plt.semilogx(regularization_strength, gen_error_model_logistic * 100)
        plt.semilogx(opt_lambda, min_error_logistic * 100, "o")
        plt.text( 1e-8, 3,"Minimum test error: "+ str(np.round(min_error_logistic * 100, 2)) + " % at 1e" + str(np.round(np.log10(opt_lambda), 2)),fontdict=font1)
        plt.xlabel("Regularization strength, $\log_{10}(\lambda)$", font1)
        plt.ylabel("Error rate (%)",font1)
        # plt.title("Classification error, last outer fold, logistic regression")
        plt.legend(["Training error", "Test error", "Test minimum"], loc="upper right")
        # plt.ylim([0, 4])
        plt.grid()
        plt.show()

 
    mu = np.mean(X_par, 0)
    sigma = np.std(X_par, 0)
    X_par = (X_par - mu) / sigma
    X_test = (X_test - mu) / sigma

    mdl = lm.LogisticRegression(solver="lbfgs", 
                # multi_class="multinomial", 
                tol=1e-4, random_state=1,
                penalty="l2",
                C=1 / opt_lambda,
                max_iter =1000
            )
    mdl.fit(X_par, y_par)
    y_test_est_logistic = mdl.predict(X_test)
    err_logistic = y_test_est_logistic != y_test
    test_error_rate_logistic[i] = np.sum(err_logistic) / len(y_test)

    gen_error_model_ann = np.mean(val_error_rate_ann, axis = 0)
    min_error = np.min(gen_error_model_ann)
    best_h_idx = np.argmin(gen_error_model_ann)
    best_h = n_hidden_units[best_h_idx] 
    optimal_hidden_units[i] = best_h
    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, best_h),  
        torch.nn.ReLU(),  
        torch.nn.Linear(best_h, C),
        torch.nn.Softmax(dim=1),  # final tranfer function
    )
    net, final_loss, learning_curve = train_neural_net(
        model,
        loss_fn,
        X=torch.tensor(X_par, dtype=torch.float),
        y=torch.tensor(y_par, dtype=torch.long),
        n_replicates=3,
        max_iter=10000,
    )

   

    softmax_logits2 = net(torch.tensor(X_test, dtype=torch.float))
    y_test_est_ann = (torch.max(softmax_logits2, dim=1)[1]).data.numpy()
    err_ann = y_test_est_ann != y_test
    test_error_rate_ann[i] = (sum(err_ann)/len(y_test))

    if i == K1-1:
        plt.figure(figsize=(9, 9))
        axis_range = [np.min([y_test_est_ann, y_test]) - 1, np.max([y_test_est_ann, y_test]) + 1]
        plt.plot(axis_range, axis_range, "k--")
        plt.plot(y_test, y_test_est_ann, "ob", alpha=0.1)
        plt.legend(["Perfect estimation", "Model estimations"])
        # plt.title("House type: estimated versus true value (for last CV-fold)", font1)
        plt.ylim(axis_range)
        plt.xlim(axis_range)
        plt.xlabel("True value", font1)
        plt.xticks(range(C), labels=classNames)
        plt.ylabel("Estimated value", font1)
        plt.yticks(range(C), labels=classNames)
        plt.grid()
        plt.show()
    
    baseline = mode*np.ones(shape=y_test.shape)
    base_err = y_test != baseline
    err_baseline[i] = np.sum(base_err)/len(y_test)
    dy = []
    dy.append([y_test_est_logistic,y_test_est_ann,baseline])
    dy = np.stack(dy,axis =0)
    yhat.append(dy)
    y_true.append([y_test])
    r_ann_vs_baseline.append(np.mean(test_error_rate_ann-err_baseline))
    r_lr_vs_baseline.append(np.mean(test_error_rate_logistic-err_baseline))
    r_ann_vs_lr.append(np.mean(test_error_rate_ann-test_error_rate_logistic))
    i += 1



print(f"optimal hs: {optimal_hidden_units}")
yhat = np.concatenate(yhat)
y_true = np.concatenate(y_true)
print(f"Test error ANN: {test_error_rate_ann}")
print(f"Test error LR: {test_error_rate_logistic}")
print(f"Test error baseline: {err_baseline}")

# %%
# print(np.mean(err_ann-base_err))
print(test_error_rate_ann)
print(test_error_rate_logistic)
print(err_baseline)
print(r_ann_vs_baseline)
print(r_ann_vs_lr)
print(r_lr_vs_baseline)

alpha = 0.05
rho = 1/K1
p_setupII_ann_vs_bs, CI_setupII_ann_vs_bs = correlated_ttest(r_ann_vs_baseline, rho, alpha=alpha)
p_setupII_lr_vs_bs, CI_setupII_lr_vs_bs = correlated_ttest(r_lr_vs_baseline, rho, alpha=alpha)
p_setupII_ann_vs_lr, CI_setupII_ann_vs_lr = correlated_ttest(r_ann_vs_lr, rho, alpha=alpha)

print(f"Stats setup II: {p_setupII_ann_vs_bs}, {CI_setupII_ann_vs_bs}" )
print(f"Stats setup II: {p_setupII_ann_vs_lr}, {CI_setupII_ann_vs_lr}" )
print(f"Stats setup II: {p_setupII_lr_vs_bs}, {CI_setupII_lr_vs_bs}" )
# %%

plotproba = np.zeros((C,C))

for idx  in range(len(y_test_est_ann)):
    plotproba[y_test_est_ann[idx],y_test[idx] ] +=1 
plt.figure(figsize=(7.5,6))
# cmap = sns.diverging_palette(220, 0, as_cmap=True)
sns.heatmap(plotproba, annot=True,vmin = 0, vmax = 60, cmap="Blues", cbar_kws={"shrink": .5})
plt.xlabel("Predicted value")
# plt.ylabel("True value")
plt.xticks(np.arange(C)+0.5, labels=classNames, rotation = 45)
# plt.yticks(np.arange(C)+0.5, labels=classNames, rotation = 45)
plt.tick_params(left=False)
plt.yticks([])
plt.show()


for idx  in range(len(y_test_est_logistic)):
    plotproba[y_test_est_logistic[idx],y_test[idx] ] +=1 
plt.figure(figsize=(6,6))
# cmap = sns.diverging_palette(220, 0, as_cmap=True)
sns.heatmap(plotproba, annot=True,vmin = 0, vmax = 60, cmap="Blues", cbar=False)
plt.xticks(np.arange(C)+0.5, labels=classNames, rotation = 45)
plt.yticks(np.arange(C)+0.5, labels=classNames, rotation = 45)
plt.xlabel("Predicted value")
plt.ylabel("True value")
plt.show()



