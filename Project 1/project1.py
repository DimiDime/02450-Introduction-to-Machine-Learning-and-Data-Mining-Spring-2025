# Colleciton of python code used for the analysis and conclusion of 02450 Project 1
# s236115 - Dimitar Ilev
# The code is in order of implementation
#%% Imports 
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
import os
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px

#%% Dataset review
df = pd.read_parquet("path_to_df")
df.info()
df['date'] = pd.to_datetime(df['date']) # convert date
df.describe()
# Check out NaN
df[df['sqm_price'].isnull()]
df = df[df['sqm'].notna()]
missing_rows = df[df['dk_ann_infl_rate%'].isna() & df['yield_on_mortgage_credit_bonds%'].isna()]
min_date = missing_rows['date'].min()
max_date = missing_rows['date'].max()
print("Minimum date:", min_date)
print("Maximum date:", max_date)

# Impute missing values
df['dk_ann_infl_rate%'] = df['dk_ann_infl_rate%'].fillna(1.6)
df['yield_on_mortgage_credit_bonds%'] = df['yield_on_mortgage_credit_bonds%'].fillna(4.1)
df.isnull().sum()
df["sales_type"] = df["sales_type"].replace("-", "other_sale")

#%% Distribution & outliers
numeric_columns = ["year_build", "purchase_price", "%_change_between_offer_and_purchase", 
                   "no_rooms", "sqm", "sqm_price", "zip_code", 
                   "nom_interest_rate%", "dk_ann_infl_rate%", "yield_on_mortgage_credit_bonds%"]
# Personalization
colors = ["royalblue", "darkorange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]
font1 = {'family':'arial','color':'black','fontsize':16}
font2 = {'family':'arial','color':'dimgray','fontsize':15}
font3 = {'family':'arial','color':'dimgray','fontsize':12}

histograms = {}
for i,col in enumerate(numeric_columns):
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], bins=120, kde=True, color=colors[i % len(colors)])
    mean_value = df[col].mean()
    mean_str = f"{mean_value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")  # Format as 100.000,00
    plt.axvline(mean_value, color="black", linestyle="dashed", linewidth=1.5, label=f"Mean: {mean_str}")
    #plt.title(f"Distribution of {col}", fontdict = font1)
    plt.xlabel('', fontdict = font3)
    plt.ylabel("", fontdict = font2)
    plt.grid(True, linestyle="-", alpha=0.6, linewidth=0.5)
    plt.tight_layout()
    legend_loc = "upper left" if col == "year_build" else "upper right"
    plt.legend(loc=legend_loc, fontsize=18, frameon=True)
    plt.savefig(os.path.join('img', f"{col}_distribution.png"), dpi=300, bbox_inches="tight")
    plt.show()
      

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(16, 25))
axes = axes.flatten()
for i, col in enumerate(numeric_columns):
    ax = axes[i]
    sns.histplot(df[col], bins=120, kde=True, color=colors[i % len(colors)], ax=ax)
    mean_value = df[col].mean()
    mean_str = f"{mean_value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")  
    ax.axvline(mean_value, color="black", linestyle="dashed", linewidth=1.5, label=f"Mean: {mean_str}")
    ax.set_title(f"Distribution of {col}", fontdict=font1)
    ax.set_xlabel(col, fontdict=font3)
    ax.set_ylabel("Frequency", fontdict=font2)
    ax.grid(True, linestyle="-", alpha=0.6, linewidth=0.5)
    legend_loc = "upper left" if col == "year_build" else "upper right"
    ax.legend(loc=legend_loc, fontsize=10, frameon=True)
plt.tight_layout()
plt.show()

n_cols = 5
n_rows = 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
axes = axes.flatten()
for i, column in enumerate(numeric_columns):
    sns.boxplot(data=df, y=column, ax=axes[i], color=colors[i], width=0.1, 
                linewidth=0.5, whiskerprops=dict(color='red'), 
                capprops=dict(color='green'), medianprops=dict(color='white'), 
                flierprops=dict(markerfacecolor='red', marker='o'))
    axes[i].set_title(column, fontdict=font1)
    axes[i].set_xlabel('', fontdict=font3)
    #axes[i].set_ylabel('Value', fontdict=font2)
    axes[i].tick_params(axis='both', which='major', labelsize=10)
plt.tight_layout()
plt.show()


#%%Regions
nominal_column = ["region"]
for col in nominal_column:
    plt.figure(figsize=(10, 5))
    ax = sns.countplot(y=df[col], order=df[col].value_counts().index, palette="pastel") 
    for p in ax.patches:
        count = p.get_width() 
        plt.text(count + 5000,
                 p.get_y() + p.get_height() / 2,
                 f"{int(count):,}".replace(",", "."),
                 ha="left",
                 va="center",
                 fontsize=10)
    plt.xticks(np.arange(0, 800001, 50000),
           [f'{x:,.0f}'.replace(',', '.') for x in np.arange(0, 800001, 50000)],
           rotation=45)
    plt.xlim(0, 850000) 
    plt.title(f"Number of Sales per Region", fontdict=font1, pad=20)
    plt.xlabel("# Sales", fontdict=font3)
    plt.ylabel("Region", fontdict=font2)
    plt.grid(axis='x', linestyle="--", alpha=0.3)
    plt.show()

#%% Correlation
new_names = {
    '%_change_between_offer_and_purchase': '%_change',
    'yield_on_mortgage_credit_bonds%': 'yield_%'
}
try:
    df_numeric = df[numeric_columns]
except KeyError as e:
    print(f"Error: One or more columns in numeric_columns not found in DataFrame. Check your column names.")
    print(f"Missing columns: {e}")
    raise
correlation_matrix = df_numeric.corr(method='spearman')
correlation_matrix = correlation_matrix.rename(columns=new_names, index=new_names)
cmap = sns.diverging_palette(220, 10, as_cmap=True)
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap=cmap, linewidths=.5, fmt=".2f", cbar_kws={"shrink": .5})
#plt.title("Spearman Correlation Matrix", fontdict=font1)
plt.show()

#PCA
scaler = StandardScaler()  
df_scaled = scaler.fit_transform(df_numeric)
df_scaled = pd.DataFrame(df_scaled, columns=df_numeric.columns)
pca = PCA()
pca_result = pca.fit_transform(df_scaled)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

threshold = 0.9
threshold95 = 0.95
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, "x-")
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, "o-")
plt.plot([1, len(explained_variance_ratio)], [threshold, threshold], "k--")
plt.plot([1, len(explained_variance_ratio)], [threshold95, threshold95], "--", color="green")
#plt.title("Explained Variance PCs")
plt.xlabel("Principal component", fontdict=font2)
plt.ylabel("Variance explained", fontdict=font2)
plt.legend(["Individual", "Cumulative", "Threshold 90%", "Treshhold 95%"], loc="center right")
plt.grid(True)
plt.xticks(np.arange(1, len(explained_variance_ratio) + 1, step=1))
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.ylim(0, 1.1)
plt.tight_layout()
plt.show()

loadings = pd.DataFrame(pca.components_.T, index=df_numeric.columns, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
print("\nLoadings:")
print(loadings)
loadings = loadings.rename(index=new_names)
plt.figure(figsize=(12, 8))
sns.heatmap(loadings.iloc[:, :10], annot=True, cmap='coolwarm', center=0, cbar_kws={"shrink": .5})
#plt.title("Loadings of attributes on PCs")
plt.show()

n_components = 7
pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(df_scaled)
df_pca = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
df_pca['purchase_price'] = df_scaled['purchase_price']

fig = px.scatter_matrix(
    df_pca, 
    dimensions=[f'PC{i+1}' for i in range(n_components)], 
    color='purchase_price',
    color_continuous_scale='viridis',
    hover_data=['purchase_price']
)

fig.update_traces(marker=dict(size=4))
fig.update_layout(
    width=1400,
    height=1000,
    legend=dict(orientation="h", y=-0.2),
    coloraxis_colorbar=dict(title='Purchase Price')
)

fig.show()