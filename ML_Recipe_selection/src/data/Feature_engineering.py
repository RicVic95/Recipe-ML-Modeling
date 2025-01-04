import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Read in the data 
recipes_boxcox = pd.read_pickle('../../data/processed/recipes_box_cox.pkl')
recipes_boxcox.info()

box_cox_columns = ['calories_boxcox', 'carbohydrate_boxcox', 'sugar_boxcox', 'protein_boxcox']

model_data = recipes_boxcox[box_cox_columns]

# Use K-means clustering to group the data based on their nutritional properties (after box_cox transformation)
scaler = StandardScaler()
scaled_model_data = scaler.fit_transform(model_data)

# Determine the correct number of clusters
inertia = []

for k in range(1,11): 
    kmeans = KMeans(n_clusters=k, random_state=42,n_init=10)
    kmeans.fit(scaled_model_data)
    inertia.append(kmeans.inertia_)
    
# Plot the elbow curve
# Plot the elbow curve
plt.figure(figsize=(10,8))    
plt.plot(range(1,11), inertia, marker='o', label='Inertia')
plt.axvline(x=5, color='red', linestyle='--', label='Optimal k (k=5)')  # Add vertical line at k=5
plt.title('Elbow Method for Optimal K')
plt.xlabel('n_clusters')
plt.ylabel('Inertias')
plt.legend()  # Add legend for better interpretation
plt.savefig('../../reports/figures/elbow_method_optimal_k.png')
plt.show();

# Select k=5 for clustering 
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
recipes_boxcox['nutritional_cluster'] = kmeans.fit_predict(scaled_model_data)

# Summarize nutritional features by cluster
cluster_summary = recipes_boxcox.groupby('nutritional_cluster')[['calories','carbohydrate','sugar','protein']].mean().round(2)
cluster_summary.to_csv('../../reports/tables/cluster_summary.csv')

# Traffic ratio across clusters 
traffic_by_clust = recipes_boxcox.groupby('nutritional_cluster')['high_traffic_bool'].value_counts().unstack()
traffic_by_clust['high_traffic_ratio_by_clust'] = recipes_boxcox.groupby('nutritional_cluster')['high_traffic_bool'].sum() / recipes_boxcox.groupby('nutritional_cluster')['high_traffic_bool'].size()

# Use the traffic ratio by cluster table as a feature for the dataset
recipes_boxcox = recipes_boxcox.merge(traffic_by_clust['high_traffic_ratio_by_clust'], on='nutritional_cluster', how='left')

# Traffic ratio across categories 
traffic_by_cat = recipes_boxcox.groupby('category', observed=False)['high_traffic_bool'].value_counts().unstack()
traffic_by_cat['high_traffic_ratio_by_cat'] = recipes_boxcox.groupby('category',observed=False)['high_traffic_bool'].sum() / recipes_boxcox.groupby('category',observed=False)['high_traffic_bool'].size()

# Use the traffic ratio by category table as a feature for the dataset
recipes_boxcox = recipes_boxcox.merge(traffic_by_cat['high_traffic_ratio_by_cat'], on='category', how='left')

# Interaction Features between Nutritional Components
recipes_boxcox['protein_calorie_ratio'] = recipes_boxcox['protein'] / recipes_boxcox['calories']
recipes_boxcox['sugar_carb_ratio'] = recipes_boxcox['sugar'] / recipes_boxcox['carbohydrate']
recipes_boxcox['carb_protein_ratio'] = recipes_boxcox['carbohydrate'] / recipes_boxcox['protein']

# Category Popularity Features
category_calories = recipes_boxcox.groupby('category', observed=False)['calories'].mean()
recipes_boxcox['avg_category_calories'] = recipes_boxcox['category'].map(category_calories)

# New features and Correlation
correlation_results = (
    recipes_boxcox.drop(
        [
            'calories', 
            'carbohydrate', 
            'sugar', 
            'protein', 
            'servings',
            'calories_boxcox', 
            'carbohydrate_boxcox', 
            'sugar_boxcox',
            'protein_boxcox'
        ], 
        axis=1
    )
    .corr(numeric_only=True)
    .loc['high_traffic_bool']
    .sort_values(ascending=False)
    .drop('high_traffic_bool', axis=0)
)

sns.barplot(x=correlation_results, y=correlation_results.index, legend=False)
plt.title('Correlations of Engineered Features with Target Label')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Engineered Features') 
plt.savefig('../../reports/figures/corr_engineered_features.png', bbox_inches='tight')
plt.show();

# Save the data for 
recipes_boxcox.drop(['calories', 'carbohydrate', 'sugar', 'protein'], axis=1).to_pickle('../../data/processed/recipes_boxcox_engineered.pkl')
recipes_boxcox.drop(['calories_boxcox', 'carbohydrate_boxcox', 'sugar_boxcox', 'protein_boxcox'], axis=1).to_pickle('../../data/processed/recipes_engineered.pkl')
