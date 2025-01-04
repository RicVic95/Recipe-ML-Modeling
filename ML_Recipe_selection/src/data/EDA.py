import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Read in the data 
recipes = pd.read_pickle('../../data/processed/recipes_site_traffic_clean.pkl')
nutritional_columns = ['calories', 'carbohydrate', 'sugar', 'protein']

# ----------------------------- # 
# Exploratory Data Analysis     # 
# ----------------------------- # 

recipes['high_traffic'].value_counts()

# Distribution of Categories
traffic_by_cat = recipes.groupby('category', observed=False)['high_traffic_bool'].value_counts().unstack()
traffic_by_cat['total_recipe_count'] = recipes.groupby('category', observed=False).size()
traffic_by_cat['high_traffic_ratio_by_cat'] = recipes.groupby('category', observed=False)['high_traffic_bool'].sum() / recipes.groupby('category', observed=False).size()
traffic_by_cat.to_csv('../../reports/tables/Traffic_by_cat_summary.csv')

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
sns.barplot( # Barplot 1: Distribution of Recipes Across Categories
    data=traffic_by_cat.sort_values(by='total_recipe_count', ascending=False), 
    x='total_recipe_count',
    y='category',
    order=traffic_by_cat.sort_values(by='total_recipe_count', ascending=False).index,
    ax=axes[0], # Left Plot
    hue='category',
    legend=False 
)
axes[0].set_title('Distribution of Recipes Across Categories')
axes[0].set_xlabel('Number of Recipes')
axes[0].set_ylabel('Category')


# Format x-axis of the second plot to show percentage
def percent(x, pos):
    return f'{x*100:.0f}%'

sns.barplot( # Barplot 2: High Traffic Ratio by Category
    data=traffic_by_cat.sort_values(by='total_recipe_count', ascending=False), 
    x='high_traffic_ratio_by_cat',
    y='category',
    order=traffic_by_cat.sort_values(by='total_recipe_count', ascending=False).index,
    ax=axes[1], # Right Plot
    hue='category',
    legend=False 
)
axes[1].set_title('High Traffic Ratio By Category')
axes[1].set_xlabel('Percentage of High Traffic Recipes')
axes[1].set_ylabel('Category')
axes[1].xaxis.set_major_formatter(FuncFormatter(percent))
plt.tight_layout()
plt.savefig('../../reports/figures/category_summary.png', bbox_inches='tight')

# ----------------------------- # 
# Box-cox Transformation        # 
# ----------------------------- # 
from scipy.stats import boxcox

# Plot distributions for nutritional features
fig, axes = plt.subplots(2, 2, figsize=(12, 10)) 
for i, column in enumerate(nutritional_columns):
    row = i // 2  
    col = i % 2   
    sns.histplot(data=recipes, x=column, kde=True, ax=axes[row, col])  
    axes[row, col].set_title(f'Distribution of {column.title()}')  
plt.tight_layout() 
plt.savefig('../../reports/figures/nutritional_features_distributions.png', bbox_inches='tight')

# Check if there are values = 0 in the nutritional columns: 
for col in nutritional_columns: 
    if len(recipes[recipes[col]==0] == 0) == 0: 
        print(f"'{col.title()}' column does not contain any values with 0")
    else: 
        print(f"'{col.title()}' contains values equal to 0") 
        
print('-----------------------\n')        
        
# Check for non-negative values
for col in nutritional_columns: 
    if len(recipes[recipes[col] < 0]) == 0: 
        print(f"'{col.title()}' contains only positive values")
    else: 
        print(f"'{col.title()}' contains negative values")       
        
print('-----------------------\n')              

# Apply Box-Cox transformation to each column
for col in nutritional_columns:
    recipes[col] = recipes[col] + 1  # Add 1  to handle zeros
    
    # Apply Box-Cox transformation
    recipes[f'{col}_boxcox'], lambda_val = boxcox(recipes[col])
    
    # Print lambda for each column
    print(f"Optimal λ for '{col.title()}': {lambda_val}") 
    
# Plot distributions for nutritional features (after box-cox transformation)
nutritional_columns_boxcox = ['calories_boxcox','carbohydrate_boxcox', 'sugar_boxcox', 'protein_boxcox']      

fig, axes = plt.subplots(2, 2, figsize=(12, 10)) 
for i, column in enumerate(nutritional_columns_boxcox):
    row = i // 2  
    col = i % 2   
    sns.histplot(data=recipes, x=column, kde=True, ax=axes[row, col], bins=25)  
    axes[row, col].set_title(f'Distribution of {column.replace("_boxcox", "").title()} After Box-cox Transformation')  
plt.tight_layout() 
plt.savefig('../../reports/figures/BC_nutritional_features_distributions.png', bbox_inches='tight')

# Correlation between old and new variables
correlation = pd.DataFrame(recipes.corr(numeric_only=True)['high_traffic_bool']).sort_values(by='high_traffic_bool').drop('high_traffic_bool', axis=0)
correlation.plot(kind='barh')
plt.title('Correlation Between Variables and Target Label')
plt.xlabel('Feature')
plt.ylabel('Correlation Coefficient')
plt.legend().remove()
plt.savefig('../../reports/figures/correlation_barplot.png', bbox_inches='tight')

# Save Dataset 
recipes.to_pickle('../../data/processed/recipes_box_cox.pkl')
recipes.info()