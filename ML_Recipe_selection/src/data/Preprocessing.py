import numpy as np 
import pandas as pd 
import seaborn as sns

# Read in the data 
recipes_raw = pd.read_csv('../../data/raw/recipe_site_traffic_2212.csv')
print('Raw Data Info: \n')
recipes_raw.info()

# Check for percentage of missing values
recipes_raw.isnull().sum()/len(recipes_raw)

# ------------------------------------ #
# Data Validation                      #
# ------------------------------------ #

# Check recipe is a unique identifier 
recipes_raw['recipe'].nunique()

# Confirm that categories are correct 
recipes_raw['category'].unique()

# Replace incorrect categories
recipes_raw['category'] = recipes_raw['category'].replace('Chicken Breast', 'Chicken')

# Adjust servings column to display values as intended
recipes_raw['servings'] = recipes_raw['servings'].str.replace(' as a snack','', regex=False).astype('int')

# Evaluate missing values per category
df_missing_perc = recipes_raw.groupby('category').agg(lambda cat: cat.isnull().mean())[['calories', 'carbohydrate', 'sugar', 'protein']]

# Assumption: Missing values on 'high_traffic' column represent instances where the recipe is not high traffic
# Create 'high_traffic_bool' column to display Boolean Values for High_traffic
recipes_raw['high_traffic_bool'] = recipes_raw['high_traffic'].apply(lambda x: False if pd.isnull(x) else True)

# Adjust 'high_traffic' column to display 'Other' for NaN values 
recipes_raw['high_traffic'] = recipes_raw['high_traffic'].fillna('Other')

# Drop rows with null values 
recipes_raw.dropna(inplace=True)

# Make sure columns have appropriate categories 
recipes_raw['category'] = recipes_raw['category'].astype('category')

# Confirm there are no missing values and that column types have the appropriate categories
print('------------------\n','Info after Validation and Cleaning \n')
print(recipes_raw.info())

# Drop Recipe Column 
recipes_raw = recipes_raw.drop('recipe', axis=1)

# Save the data
recipes_raw.to_pickle('../../data/processed/recipes_site_traffic_clean.pkl')






