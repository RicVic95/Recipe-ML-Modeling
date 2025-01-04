import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, PrecisionRecallDisplay
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingClassifier

# ------------------------------------------------------------------
# Random Forests with base data 

# Read in data
recipes = pd.read_pickle('../../data/processed/recipes_site_traffic_clean.pkl')

# Features and Target 
X = recipes.drop(['high_traffic', 'high_traffic_bool'], axis=1)    
y = recipes['high_traffic']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Separate numerical and categorical features
categorical_columns = ['category']
numerical_columns = [
    'servings',
    'calories',
    'carbohydrate',
    'sugar',
    'protein'
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns), 
        ('num', StandardScaler(), numerical_columns)
    ]
)

# Instantiate Decision Tree Model (base params)
rf = RandomForestClassifier(
    random_state=42,
    class_weight='balanced'
)

# Create a Pipeline 
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', rf)
])

# Hyperparameter Tunning
param_grid_rf = {
    'model__n_estimators': [100, 200, 500],  
    'model__max_depth': [None, 10, 20, 30],  
    'model__min_samples_split': [2, 5, 10],  
    'model__min_samples_leaf': [1, 2, 4],   
    'model__max_features': ['sqrt', 'log2'], 
    'model__bootstrap': [True, False]
}

# GridSearchCV setup
grid_rf = GridSearchCV(
    estimator=pipeline, 
    param_grid=param_grid_rf, 
    cv=3, 
    n_jobs=-1, 
    verbose=1
)

# Assuming 'X_train' and 'y_train' are already defined
grid_rf.fit(X_train, y_train)

# Print the best parameters
print('RESULTS FOR RANDOM FOREST WITH ORIGINAL DATA')

# Evaluate the model on the test set
y_pred_rf = grid_rf.predict(X_test)
y_pred_proba = grid_rf.predict_proba(X_test)[:,1]
print(classification_report(y_test, y_pred_rf))
print('Best parameters found: ', grid_rf.best_params_)
print('\nROC AUC SCORE: ', roc_auc_score(y_test, y_pred_proba))

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_rf)

# Compute ROC curve and ROC AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label='Other')
roc_auc = auc(fpr, tpr)

# Set up the matplotlib figure with two subplots on the same row
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

fig.suptitle('Model Evaluation: Confusion Matrix and ROC Curve - RF with Original Data', fontsize=16)

# Plot the confusion matrix
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['High', 'Other']).plot(cmap='Blues', ax=axes[0])
axes[0].set_title('Confusion Matrix')

# Plot the ROC curve
axes[1].plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
axes[1].plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('Receiver Operating Characteristic (ROC) Curve')
axes[1].legend(loc="lower right")
plt.savefig('../../reports/figures/RF_OD_evaluation.png', bbox_inches='tight')
plt.show()

# Access feature importances
importances = grid_rf.best_estimator_.named_steps['model'].feature_importances_

# Get feature names from the preprocessor
categorical_features = grid_rf.best_estimator_.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out()
numerical_features = numerical_columns  # Use the numerical columns from your data

# Combine feature names
all_feature_names = list(categorical_features) + list(numerical_features)

feature_importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
print('Top 10 Features by Importance (Original Data)')
print(feature_importance_df.head(10))

print('-------------------------------------------------------------\n')

# ------------------------------------------------------------------
# Random Forest with Engineered Features

# Read in data
recipes_engineered = pd.read_pickle('../../data/processed/recipes_engineered.pkl')

# Features and Target 
X = recipes_engineered.drop(['high_traffic', 'high_traffic_bool'], axis=1)    
y = recipes_engineered['high_traffic']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Separate numerical and categorical features
categorical_columns = ['category', 'nutritional_cluster']
numerical_columns = [
    'calories',
    'carbohydrate',
    'sugar',
    'protein',
    'servings',
    'protein_calorie_ratio',
    'sugar_carb_ratio',
    'carb_protein_ratio',
    'avg_category_calories',
    'high_traffic_ratio_by_clust',
    'high_traffic_ratio_by_cat'
 ]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns), 
        ('num', 'passthrough', numerical_columns)
    ]
)

# Create a Pipeline 
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', rf)
])

# Hyperparameter Tunning
param_grid_rf = {
    'model__n_estimators': [100, 200, 500],  
    'model__max_depth': [None, 10, 20, 30],  
    'model__min_samples_split': [2, 5, 10],  
    'model__min_samples_leaf': [1, 2, 4],   
    'model__max_features': ['sqrt', 'log2'], 
    'model__bootstrap': [True, False], 
}

# GridSearchCV setup
grid_rf = GridSearchCV(
    estimator=pipeline, 
    param_grid=param_grid_rf, 
    cv=3, 
    n_jobs=-1, 
    verbose=1
)

# Assuming 'X_train' and 'y_train' are already defined
grid_rf.fit(X_train, y_train)

# Print the best parameters
print('RESULTS FOR RANDOM FOREST WITH ENGINEERED FEATURES')

# Evaluate the model on the test set
y_pred_rf = grid_rf.predict(X_test)
y_pred_proba = grid_rf.predict_proba(X_test)[:,1]
print(classification_report(y_test, y_pred_rf))
print('\nROC AUC SCORE: ', roc_auc_score(y_test, y_pred_proba))
print('Best parameters found: ', grid_rf.best_params_)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_rf)

# Compute ROC curve and ROC AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label='Other')
roc_auc = auc(fpr, tpr)

# Set up the matplotlib figure with two subplots on the same row
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

fig.suptitle('Model Evaluation: Confusion Matrix and ROC Curve - RF with Engineered Data', fontsize=16)

# Plot the confusion matrix
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['High', 'Other']).plot(cmap='Blues', ax=axes[0])
axes[0].set_title('Confusion Matrix')

# Plot the ROC curve
axes[1].plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
axes[1].plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('Receiver Operating Characteristic (ROC) Curve')
axes[1].legend(loc="lower right")
plt.savefig('../../reports/figures/RF_ED_evaluation.png', bbox_inches='tight')
plt.show()

# Access feature importances
importances = grid_rf.best_estimator_.named_steps['model'].feature_importances_

# Get feature names from the preprocessor
categorical_features = grid_rf.best_estimator_.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out()
numerical_features = numerical_columns  # Use the numerical columns from your data

# Combine feature names
all_feature_names = list(categorical_features) + list(numerical_features)

feature_importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
print('Top 10 Features by Importance (Original Data)')
print(feature_importance_df.head(10))

