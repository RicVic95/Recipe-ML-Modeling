import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ------------------------------------------------------------------
# Gradient Boosting with base data

# Read in data
recipes = pd.read_pickle('../../data/processed/recipes_site_traffic_clean.pkl')

# Features and Target
X = recipes.drop(['high_traffic', 'high_traffic_bool'], axis=1)
y = recipes['high_traffic']

# Train-test split
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

# Instantiate Gradient Boosting Classifier
gb = GradientBoostingClassifier(random_state=42)

# Create a Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', gb)
])

# Hyperparameter Tuning
param_grid = {
    'model__n_estimators': [100, 200, 500],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__max_depth': [3, 6, 10]
}

# GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)

# Fit the model
grid_search.fit(X_train, y_train)

# Evaluate
y_pred = grid_search.predict(X_test)
y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
print('\nRESULTS FOR GRADIENT BOOSTING WITH BASE DATA')
print(classification_report(y_test, y_pred))
print('ROC AUC SCORE: ', roc_auc_score(y_test, y_pred_proba))
print("Best parameters found: ", grid_search.best_params_)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Compute ROC curve and ROC AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label='Other')
roc_auc = auc(fpr, tpr)

# Set up the matplotlib figure with two subplots on the same row
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

fig.suptitle('Model Evaluation: Confusion Matrix and ROC Curve - GB with Original Data', fontsize=16)

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
plt.savefig('../../reports/figures/GB_OD_evaluation.png', bbox_inches='tight')
plt.show()

# Access feature importances
importances = grid_search.best_estimator_.named_steps['model'].feature_importances_

# Get feature names from the preprocessor
categorical_features = grid_search.best_estimator_.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out()
numerical_features = numerical_columns 

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
# Gradient Boosting with Engineered Features

# Read in data
recipes_engineered = pd.read_pickle('../../data/processed/recipes_engineered.pkl')

# Features and Target 
X = recipes_engineered.drop(['high_traffic'], axis=1)    
y = recipes_engineered['high_traffic']

# Train-test split
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
        ('num', StandardScaler(), numerical_columns)
    ]
)

# Instantiate Gradient Boosting Classifier
gb = GradientBoostingClassifier(random_state=42)

# Create a Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', gb)
])

# Hyperparameter Tuning
param_grid = {
    'model__n_estimators': [100, 200, 500],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__max_depth': [3, 6, 10]
}

# GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)

# Fit the model
grid_search.fit(X_train, y_train)

# Evaluate
y_pred = grid_search.predict(X_test)
y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
print('\nRESULTS FOR GRADIENT BOOSTING WITH ENGINEERED FEATURES')
print(classification_report(y_test, y_pred))
print('ROC AUC SCORE: ', roc_auc_score(y_test, y_pred_proba))
print("Best parameters found: ", grid_search.best_params_)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Compute ROC curve and ROC AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label='Other')
roc_auc = auc(fpr, tpr)

# Set up the matplotlib figure with two subplots on the same row
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

fig.suptitle('Model Evaluation: Confusion Matrix and ROC Curve - GB with Engineered Data', fontsize=16)

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
plt.savefig('../../reports/figures/GB_ED_evaluation.png', bbox_inches='tight')
plt.show()

# Access feature importances
importances = grid_search.best_estimator_.named_steps['model'].feature_importances_

# Get feature names from the preprocessor
categorical_features = grid_search.best_estimator_.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out()
numerical_features = numerical_columns 

# Combine feature names
all_feature_names = list(categorical_features) + list(numerical_features)

feature_importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
print('Top 10 Features by Importance (Original Data)')
print(feature_importance_df.head(10))

print('-------------------------------------------------------------\n')
