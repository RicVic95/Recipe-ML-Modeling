import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingClassifier

# ------------------------------------------------------------------
# Logistic Regression with box-cox transformations

# Read in data
recipes_base = pd.read_pickle('../../data/processed/recipes_box_cox.pkl')

# Features and Target
X = recipes_base.drop(['high_traffic','high_traffic_bool', 'calories', 'carbohydrate', 'sugar', 'protein'], axis=1)
y = recipes_base['high_traffic']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Separate numerical and categorical features
categorical_columns = ['category']
numerical_columns = [col for col in X.columns if col not in categorical_columns]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns), 
        ('num', StandardScaler(), numerical_columns)
    ]
)

# Instantiate Logistic Regression Model 
log_reg = LogisticRegression(
    max_iter = 10000, 
    random_state=42,
    class_weight='balanced'
)

# Create a Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', log_reg)
])

# Hyperparameter Tunning
param_grid_lr = {
    'model__penalty': ['l1', 'l2'],                  
    'model__C': [0.01, 0.1, 1, 10, 100],            
    'model__solver': ['liblinear', 'saga']   
}

# GridSearchCV 
grid = GridSearchCV(
    pipeline, 
    param_grid=param_grid_lr,
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit the model
grid.fit(X_train, y_train)

# Evaluate 
y_pred = grid.predict(X_test)
y_pred_proba = grid.predict_proba(X_test)[:,1]
print('\nRESULTS FOR LOGISTIC REGRESSION WITH BOX-COX TRANSFORMATIONS')
print(classification_report(y_test, y_pred))
print('\nROC AUC SCORE: ', roc_auc_score(y_test, y_pred_proba))
print('Best parameters found: ', grid.best_params_)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Compute ROC curve and ROC AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label='Other')
roc_auc = auc(fpr, tpr)

# Set up the matplotlib figure with two subplots on the same row
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

fig.suptitle('Model Evaluation: Confusion Matrix and ROC Curve - LR with Original Data', fontsize=16)

# Plot the confusion matrix
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['High', 'Other']).plot(cmap='Blues', ax=axes[0])
axes[0].set_title('Confusion Matrix')

# Plot the ROC curve
axes[1].plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
axes[1].plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('Receiver Operating Characteristic (ROC) Curve')
axes[1].legend(loc="lower right")

# Save and Display the plot
plt.savefig('../../reports/figures/logistic_regression_OD_evaluation.png', bbox_inches='tight')
plt.tight_layout()
plt.show()

# Extract Coefficients for Logistic Regression with Original Data
preprocessor_fitted = grid.best_estimator_.named_steps['preprocessor']
cat_feature_names = preprocessor_fitted.named_transformers_['cat'].get_feature_names_out(categorical_columns)
all_feature_names = list(cat_feature_names) + numerical_columns
coefs = grid.best_estimator_.named_steps['model'].coef_[0]

# Create a DataFrame to view the feature names and their corresponding coefficients
feature_importances = pd.DataFrame({
    'Feature': all_feature_names,  
    'Coefficient': coefs 
})

# Sort the features by the magnitude of the coefficient (ascending order)
feature_importances = feature_importances.sort_values(by='Coefficient', ascending=False)
print('Top 10 Features by Coefficient (original Data)')
print(feature_importances.head(10))

print('----------------------\n')

# ------------------------------------------------------------------
# Logistic Regression with Engineered Data

# Read in data 
recipes = pd.read_pickle('../../data/processed/recipes_boxcox_engineered.pkl')

# Features and Split
X = recipes.drop(['high_traffic', 'high_traffic_bool'], axis=1)
y = recipes['high_traffic']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Separate numerical and categorical features
categorical_columns = ['category', 'nutritional_cluster']
numerical_columns = [
    'servings',
    'calories_boxcox',
    'carbohydrate_boxcox',
    'sugar_boxcox',
    'protein_boxcox',
    'high_traffic_ratio_by_clust',
    'high_traffic_ratio_by_cat',
    'protein_calorie_ratio',
    'sugar_carb_ratio',
    'carb_protein_ratio',
    'avg_category_calories'
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns), 
        ('num', StandardScaler(), numerical_columns)
    ]
)

# Pipeline for processing and training the model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor), 
    ('model', log_reg)
])

# Hyperparameter Tunning
param_grid_lr = {
    'model__penalty': ['l1', 'l2'],                  
    'model__C': [0.01, 0.1, 1, 5],            
    'model__solver': ['liblinear', 'saga'],    
}

# GridSearchCV 
grid = GridSearchCV(
    pipeline, 
    param_grid=param_grid_lr,
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit the model
grid.fit(X_train, y_train)

# Model Evaluation 
# ----------------

y_pred = grid.predict(X_test)
y_pred_proba = grid.predict_proba(X_test)[:,1]
print('\nRESULTS FOR LOGISTIC REGRESSION WITH ENGINEERED DATA')
print(classification_report(y_test, y_pred))
print('\nROC AUC SCORE: ', roc_auc_score(y_test, y_pred_proba))
print('Best parameters found: ', grid.best_params_)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Compute ROC curve and ROC AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label='Other')
roc_auc = auc(fpr, tpr)

# Set up the matplotlib figure with two subplots on the same row
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

fig.suptitle('Model Evaluation: Confusion Matrix and ROC Curve - LR with Engineered Data', fontsize=16)

# Plot the confusion matrix
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['High', 'Other']).plot(cmap='Blues', ax=axes[0])
axes[0].set_title('Confusion Matrix')

# Plot the ROC curve
axes[1].plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
axes[1].plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('Receiver Operating Characteristic (ROC) Curve')
axes[1].legend(loc="lower right")

# Save and Display the plot
plt.savefig('../../reports/figures/logistic_regression_ED_evaluation.png', bbox_inches='tight')
plt.tight_layout()
plt.show()

# Extract Coefficients for Logistic Regression with Original Data
preprocessor_fitted = grid.best_estimator_.named_steps['preprocessor']
cat_feature_names = preprocessor_fitted.named_transformers_['cat'].get_feature_names_out(categorical_columns)
all_feature_names = list(cat_feature_names) + numerical_columns
coefs = grid.best_estimator_.named_steps['model'].coef_[0]

# Create a DataFrame to view the feature names and their corresponding coefficients
feature_importances = pd.DataFrame({
    'Feature': all_feature_names,  
    'Coefficient': coefs 
})

# Sort the features by the magnitude of the coefficient (ascending order)
feature_importances = feature_importances.sort_values(by='Coefficient')
print('Top 10 Features by Coefficient (original Data)')
print(feature_importances.head(10))

