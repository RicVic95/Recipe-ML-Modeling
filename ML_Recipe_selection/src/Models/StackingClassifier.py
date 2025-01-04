import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin

# ------------------------------------------------------------------
# Custom Transformer to add a small constant to specified columns

class AddConstantTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to add a small constant to specified columns.
    """
    def __init__(self, constant=1, columns=None):
        self.constant = constant
        self.columns = columns  # Columns to apply the transformation

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.columns is not None:
            X[self.columns] += self.constant
        return X


# ------------------------------------------------------------------
# Voting Classifier with RandomForest, LogReg, and Gradient Boosting

# Read in data
recipes = pd.read_pickle('../../data/processed/recipes_site_traffic_clean.pkl')

# Features and Target
X = recipes.drop(['high_traffic', 'high_traffic_bool'], axis=1)
y = recipes['high_traffic']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Separate numerical and categorical features
categorical_columns = ['category']
numerical_columns = ['servings', 'calories', 'carbohydrate', 'sugar', 'protein']

# ------------------------------------ # 
# Logistic Regression                  #
# ------------------------------------ #

# Preprocessor for Logistic Regression
logreg_preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
        ('num', Pipeline([
            ('add_constant', AddConstantTransformer(constant=1, columns=['calories', 'carbohydrate', 'sugar', 'protein'])),  # Add constant to columns
            ('boxcox', PowerTransformer(method='box-cox')),  # Apply Box-Cox Transformation
            ('scaler', StandardScaler())  # Scale features
        ]), numerical_columns)
    ]
)

# Logistic Regression model with best hyperparameters
logreg = LogisticRegression(
    C=10, 
    penalty='l1', 
    solver='liblinear', 
    class_weight='balanced',
    random_state=42
)

# Logistic Regression pipeline
logreg_pipeline = Pipeline([
    ('preprocessor', logreg_preprocessor),
    ('model', logreg)
])

# ------------------------------------ # 
# Random Forest                        #
# ------------------------------------ #

# Random Forest model with best hyperparameters
rf = RandomForestClassifier(
    bootstrap=True, 
    max_depth=10, 
    max_features='sqrt', 
    min_samples_leaf=4, 
    min_samples_split=10, 
    n_estimators=200, 
    random_state=42,
    class_weight='balanced',
)

# Random Forest preprocessor
rf_preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
        ('num', StandardScaler(), numerical_columns)  # Standard scaler only, no Box-Cox
    ]
)

# Random Forest pipeline
rf_pipeline = Pipeline([
    ('preprocessor', rf_preprocessor),
    ('model', rf)
])

# ------------------------------------ #
# Stacking Classifier                  #
# ------------------------------------ #

# Base models as before
base_models = [
    ('logreg', logreg_pipeline),
    ('rf', rf_pipeline)
]

# Meta-model (Logistic Regression) 

# Define a meta-model (Logistic Regression)
meta_model = LogisticRegression(
    solver='liblinear',
    class_weight='balanced',
    random_state=42
)

# Stacking Classifier              

# Create Stacking Classifier
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)

# Parameter grid for the meta-model
param_grid_stacking = {
    'final_estimator__C': [0.01, 0.1, 1, 10, 100],  
    'final_estimator__penalty': ['l1', 'l2'],        
}

# Stacking Classifier with GridSearchCV
grid_search_stacking = GridSearchCV(
    estimator=stacking_clf,
    param_grid=param_grid_stacking,
    cv=3,
    n_jobs=-1,
    verbose=1
)

# Fit the GridSearchCV with Stacking Classifier
grid_search_stacking.fit(X_train, y_train)

# Best parameters for the meta-model
print("Best parameters for the meta-model:", grid_search_stacking.best_params_)

# Evaluate the optimized Stacking Classifier
y_pred = grid_search_stacking.predict(X_test)
y_prob = grid_search_stacking.predict_proba(X_test)[:, 1]

# Print evaluation metrics
print(classification_report(y_test, y_pred))
print('ROC AUC SCORE:', roc_auc_score(y_test, y_prob))

# Extract coefficients from the meta-model
meta_model_coefs = grid_search_stacking.best_estimator_.final_estimator_.coef_[0]

# Get base model names as "features"
base_model_names = [name for name, _ in base_models]

# Create a DataFrame for visualization
meta_model_importance_df = pd.DataFrame({
    'Base Model': base_model_names,
    'Coefficient': meta_model_coefs
}).sort_values(by='Coefficient', ascending=False)

print("Meta-Model Coefficients:")
print(meta_model_importance_df)

# Extract the best estimator (optimized StackingClassifier)
best_stacking_clf = grid_search_stacking.best_estimator_

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label='Other')
roc_auc = auc(fpr, tpr)

# Create a single figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot the confusion matrix on the first subplot
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=['Other', 'High'], yticklabels=['Other', 'High'])
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

# Plot the ROC curve on the second subplot
axes[1].plot(fpr, tpr, color='b', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
axes[1].plot([0, 1], [0, 1], color='gray', linestyle='--')
axes[1].set_title('Receiver Operating Characteristic (ROC) Curve')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(loc='lower right')

# Adjust layout and show the figure
fig.suptitle('Meta-model Evaluation: Confusion Matrix and ROC Curve (Stacking Classifier)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('../../reports/figures/stacking_classifier_evaluation.png', bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------
# Production Model: Stacking Classifier
# ------------------------------------------------------------------

import joblib  # Library to save the trained model

# Retrieve the best Stacking Classifier from GridSearchCV
optimized_stacking_clf = grid_search_stacking.best_estimator_

# Train the best estimator on the entire dataset
optimized_stacking_clf.fit(X, y)

# Save the trained model to disk
model_path = '../../models/stacking_classifier.pkl'
joblib.dump(optimized_stacking_clf, model_path)

print(f"Trained Stacking Classifier with best hyperparameters saved to {model_path}")