# Executive Summary

## **Objective**
The goal of this project was to develop a robust machine learning solution to help the business identify recipes that drive high website traffic. By accurately predicting which recipes will generate high traffic, the company aims to maximize user engagement and increase subscription rates, while minimizing the risk of recommending unpopular recipes.

## **Approach**
To achieve this, the project followed a systematic process involving data validation, exploratory analysis, feature engineering, model development, and evaluation. Three machine learning models—**Logistic Regression**, **Random Forest**, and a **Stacking Classifier**—were evaluated to identify the best-performing approach.

Key considerations included:
- Ensuring the models predicted high-traffic recipes with **80% accuracy or higher**.
- Developing and testing models on both the **original dataset** and an **engineered dataset** enriched with traffic-based and interaction features.
- Addressing class imbalances, as recipes labeled as "Other" (low traffic) comprised 40% of the data.
- Balancing model precision and recall to avoid both missed opportunities for high-traffic recipes and the inclusion of low-traffic recipes on the homepage.

## **Results**
The Stacking Classifier, combining **Logistic Regression** and **Random Forest** as base models, emerged as the best-performing solution, achieving:
- **F1-Score for High-Traffic Recipes**: 0.81
- **Precision for High-Traffic Recipes**: 0.82
- **Recall for High-Traffic Recipes**: 0.80
- **ROC AUC Score**: 0.83

These results met the project’s primary objective of predicting high-traffic recipes with strong reliability. The Stacking Classifier also improved recall for the "Other" class (0.74), ensuring better identification of low-traffic recipes compared to individual models.

While the engineered dataset contributed valuable insights (e.g., category-based traffic ratios), concerns about overfitting and reliance on dominant features led to prioritizing models trained on the **original dataset** for production.

## **Recommendations**
1. **Deploy the Stacking Classifier**:
   - This model should be implemented as the primary tool for recommending recipes on the homepage. Its robust performance ensures a balance between precision and recall for both high- and low-traffic recipes.

2. **Monitor and Evaluate Performance**:
   - Use **Recall for High-Traffic Recipes** as the primary metric, with thresholds of 70%-80% for good performance and >80% for optimal performance.
   - Track user engagement metrics, such as clicks and subscriptions, to validate predictions.

3. **Address Class Imbalance**:
   - Consider advanced techniques like cost-sensitive learning or oversampling to further improve recall for the "Other" class.

4. **Periodic Model Retraining**:
   - Regularly update the model using newly collected data, particularly during seasonal or holiday periods, to capture shifting trends and ensure ongoing accuracy.

5. **Expand Dataset**:
   - Incorporate additional recipes and user behavior data to improve model generalizability and robustness.

6. **Leverage Engineered Features for Insights**:
   - While the engineered dataset raised overfitting concerns, features like `high_traffic_ratio_by_cat` provided actionable insights into category popularity. These insights can inform broader marketing strategies beyond recipe recommendations.

## **Key Impact**
Deploying the Stacking Classifier has the potential to significantly improve homepage traffic predictions. By moving from the current manual selection process (60% accuracy) to a data-driven approach, the business can expect:
- A **10%-20% increase in prediction accuracy** during the early phases of deployment.
- Enhanced user engagement and subscription rates due to more accurate recommendations.
- Data-driven insights that empower strategic decision-making across marketing and product initiatives.

With these recommendations, the company can effectively align its homepage recipe selection process with its strategic goal of maximizing website traffic and driving subscriptions.

# **Introduction**

Recipes play a pivotal role in attracting visitors to Tasty Byte's website, with popular recipes driving up to 40% more traffic and increasing subscriptions. However, selecting recipes likely to generate high user engagement has traditionally relied on subjective judgment.

This project aims to address this challenge by leveraging data-driven machine learning models to predict which recipes are likely to experience high traffic. By accurately identifying high-traffic recipes, the company can enhance user satisfaction, optimize homepage content, and ultimately drive more subscriptions.

The following report outlines the data validation, analysis, feature engineering, and modeling steps taken to achieve this goal, as well as recommendations for deploying and monitoring the selected model to maximize business impact.

# **Data Validation**

This project utilized a dataset containing recipe information, including both numerical and categorical features, as well as a binary target variable, `high_traffic`. The following validation and cleaning steps were conducted to ensure the data was accurate, consistent, and ready for analysis.

## **Validation and Adjustments by Column**

**1. `recipe`**
- **Description**: A unique numeric identifier for each recipe.
- **Validation**: 
  - No missing or duplicate values were detected. 
- **Adjustment**: 
  - None required.

**2. Nutritional Features: `calories`, `carbohydrate`, `sugar`, `protein`**
- **Description**: Numerical features providing nutritional information for each recipe.
- **Validation**: 
  - Missing values were detected in some columns. 
- **Adjustment**: 
  - Rows with missing values were removed, given the small proportion of missing data, to maintain data integrity and avoid potential biases from imputation.

**3. `category`**
- **Description**: A categorical variable representing one of ten recipe groupings:
  - `Lunch/Snacks`, `Beverages`, `Potato`, `Vegetable`, `Meat`, `Chicken`, `Pork`, `Dessert`, `Breakfast`, and `One Dish Meal`.
- **Validation**: 
  - Inconsistent labeling was identified. For instance:
    - `Chicken Breast` was incorrectly categorized instead of being grouped under the broader category `Chicken`.
- **Adjustment**: 
  - Labels like `Chicken Breast` were corrected to their general categories.
  - The final categories were verified to align with the ten intended groupings.

**4. `servings`**
- **Description**: A numerical feature indicating the number of servings for each recipe.
- **Validation**: 
  - Inconsistent entries were detected, such as textual formats (e.g., "4 as a snack").
- **Adjustment**: 
  - Entries were cleaned and converted into a uniform numeric format to ensure compatibility with downstream transformations.

**5. `high_traffic`**
- **Description**: A binary categorical variable indicating whether a recipe experienced high traffic (`High`).
- **Validation**: 
  - Missing values were found for all entries that did not experience `High` traffic.
- **Adjustment**: 
  - Missing values were encoded as `Other` for clarity.
  - A new column, `high_traffic_bool`, was created to facilitate numerical operations:
    - Recipes with `High` traffic were labeled as `True`.
    - Recipes with `Other` traffic were labeled as `False`.

## **Post-Validation Summary**

Following cleaning and validation, the dataset contained **895 entries**, with each record representing a recipe characterized by consistent and well-defined features. These adjustments ensured the data was robust and ready for subsequent analysis and modeling.

# **Exploratory Data Analysis**

The dataset was analyzed to understand the relationships between features and the target variable (`high_traffic`). This analysis aimed to uncover patterns and distributions that could inform the subsequent feature engineering and modeling processes.

## **Exploring Recipe Categories**

Before examining feature relationships, the distribution of the `high_traffic` target variable was explored. The dataset includes:
- **535 recipes labeled as 'high' traffic**, representing 59.8% of entries.
- **360 recipes categorized as 'other' traffic**, comprising 40.2%.

The figure below displays:
1. The distribution of recipes across categories.
2. The percentage of high-traffic recipes within each category.

<div align='center'> 
    <img src= 'ML_Recipe_selection/reports/figures/category_summary.png' alt='Cat_summary' width=500> 
</div>    

![Category Plots](figures/category_summary.png)

Key Observations:
- Categories like `Chicken`, `Breakfast`, and `Beverages` represent approximately 40% of the dataset but have lower proportions of high-traffic recipes (42%, 31%, and 5%, respectively).
- Categories such as `Vegetable`, `Potato`, and `Pork` show significantly higher proportions of high-traffic recipes (nearly 100% for `Vegetable` and 94% for `Pork`).

These findings underscore the importance of recipe categories in predicting traffic levels. The disparities in high-traffic proportions suggest that category-level features, such as the ratio of high-traffic recipes per category, could enhance prediction accuracy.

## **Inspecting Nutritional Features**

Four key nutritional features—`calories`, `carbohydrate`, `sugar`, and `protein`—were analyzed for their potential predictive value. Initial visualizations revealed significant skewness in these distributions, indicating the presence of outliers and suggesting the need for normalization.

![Nutritional Features Distribution](figures/nutritional_features_distributions.png)

## **Addressing Skewness: Box-Cox Transformation**
A **Box-Cox transformation** was applied to normalize the skewed nutritional features. Before applying the transformation:
1. **Zero Values Check**: 
   - Zero values were identified in the `protein` column. A constant value of `1` was added to all nutritional columns to ensure positivity.
2. **Negative Values Check**: 
   - All columns were verified to contain only positive values.

Post-transformation, new columns (`calories_boxcox`, `carbohydrate_boxcox`, `sugar_boxcox`, `protein_boxcox`) were created. The optimal lambda (λ) values for each feature were recorded:

| **Feature**       | **Optimal Lambda Value (λ)** |
|-------------------|------------------------------|
| Calories          | 0.2426                       |
| Carbohydrate      | 0.0958                       |
| Sugar             | -0.2042                      |
| Protein           | 0.0054                       |

### **Evaluating the Transformation**
The transformed distributions showed improved symmetry, indicating successful normalization:

![Box-Cox Nutritional Features Distribution](figures/BC_nutritional_features_distributions.png)

## **Correlation Analysis**

The relationship between features and the target variable was examined. The bar plot below illustrates the correlations:

![Correlation Barplot](figures/correlation_barplot.png)

Key Findings:
- Correlations were generally weak, suggesting potential non-linear relationships between features and the target variable.
- This emphasizes the need for advanced feature engineering and the application of non-linear models.

## **Conclusion and Transition to Feature Engineering**

The exploratory analysis revealed critical insights:
1. **Recipe Categories**: Category-level features significantly impact traffic levels, with disparities in high-traffic proportions across categories.
2. **Nutritional Features**: Skewness was successfully addressed using Box-Cox transformations, making the data more suitable for machine learning.
3. **Weak Correlations**: The observed weak correlations highlight the importance of feature engineering to uncover latent patterns.

The next phase will focus on feature engineering to create meaningful features and capture non-linear relationships, enhancing the predictive power of the models.

# **Feature Engineering**

## **Objective**
The objective of this phase was to enrich the dataset by generating additional features that leverage clustering and interaction effects. These engineered features aimed to capture deeper relationships within the data, improving the predictive power of machine learning models.

## **Clustering Nutritional Profiles Using K-Means**

K-Means clustering was utilized to group recipes based on their nutritional profiles, uncovering distinct recipe segments and providing valuable insights into traffic patterns. To optimize the clustering process, the Box-Cox transformations created before were utilized in order to address skewness in the data and ensure compatibility with the algorithm’s reliance on Euclidean distances and parametric assumptions. The Elbow Method was employed to identify the optimal number of clusters, selecting five clusters, as illustrated in the plot below:

![Elbow-Method](figures/elbow_method_optimal_k.png)

**Cluster Insights**
Each cluster was analyzed for its nutritional characteristics and assigned a meaningful label. A new feature, `nutritional_cluster`, was created and assigned to each recipe. The table below summarizes the clusters:

| **Cluster** | **Label**                 | **Calories (Mean)** | **Carbohydrates (Mean)** | **Sugar (Mean)** | **Protein (Mean)** |
|-------------|---------------------------|----------------------|--------------------------|------------------|--------------------|
| 0           | High-Protein / High-Calorie | 718.73               | 9.91                     | 8.09             | 49.10             |
| 1           | Low-Calorie / Moderate-Carb | 237.52               | 17.62                    | 2.34             | 6.58              |
| 2           | Balanced / High-Protein    | 110.80               | 51.56                    | 9.65             | 37.29             |
| 3           | Sweet / Dessert-Like       | 233.58               | 26.31                    | 22.97            | 4.18              |
| 4           | High-Carb / High-Calorie   | 762.64               | 71.36                    | 5.53             | 22.43             |

## **Traffic Ratio Features**

**Traffic Ratio by Cluster**
The proportion of high-traffic recipes (`high_traffic=1`) was calculated within each nutritional cluster and stored as `high_traffic_ratio_by_clust`. This feature highlights the relative popularity of recipes within each cluster.

**Traffic Ratio by Category**
Similarly, the proportion of high-traffic recipes within each category (e.g., "Lunch/Snacks", "Dessert") was computed and stored as `high_traffic_ratio_by_cat`. This feature reflects category-level performance trends and their influence on recipe traffic.

## **Interaction Features**

To explore relationships between nutritional components, the following interaction features were introduced:
- **Protein-Calorie Ratio**: Ratio of protein to calories (`protein / calories`).
- **Sugar-Carbohydrate Ratio**: Ratio of sugar to carbohydrates (`sugar / carbohydrate`).
- **Carbohydrate-Protein Ratio**: Ratio of carbohydrates to protein (`carbohydrate / protein`).

These features were designed to uncover more granular relationships between nutritional composition and recipe popularity.

## **Category Popularity Features**

A feature named `avg_category_calories` was created by calculating the average calories for recipes within each category. This feature contextualizes trends within categories, offering insights into consumer preferences and typical caloric content.

## **New Features and Correlation**

The correlation analysis revealed the following relationships between the engineered features and high traffic:

![Corr-eng-feat](figures/Corr_engineered_features.png)

**Key Findings:**
- **Strongest Correlation**: `high_traffic_ratio_by_cat` exhibited the highest positive correlation with high traffic (0.58), indicating that recipes from popular categories are more likely to generate high traffic.
- **Moderate Correlation**: `high_traffic_ratio_by_clust` showed a moderate correlation (0.17), suggesting nutritional clusters influence traffic but less significantly than category popularity.
- **Weaker Correlations**: Features like `protein_calorie_ratio` (0.04), `carb_protein_ratio` (-0.02), `nutritional_cluster` (-0.05), and `sugar_carb_ratio` (-0.05) displayed weak linear relationships with high traffic. Despite this, they may still capture non-linear patterns that advanced models can exploit.

## **Conclusion for Feature Engineering**

1. **Clustering Enhancements**: Nutritional clustering enabled the abstraction of complex relationships into meaningful segments, potentially reflecting consumer preferences.
2. **Traffic Ratios**: Features such as `high_traffic_ratio_by_cat` and `high_traffic_ratio_by_clust` provided valuable insights into recipe popularity, with the former demonstrating strong predictive potential.
3. **Interaction Features**: While linear correlations for interaction features were weak, they may reveal non-linear patterns when used in sophisticated models.
4. **Category Popularity Features**: Features like `avg_category_calories` contextualize recipe trends, aligning with observed correlations between category-level popularity and high traffic.

These engineered features enhanced the dataset’s representation, laying a strong foundation for the modeling phase. The next section will detail the models developed and their performance in predicting high-traffic recipes.

# **Model Development**

## Model Development

As outlined in the introduction, the objective of this project is to accurately identify recipes that can drive increased website traffic. This project addresses a **binary classification problem**, where the goal is to predict whether a recipe will generate high traffic (`High`) or not (`Other`). Specifically, the aim is to correctly predict high-traffic recipes at least 80% of the time while minimizing the likelihood of recommending unpopular recipes.

To achieve this, the following machine learning models were explored: **Logistic Regression**, **Random Forest Classifier**, and a **Stacking Classifier** that combined the strengths of both LR and RF models and reduced their weaknesses.

These models were trained and evaluated using both the **original dataset** and the **engineered dataset** developed during the Feature Engineering section of this project. This was done to assess the impact of feature engineering on predictive performance. A **5-fold cross-validation** was used for hyperparameter tunning and to ensure each model achieves its optimal configuration. This iterative process allowed the selection of the best parameters while maintaining robust performance across different subsets of the data.

It's important to note that in all models, the class labeled as 'Other' was considered as the **positive class** 

## **Evaluation Framework**

The success of the models was evaluated based on their ability to meet the following business-driven criteria:

### **Primary Metric**
- **F1-Score for the "High" Class**: The primary metric, emphasizing the balance between precision and recall, with a target of **≥ 0.80** for the "High" class.

### **Supporting Metrics**
- **Precision for the "High" Class**: Measures the proportion of correctly identified high-traffic recipes among those predicted as such, with a threshold of **≥ 0.80**.
- **Recall for the "High" Class**: Assesses the proportion of actual high-traffic recipes that are correctly identified, also with a target of **≥ 0.80**.
- **Precision for the “Other” Class**: Minimizes the risk of recommending unpopular recipes by ensuring accurate classification of low-traffic recipes.
- **ROC AUC Score**: Provides a global measure of model performance across all classification thresholds, with a goal of **≥ 0.80**.

## Logistic Regression Model Analysis

### Best Hyperparameters 

| **Dataset**       | **C**  | **Penalty** | **Solver**   |
|--------------------|--------|-------------|--------------|
| Original Dataset   | 10     | l1          | liblinear    |
| Engineered Dataset | 0.01   | l1          | liblinear    |

GridSearchCV revealed differences in optimal hyperparameters between the original and engineered datasets. The engineered dataset required stronger regularization (C=0.01), likely due to the added complexity from engineered features. Both models selected the l1 penalty, emphasizing feature selection and the sparsity of predictive information. These findings highlight the importance of tuning to balance model complexity and performance.

### Model Performance Comparison

| Metric                | Original Data | Engineered Data |
|-----------------------|---------------|-----------------|
| **Precision (High)**  | 0.80          | 0.81            |
| **Precision (Other)** | 0.72          | 0.73            |
| **Recall (High)**     | 0.82          | 0.82            |
| **Recall (Other)**    | 0.69          | 0.71            |
| **F1-Score (High)**   | 0.81          | 0.81            |
| **F1-Score (Other)**  | 0.71          | 0.72            |
| **Accuracy**          | 0.77          | 0.78            |
| **ROC AUC Score**     | 0.84          | 0.84            |

The engineered data model showed slightly better precision for both classes and a marginally higher macro average F1-score. Both models performed similarly in terms of ROC AUC (~0.84), indicating strong ability to distinguish between high and non-high traffic recipes. The improvements in the engineered model stem from the inclusion of engineered features.

### Feature Coefficients and Model Interpretation  
   - **Original Data**: The top features were primarily **category-based**, with **category_Beverages** (coefficient = 3.73) being the most influential, suggesting a strong association with the Other class. Features like **Lunch/Snacks** had zero coefficients, indicating no predictive value.
   - **Engineered Data**: The top feature, **high_traffic_ratio_by_cat** (coefficient = -0.62), showed a strong negative relationship with the class labeled as '**Other**', making it a key predictor of high-traffic recipes. All other engineered features had near-zero coefficients, indicating limited contribution.

### Model Implications  
   - **Original Data Model**: Relied heavily on category features, indicating potential overfitting to these variables. 
   - **Engineered Data Model**: Outperformed the original model slightly due to **high_traffic_ratio_by_cat**, which captures traffic distribution within categories. 

### Caveats and Considerations  
   - **Class Imbalance**: Stratification was used to balance the class distribution, but the model still performed better on the **High** class. Techniques like class weighting or SMOTE could be explored to address this.
   - **Overfitting**: The engineered data model performed slightly better, but overfitting is a concern, especially with the small dataset. Cross-validation mitigates this, but model generalization should be closely monitored.

### Conclusion  
Both models performed well, with the engineered data model showing marginal improvements, mainly due to the `high_traffic_ratio_by_cat` feature. However, the engineered data model relies heavily on this single feature, which raises concerns about overfitting, particularly given the small dataset size. On the other hand, the original data model, while slightly less precise, uses a broader set of features and seems less prone to overfitting, making it more likely to generalize well to new data. Considering these factors, the model trained on the original dataset was chosen as the best perfoming model, since it may be the safer choice for avoiding overfitting and ensuring robust performance across diverse scenarios. 

![LR_Eval](figures/logistic_regression_OD_evaluation.png)

## Random Forest Model Analysis  

### Best Hyperparameters Found 

| Parameter              | Original Dataset | Engineered Dataset |
|------------------------|------------------|--------------------|
| Bootstrap              | True            | True               |
| Max Depth              | 10              | 10                 |
| Max Features           | sqrt            | log2               |
| Min Samples Leaf       | 4               | 2                  |
| Min Samples Split      | 10              | 10                 |
| Number of Estimators   | 200             | 200                |

GridSearchCV identified similar hyperparameters for both the original and engineered datasets, with bootstrap=True and a maximum depth of 10, suggesting the model benefits from reduced complexity to avoid overfitting. However, the engineered dataset selected max_features='log2' and min_samples_leaf=2, reflecting the increased importance of exploring more feature subsets and using fewer samples per leaf. This adaptation to the engineered dataset highlights the necessity of tuning to balance model interpretability and predictive power.

### Model Performance Comparison  

| Metric                | Original Data | Engineered Data |
|-----------------------|---------------|-----------------|
| **Precision (High)**  | 0.73          | 0.76            |
| **Precision (Other)** | 0.69          | 0.71            |
| **Recall (High)**     | 0.84          | 0.83            |
| **Recall (Other)**    | 0.53          | 0.61            |
| **F1-Score (High)**   | 0.78          | 0.79            |
| **F1-Score (Other)**  | 0.60          | 0.66            |
| **Accuracy**          | 0.72          | 0.74            |
| **ROC AUC Score**     | 0.82          | 0.81            |

The **engineered model** performed better overall, particularly in recall and F1-score for the **Other class**, though the difference in ROC AUC scores was minimal.

### Feature Importances and Model Interpretation

**Original Data**  
The most important features were **category_Beverages** (18.2%), **protein** (17.1%), and **calories** (9.8%), indicating that both categorical and numerical variables contributed meaningfully to the model.  

**Engineered Data**  
The engineered feature **high_traffic_ratio_by_cat** was the most significant (18.7%), followed by **avg_category_calories** (10.6%) and **protein** (8.9%). This suggests the engineered features successfully captured patterns beyond the raw dataset.

### Caveats and Considerations  

1. **Bootstrapping Benefits**  
   Both models used **bootstrapping** (`bootstrap=True`), which aims to mitigate the impact of class imbalances by ensuring minority classes were included in training samples.  

2. **Class Imbalance Challenges**  
   Despite bootstrapping, the models had difficulty with recall for the **'Other' class**, particularly in the original data model.

3. **Feature Dependence and Generalization**  
   The engineered model heavily relied on **high_traffic_ratio_by_cat**, which may overfit to the training data and limit generalization.

### Conclusion  
The Random Forest model trained on the engineered dataset achieves strong performance, with an F1-score of 0.79 and recall of 0.83 for high-traffic recipes, ensuring it identifies popular recipes effectively. However, it struggles with the ‘Other’ class, achieving a recall of 0.61, potentially missing 39% of unpopular recipes. While its precision for high-traffic recipes (0.76) aligns with project objectives, the model’s reliance on the high_traffic_ratio_by_cat feature (18.7% importance) raises concerns about overfitting.

Given the importance of robustness and generalizability, we selected the Random Forest model trained on the original data as the best option. Although its F1-score for high-traffic recipes (0.78) is slightly lower, it avoids over-reliance on a single feature, reducing overfitting risks and improving its suitability for unseen data.

![RF_Eval](figures/RF_OD_evaluation.png)

## Stacking Classifier Model Analysis  

### Model Overview  
The Stacking Classifier was built using Logistic Regression and Random Forest (optimized with their best hyperparameters) as base models, with Logistic Regression serving as the meta-model. To address class imbalance observed in previous iterations, balanced class weights were applied across all models. Since both Logistic Regression and Random Forest performed best with the original dataset, this dataset was chosen for training the Stacking Classifier.

The meta-model was optimized using GridSearchCV, resulting in a regularization strength of C = 10 and an L1 penalty for feature selection.

### Model Performance  

| Metric                | Value     |
|-----------------------|-----------|
| **Precision (High)**  | 0.82      |
| **Precision (Other)** | 0.72      |
| **Recall (High)**     | 0.80      |
| **Recall (Other)**    | 0.74      |
| **F1-Score (High)**   | 0.81      |
| **F1-Score (Other)**  | 0.73      |
| **Accuracy**          | 0.78      |
| **ROC AUC Score**     | 0.83      |

The Stacking Classifier achieved strong overall performance, with an **F1-score of 0.81** for the **High class**, indicating reliable prediction of high-traffic recipes. The **ROC AUC score of 0.83** demonstrating a strong capability to distinguish between high-traffic and low-traffic recipes effectively.  

### Meta-Model Coefficients  

| Base Model           | Coefficient |
|-----------------------|-------------|
| Logistic Regression   | 3.78        |
| Random Forest         | 1.49        |

The coefficients reveal that the **Logistic Regression** base model contributed more to the final predictions compared to the **Random Forest**, reflecting the meta-model’s preference for the more interpretable and generalized patterns captured by Logistic Regression.

### Key Strengths  

1. **Balanced Performance Across Classes**  
   - **High Class**: Precision (0.82) and recall (0.80) indicate the model’s ability to identify high-traffic recipes while minimizing false positives.
   - **Other Class**: Precision (0.72) and recall (0.74) demonstrate a balanced trade-off, ensuring reasonable performance for identifying low-traffic recipes.  

2. **Class Weighting**  
   - By setting `class_weight='balanced'`, the models effectively accounted for class imbalance, ensuring that both high-traffic and non-high-traffic recipes were adequately represented during training.  

4. **Combining Model Strengths**  
   - The Stacking Classifier leverages both the interpretability of Logistic Regression and the ability of Random Forest to capture non-linear relationships, resulting in robust performance.

### Caveats and Considerations  

1. **Class Imbalance Challenges**  
   - Despite the use of balanced class weights, precision for the **'Other' class** (0.72) suggests that some low-traffic recipes may still be misclassified.  

2. **Meta-Model Dependence**  
   - The meta-model relied heavily on Logistic Regression, as indicated by its higher coefficient. While this enhances interpretability, it may limit the model’s ability to fully leverage Random Forest’s strengths in capturing complex interactions.  

3. **Generalizability**  
   - The dataset size remains a limitation, potentially affecting the model’s performance on unseen data. Testing on external datasets is recommended to validate its robustness and ensure long-term generalizability.  

### Conclusion  
The Stacking Classifier demonstrates robust performance, aligning well with the project’s objective of accurately predicting high-traffic recipes while minimizing false positives. With a **ROC AUC score of 0.83** and balanced performance across both classes, the model effectively meets the criteria for identifying high-traffic recipes. However, further refinements, such as addressing residual class imbalance and validating on larger datasets, could enhance its predictive power and generalizability. 

![SC_Eval](figures/stacking_classifier_evaluation.png)

## **Conclusion for Model Development**

The development and evaluation of machine learning models for predicting high-traffic recipes have highlighted the strengths and limitations of various approaches, including Logistic Regression, Random Forest, and the Stacking Classifier. This process underscored the importance of balancing predictive accuracy with robustness and generalizability.

**Usefulness of the Engineered Dataset:**  
The engineered dataset was instrumental in improving model performance, particularly for Random Forest and Logistic Regression, due to features like `high_traffic_ratio_by_cat` and `avg_category_calories`, which captured meaningful traffic patterns. However, reliance on a single dominant feature (`high_traffic_ratio_by_cat`) raised concerns about overfitting, especially with a small dataset. While the engineered features added value, their potential to introduce noise and overfit the training data ultimately led to prioritizing models trained on the original dataset for better generalizability.

**Model Selection Rationale:**  
The Stacking Classifier emerged as the best-performing model, combining the strengths of Logistic Regression and Random Forest. As shown in the table below, this meta-model demonstrated balanced performance across both high-traffic and non-high-traffic recipes, achieving a **ROC AUC score of 0.83**, precision of **0.82 for high-traffic recipes**, and recall of **0.80**, aligning with the project's primary objective of predicting high-traffic recipes with high reliability. Additionally, the Stacking Classifier improved recall for the `Other` class (0.74), ensuring better identification of low-traffic recipes compared to previous models.

| **Metric**           | **Logistic Regression (Original Data)** | **Random Forest (Original Data)** | **Stacking Classifier** |
|-----------------------|------------------------------------------|------------------------------------|--------------------------|
| **Precision (High)**  | 0.80                                    | 0.73                              | 0.82                    |
| **Precision (Other)** | 0.72                                    | 0.69                              | 0.72                    |
| **Recall (High)**     | 0.82                                    | 0.84                              | 0.80                    |
| **Recall (Other)**    | 0.69                                    | 0.53                              | 0.74                    |
| **F1-Score (High)**   | 0.81                                    | 0.78                              | 0.81                    |
| **F1-Score (Other)**  | 0.71                                    | 0.60                              | 0.73                    |
| **Accuracy**          | 0.77                                    | 0.72                              | 0.78                    |
| **ROC AUC Score**     | 0.84                                    | 0.82                              | 0.83                    |

**Future Directions:**  
While the engineered dataset added valuable insights, future iterations should focus on refining feature engineering to reduce noise and mitigate overfitting risks. Class imbalance remains a challenge, suggesting the need for advanced techniques such as cost-sensitive learning or oversampling. Expanding the dataset and testing on external data will further enhance model generalizability and robustness.

# Final Summary and Recommendations 

This project successfully developed a machine learning pipeline to predict high-traffic recipes with strong precision and recall, achieving a ROC AUC of 0.83 and an F1-score of 0.81 for high-traffic recipes using the Stacking Classifier. The engineered dataset provided valuable insights but introduced overfitting risks. Consequently, models trained on the original dataset demonstrated better generalizability.

**Key Recommendations**

1.	**Deploy the Stacking Classifier**
    - The Stacking Classifier should be implemented as the production model due to its robust performance across both classes. This model aligns closely with the business goal of recommending recipes that maximize user engagement while minimizing the risk of serving unpopular options.
2.	**Address Class Imbalance**
    - Even though the Stacking Classifier reduced the concerns about class imbalance, other options like oversampling, undersampling, or cost-sensitive learning approaches could be taken into consideration should there be a need to further improve the model's performance. 
3.	**Scale and Validate the Model**
    - Test the model on external or newly acquired datasets to validate performance and ensure robustness.
    - Expand the dataset to capture a broader range of recipes and user preferences.
    - Periodically retrain the model on new data to improve performance.
4.	**Continuous Monitoring and Feedback**
    - Implement a real-time monitoring system to track prediction quality and user engagement metrics.
    - Stay up to date with seasonal trends and monitor the performance of holiday recipes (E.g. Christmas, Thanksgiving, etc.) as these may impact the results of the model. 
    - Incorporate user feedback to refine the model and improve future recommendations. (E.g. Ratings for recipes)
5.	**Strategic Integration of the Model**
    - Expand the model to enhance recipe recommendations for subscribers, content personalization, and user experience.
    - Integrate the prediction system into marketing campaigns to promote high-traffic recipes effectively.

# Monitoring Plan After Deployment 

## Proposed Metrics to Follow

**Primary Metric**: 

To align with the business goal of maximizing website traffic by showcasing high-traffic recipes, the **Recall for the “High” Class** is recommended as the primary metric to monitor. Recall directly measures the proportion of actual high-traffic recipes that are correctly identified by the model.

**Supporting Metrics**: 

While Recall is the primary focus, three supporting metrics can provide a comprehensive view of model performance:
1.	**Precision for the “High” Class**: Ensures the predictions made for high-traffic recipes are accurate, minimizing the risk of displaying recipes that won’t resonate with users.
2.	**F1-Score for the “High” Class**: Balances precision and recall, offering a single metric that reflects both capturing high-traffic recipes and avoiding false positives.
3.	**Precision for the "Other" Class**: Ensures that recipes that do not experience high traffic are filtered out by the model. 

## Monitoring Process

The Business can monitor the model’s performance using a structured workflow:
1.	**Daily Metrics Review**: Track the recall for high-traffic recipes daily to ensure the model continues to meet the target of correctly identifying 80% of high-traffic recipes.
2.	**Recipe Performance Tracking**: Analyze actual user engagement (e.g., clicks, time spent, or subscriptions) for recipes displayed on the homepage. Compare predicted high-traffic recipes against those that achieved high engagement to validate model predictions.
3.	**Periodic Model Reevaluation**: Perform monthly checks to ensure the model’s predictions remain robust, take additional consideration for seasonal/holiday trends.

**Metric Thresholds**
1. **Good Performance**:
   - **Prediction Accuracy**: Between 70%-80%
   - **Precision, Recall, and F1 Score for 'High' Class**: Between 0.7 and 0.8
   - **Precision for 'Other' Class**: Above 68%
   - This range reflects a clear improvement over the baseline while allowing for variability in the early deployment phase.

2. **Optimal Performance**:
   - **Prediction Accuracy**: Above 80%
   - **Precision, Recall, and F1 Score for 'High' Class**: Greater than 0.8
   - **Precision for 'Other' Class**: Above 68%
   - Achieving this level would indicate that the model is fully meeting the project’s objectives by reliably predicting high-traffic recipes and avoiding the inclusion of low-traffic options.

These thresholds are designed to reflect both short-term improvements and long-term goals. The current manual selection process results in 60% accuracy, and the good performance threshold (70%-80%) demonstrates meaningful improvement while accounting for variability during early deployment. Optimal performance (>80%) aligns with the strategic aim of significantly exceeding manual selection capabilities and reliably predicting high-traffic recipes. Furthermore, ensuring precision for the "Other" class remains above 68% helps maintain the business objective of minimizing the recommendation of unpopular recipes, ensuring a balanced and effective approach to achieving increased traffic and subscriptions.
