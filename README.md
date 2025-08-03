# Credit Risk Model: A Cost-Sensitive Classification Analysis

This repository contains a comprehensive machine learning project focused on developing and evaluating credit risk prediction models. The primary objective is to accurately classify loan applicants into 'Good' or 'Bad' credit risk categories, with a critical emphasis on minimizing a custom cost function that heavily penalizes False Negatives.

## Project Overview

Credit risk assessment is vital for financial institutions. This project addresses the challenge of predicting loan defaults, considering both inherent class imbalance in the dataset and the asymmetric financial implications of misclassification errors. Specifically, a False Negative (approving a bad loan) is considered five times more costly than a False Positive (rejecting a good loan).

The project systematically covers:
- Data loading and initial preparation.
- Extensive Exploratory Data Analysis (EDA) to understand data distributions and relationships.
- Robust data preprocessing and feature engineering.
- Training and evaluation of multiple machine learning classifiers.
- Advanced optimization techniques, including class weight tuning, hyperparameter optimization, and custom probability threshold adjustment, all aimed at minimizing the defined business cost.

## Dataset

The project utilizes the **German Credit Data (german.data)**, a widely used dataset for credit risk modeling.

- **Instances:** 1000 loan applicants
- **Features:** 20 distinct features covering applicant demographics, financial status, and credit history
- **Target Variable:** `risk` (binary: 0 for 'Good credit', 1 for 'Bad credit')
- **Class Distribution:** 
  - 70% 'Good Risk' (Class 0)
  - 30% 'Bad Risk' (Class 1)

## Problem Statement and Custom Cost Function

The core problem is a binary classification task to predict credit risk. The model's performance is driven by a custom cost function designed to reflect real-world financial penalties:

### Cost Function:

Cost = (5 × False Negatives) + (1 × False Positives)


This function prioritizes minimizing False Negatives (Type II error – approving a loan to a defaulting customer) due to their higher financial impact compared to False Positives (Type I error – rejecting a good loan).

## Methodology

The project follows a structured machine learning pipeline:

### 1. Data Loading and Initial Preparation:
- Loading the dataset and assigning descriptive column names.
- Mapping original alphanumeric codes in categorical features to human-readable labels.
- Transforming the target variable for standard binary classification.

### 2. Exploratory Data Analysis (EDA):
- Visualizing distributions of individual features (pie charts for categorical, box plots, and histograms for numerical).
- Analyzing relationships between features and the risk target (grouped bar charts, density plots).
- Generating correlation matrices and bar plots to understand feature inter-relationships and their correlation with risk.

### 3. Data Preprocessing and Feature Engineering:
- Categorizing features into ordinal, one-hot, and numerical types.
- Applying `OrdinalEncoder` for ordered categorical features and `OneHotEncoder` (with drop='first') for nominal ones.
- Using `ColumnTransformer` to manage diverse preprocessing steps.
- Splitting data into training and testing sets using `train_test_split` with `stratify=y` to maintain class balance.
- Scaling numerical features using `StandardScaler` to prevent feature dominance.

### 4. Model Training and Evaluation:
- Implementing a custom scoring function based on the defined cost metric.
- Evaluating various classification algorithms:
  - Logistic Regression
  - K-Nearest Neighbors
  - Gaussian Naive Bayes
  - Support Vector Classifier (SVC)
  - Decision Tree Classifier
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - XGBoost Classifier
- For each model, a systematic optimization process was followed:
  - **Baseline Evaluation:** Initial performance assessment.
  - **Class Weight Tuning:** Iterating through different class weights to penalize misclassifications of the minority class more heavily.
  - **Hyperparameter Optimization:** Using `GridSearchCV` (for Logistic Regression, SVC, Decision Tree) and `RandomizedSearchCV` (for Random Forest, Gradient Boosting, XGBoost) with stratified cross-validation, guided by the custom cost scorer.
  - **Optimal Probability Threshold Adjustment:** Fine-tuning the classification threshold on predicted probabilities to achieve the lowest possible custom cost.

## Key Results

The rigorous optimization process significantly reduced the custom misclassification cost across various models. The most impactful step was often the adjustment of the classification probability threshold, which directly aligned the model's decision boundary with the business's asymmetric costs.

| Model                    | Initial Custom Cost | Custom Cost after Class Weights / Initial Tuning | Custom Cost after Hyperparameter Tuning | Final Custom Cost (Optimal Threshold) | Optimal Threshold |
|--------------------------|---------------------|--------------------------------------------------|----------------------------------------|---------------------------------------|-------------------|
| Logistic Regression       | 202                  | 115                                               | 117                                     | 115                                 | 0.50              |
| K-Nearest Neighbors       | 268                 | 218                                               | 218                                     | N/A                                  | N/A               |
| Naive Bayes              | 197                 | 194                                               | 194                                     | N/A                                  | N/A               |
| Support Vector Classifier | 234                  | 113                                               | 117                                     | 115                                 | 0.27              |
| Decision Tree Classifier  | 224                 | 182                                               | 167                                     | 132                                  | 0.14              |
| Random Forest Classifier  | 217                 | 191                                               | 161                                     | 129                                  | 0.40              |
| Gradient Boosting Classifier | 183               | N/A                                              | 185                                     | 129                                  | 0.14              |
| XGBoost Classifier        | 187                 | N/A                                              | 182                                     | 117                                  | 0.04              |

## Key Findings

- **Best Performers:**
  - **XGBoost** emerged as the top-performing model with the **lowest final custom cost** of **117** after optimal threshold adjustment. This was the best result achieved in this experiment.
  - **Random Forest** also performed very well, achieving a final custom cost of **129** after threshold adjustment, showing its strong ability to minimize the misclassification cost.

- **Logistic Regression:**
  - **Logistic Regression** showed a good balance of performance, with an initial custom cost of **202**, reducing to **115** after class weight tuning. The final custom cost remained **115**, indicating it performed effectively despite being a simpler model compared to the ensemble methods.

- **Support Vector Classifier (SVC):**
  - **SVC** improved from an initial custom cost of **234** to **113** after class weight tuning, and the final cost after hyperparameter tuning was **115**. The threshold optimization also helped it reach similar performance to **Logistic Regression**.

- **Decision Tree Classifier:**
  - **Decision Tree** achieved a **significant improvement** in custom cost reduction from an initial **224** to **132** after optimal threshold adjustment. This showed that decision trees can be very effective with proper tuning, especially for cost-sensitive tasks.

- **K-Nearest Neighbors (KNN) and Naive Bayes:**
  - **K-Nearest Neighbors** performed the worst, with an initial cost of **268** and no final custom cost available after threshold optimization. Despite some improvement with class weight tuning, it did not perform well in minimizing misclassification costs.
  - **Naive Bayes** performed slightly better, with an initial cost of **197**, reducing to **194** after class weight tuning. However, it still showed relatively poor performance compared to other models.

- **Threshold Adjustment Impact:**
  - **Optimal threshold adjustment** was critical for most models capable of outputting probabilities, helping to reduce the custom cost. Models like **Logistic Regression**, **SVC**, **Random Forest**, and **XGBoost** all benefited significantly from this step, aligning the decision boundary to reduce False Negatives (which are penalized more heavily in the custom cost function).

- **Key Takeaways:**
  - The best models for this cost-sensitive classification task were **XGBoost** (final cost of **117**) and **Random Forest** (final cost of **129**).
  - **Logistic Regression** and **Support Vector Classifier** performed competitively with final costs of **115**, showing that simpler models can still be effective with appropriate tuning.
  - **K-Nearest Neighbors** and **Naive Bayes** did not perform well in this scenario, reinforcing the need for more sophisticated models when dealing with imbalanced datasets and asymmetric cost functions.

## Future Work

To further enhance this credit risk model, consider the following:
- **Advanced Ensemble and Stacking:** Explore combining predictions from top-performing models for potential further cost reduction.
- **Cost-Sensitive Learning Algorithms:** Investigate specialized algorithms that inherently integrate misclassification costs into their training objectives.
- **More Advanced Feature Engineering:** Create interaction terms, polynomial features, or use dimensionality reduction techniques.
- **Robustness Analysis:** Conduct extensive testing using multiple random train-test splits or nested cross-validation.
- **Model Interpretability (XAI):** Utilize techniques like SHAP or LIME to understand feature contributions and foster trust.
- **Dynamic Cost Adjustment:** Design mechanisms for dynamically adjusting misclassification costs and retraining models based on evolving economic conditions.


