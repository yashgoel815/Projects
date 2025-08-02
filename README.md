# Cost-Sensitive Credit Risk Classification Model

This repository presents a comprehensive machine learning project focused on developing a **credit risk classification model** to classify loan applicants into 'Good' or 'Bad' credit risk categories. The project places particular emphasis on minimizing a **custom business cost function**, which reflects the financial implications of misclassifications. 

The dataset used is the publicly available **German Credit Data** containing 1000 instances and 20 features.

## Key Challenges Addressed

- **Class Imbalance**: The dataset exhibits a significant imbalance, with 70% 'Good' credit instances and 30% 'Bad' credit instances. Without addressing this imbalance, models could be biased towards the majority class.
  
- **Asymmetric Misclassification Costs**: False Negatives (misclassifying a 'Bad' applicant as 'Good') are five times more costly than False Positives (misclassifying a 'Good' applicant as 'Bad'). This reflects the severe financial consequences of approving loans to high-risk individuals.

## Methodology

### Data Loading and Initial Preparation
- Loaded the `german.data` dataset and assigned meaningful column names.
- Performed initial data quality checks (no missing values or duplicates).
- Mapped categorical features and the target variable to more interpretable labels.

### Exploratory Data Analysis (EDA)
- Analyzed distributions of categorical and numerical features.
- Visualized relationships between individual features and credit risk.
- Confirmed a 70:30 class imbalance in the target variable.

### Data Preprocessing and Feature Engineering
- Categorical features were categorized into ordinal, one-hot, and numerical types.
- Applied `OrdinalEncoder` and `OneHotEncoder` using `ColumnTransformer`.
- Split the data into training and testing sets, ensuring stratification to preserve class proportions.
- Standardized numerical features using `StandardScaler`.

### Model Training and Evaluation
- Implemented a custom cost function: Cost = (5 * False Negatives) + (1 * False Positives)


Evaluated a range of classification algorithms:
- Logistic Regression
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Support Vector Classifier
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier

**Optimization Process**:
- **Baseline Performance**: Initial evaluation with default parameters.
- **Class Weight Optimization**: Applied grid search to optimize `class_weight` parameters.
- **Hyperparameter Tuning**: Used `GridSearchCV` or `RandomizedSearchCV` with a custom scoring function to minimize the cost.
- **Optimal Threshold Tuning**: Iterated through probability thresholds to find the optimal decision boundary.

## Key Results

- **Significant Cost Reduction**: After optimization, models reduced the custom cost from an initial value to much lower values across different stages of tuning.

- **Optimal Threshold**: The optimal thresholds vary by model and were found through a fine-tuning process to minimize False Negatives and control False Positives.

| Model                        | Initial Custom Cost | Custom Cost after Class Weights / Initial Tuning | Custom Cost after Hyperparameter Tuning | Final Custom Cost (Optimal Threshold) | Optimal Threshold |
|------------------------------|---------------------|--------------------------------------------------|----------------------------------------|---------------------------------------|-------------------|
| Logistic Regression           | 202                 | 115 (w={0:0.1, 1:0.3})                          | 117                                    | 115                                   | 0.50              |
| K-Nearest Neighbors           | 268                 | 218 (k=1, metric=euclidean)                      | 218                                    | N/A                                   | N/A               |
| Naive Bayes                   | 197                 | 194 (alpha=0.1)                                 | 194                                    | N/A                                   | N/A               |
| Support Vector Classifier     | 234                 | 113 (w={0:0.6, 1:1.8})                          | 117                                    | 115                                   | 0.27              |
| Decision Tree Classifier      | 224                 | 182 (w={0:0.7, 1:0.9})                          | 167                                    | 132                                   | 0.14              |
| Random Forest Classifier      | 217                 | 191 (w={0:2.0, 1:1.1})                          | 161                                    | 129                                   | 0.40              |
| Gradient Boosting Classifier  | 183                 | N/A                                              | 185                                    | 129                                   | 0.14              |
| XGBoost Classifier            | 187                 | N/A                                              | 182                                    | 117                                   | 0.04              |

- **Impact of Threshold Tuning**: The optimal threshold tuning helped significantly reduce the costly False Negatives by allowing a controlled increase in False Positives.


## Technologies Used

- **Python**: Programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations
- **Scikit-learn**: Machine learning algorithms, preprocessing, model selection, and evaluation
- **XGBoost**: Gradient Boosting implementation
- **Matplotlib**, **Seaborn**: Data visualization

## Future Work

- **Ensemble Techniques**: Explore advanced ensemble methods (e.g., stacking) for potentially further cost reduction.
- **Cost-Sensitive Learning**: Investigate specialized cost-sensitive learning algorithms.
- **Feature Engineering**: Conduct more extensive feature engineering and feature selection.
- **Probability Calibration**: Implement probability calibration techniques (e.g., Platt Scaling) for more reliable probability estimates.
- **Cross-Validation**: Perform robust cross-validation to ensure model generalization.
- **Explainable AI (XAI)**: Integrate explainable AI methods such as SHAP and LIME for better model interpretability.
- **Dynamic Model Updates**: Develop mechanisms for dynamic cost adjustment and model retraining in production environments.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **German Credit Data**: Dataset for this project. You can access it from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data)).
- **Scikit-learn**: For providing the machine learning tools and algorithms.
- **XGBoost**: For the gradient boosting classifier implementation.
- **Matplotlib/Seaborn**: For visualizations and plotting.

