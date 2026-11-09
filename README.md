# UCI Bank Marketing Prediction - Machine Learning Project (DS-471)

## Project Overview
This project presents an end-to-end machine learning workflow applied to the **UCI Bank Marketing dataset** (`id=222`). The goal is to predict whether a client will subscribe to a term deposit (`yes`/`no`) based on demographic, economic, and direct marketing features.

## Workflow Stages
1. **Data Cleaning & EDA:** Handling implicit missing values, encoding categorical variables (One-Hot Encoding), applying feature scaling (`StandardScaler`), and exploring feature correlations.
2. **Data Splitting:** A stratified 3-way split consisting of 70% Training, 15% Validation, and 15% Test sets to prevent data leakage and preserve class distribution.
3. **Model Selection & Development:** Built and trained three classifiers:
   - Logistic Regression (Linear Baseline)
   - Random Forest Classifier (Ensemble Model)
   - Multi-Layer Perceptron / MLPClassifier (Mandatory Neural Network)
4. **Training Analysis & Hyperparameter Optimization:** Applied L2 regularization, early stopping, and tracked the training loss curve for the Neural Network (converged at Iteration 37 with loss `0.1745`).
5. **Evaluation & Model Comparison:** Evaluated model performance on unseen test data using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

## Final Results Summary
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.8994 | 0.6318 | 0.3354 | 0.4382 | 0.8954 |
| Random Forest | 0.9028 | 0.6463 | 0.3733 | 0.4732 | 0.9106 |
| **Neural Network (MLP)** | **0.9017** | **0.5960** | **0.4931** | **0.5397** | **0.9106** |

- **Best Model for Business Deployment:** The **MLP Neural Network** achieved the highest **Recall (0.4931)** and **F1-Score (0.5397)**, making it the most suitable model for maximizing campaign conversion rates.

## Technologies Used
- Python 3.x
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- UCI Machine Learning Repository API (`ucimlrepo`)
