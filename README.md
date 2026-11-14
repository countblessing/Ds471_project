# Bank Marketing Prediction

**DS471 – Machine Learning \| Academic Group Project**

## Overview

This project predicts whether a bank customer will subscribe to a term
deposit before the current marketing campaign begins. It compares three
binary classification models and evaluates their ability to identify
potential subscribers in an imbalanced dataset.

## Dataset

- Source: [UCI Bank Marketing
  Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- File: `bank-full.csv` (semicolon-separated)
- Records: 45,211
- Original columns: 17 (16 predictors and one target)
- Target: `y` (yes/no)
- Class distribution: 39,922 no (88.3%) and 5,289 yes (11.7%)
- Final input features: 13

Current campaign variables (`duration`, `contact`, `day`, `month`, and
`campaign`) were excluded to avoid relying on information unavailable at
prediction time. Two features were engineered: `has_any_loan` and
`previously_contacted`.

## Machine Learning Workflow

1.  Data cleaning, exploratory data analysis, and feature engineering.
2.  Stratified training (70%), validation (15%), and test (15%)
    splitting.
3.  Model development using Logistic Regression, Random Forest, and a
    Multilayer Perceptron (MLP) Neural Network.
4.  Hyperparameter optimization using GridSearchCV and
    RandomizedSearchCV with stratified five-fold cross-validation on the
    training set.
5.  Training and validation analysis, including neural network learning
    curves from an additional manual early-stopping experiment.
6.  Final test evaluation using accuracy, precision, recall, F1-score,
    confusion matrices, and ROC-AUC.

Preprocessing uses StandardScaler, OneHotEncoder, and model pipelines to
fit transformations on training data. F1-score is the primary selection
metric because of class imbalance.

## Final Test Results

| Model               | Accuracy | Precision | Recall |   F1-score | ROC-AUC |
|---------------------|---------:|----------:|-------:|-----------:|--------:|
| Logistic Regression |   0.7159 |    0.2305 | 0.6116 |     0.3348 |  0.7277 |
| Random Forest       |   0.8108 |    0.3104 | 0.5057 | **0.3847** |  0.7357 |
| Neural Network      |   0.8937 |    0.6565 | 0.1904 |     0.2952 |  0.7358 |

Random Forest was selected based on its highest validation F1-score
(0.3832). It also achieved the highest test F1-score among the evaluated
models. Logistic Regression achieved the highest recall, while the
Neural Network achieved the highest accuracy and precision.

## Repository Structure

- `ML_Project.ipynb` — Complete machine learning workflow, results, and
  visualizations.
- `bank-full.csv` — Dataset used in the project.
- `README.md` — Project documentation.

## How to Run

1.  Download or clone this repository.

2.  Keep `bank-full.csv` in the same directory as `ML_Project.ipynb`.

3.  Open the notebook in Jupyter Notebook or VS Code with a Python
    kernel.

4.  Install the required libraries if necessary:

    ``` bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

5.  Run all notebook cells in order. Hyperparameter optimization may
    take several minutes or longer depending on hardware.

## Conclusion and Future Work

Random Forest provided the best precision-recall balance according to
F1-score. However, its moderate recall and F1-score indicate room for
improvement. Future work could investigate threshold optimization,
additional features, class imbalance handling, and temporal validation.

## Reference

Moro, S., Laureano, R., & Cortez, P. (2011). Using Data Mining for Bank
Direct Marketing: An Application of the CRISP-DM Methodology.
Proceedings of the European Simulation and Modelling Conference (ESM
2011), 117–121. https://hdl.handle.net/1822/14838
