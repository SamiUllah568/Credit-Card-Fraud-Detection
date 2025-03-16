# Credit Card Fraud Detection

### Credit Card Fraud Detection Dataset
[Kaggle Dataset Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)


## Overview
This project focuses on detecting fraudulent credit card transactions using machine learning techniques. The dataset contains 284,807 transactions, with only 492 (0.172%) being fraudulent, presenting a significant class imbalance challenge. The goal is to build a model that effectively identifies fraud while minimizing false positives.

## Key Features
- **Features**: 28 principal components (V1-V28), transaction Time (seconds since first transaction), and Amount.
- **Target**: Class (0: legitimate, 1: fraudulent).

## Challenge
The dataset's extreme imbalance makes accuracy an unreliable metric. Instead, Area Under the Precision-Recall Curve (AUPRC) is prioritized to evaluate model performance.

## Installation
### Dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost tensorflow kaggle
```

### Dataset Download:
```bash
# Set up Kaggle API
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download dataset
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data
```

## Dataset Information
- **Original Size**: 284,807 transactions, 31 features.
- **Preprocessing**:
    - Removed 1,081 duplicates.
    - **Final Size**: 283,726 transactions.
    - No missing values.

## Data Visualization
### Class Distribution
Severe imbalance: 99.83% legitimate vs. 0.17% fraudulent transactions.

### Feature Distributions
Histograms for all features, highlighting differences between classes.

### Correlation Matrix
Minimal correlation between features and target.

## Methodology
### Preprocessing:
- Split data into 80% training, 20% testing.
- Scaled features using StandardScaler.
- Applied SMOTE to balance training data.

### Models Evaluated:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- Artificial Neural Network (ANN) – Best Performer

## Results
### Neural Network Architecture
```python
Sequential([
        Dense(128, activation='relu', input_shape=(30,)),
        Dropout(0.6),
        Dense(64, activation='relu'),
        Dropout(0.5),
        # ... additional layers ...
        Dense(1, activation='sigmoid')
])
```

### Performance Metrics
| Metric                | Training Set | Testing Set |
|-----------------------|--------------|-------------|
| Accuracy              | 99.56%       | 99.56%      |
| Precision (Class 1)   | 0.27         | 0.26        |
| Recall (Class 1)      | 0.96         | 0.86        |
| AUPRC                 | -            | 0.7129      |

### Confusion Matrix (Testing)
```plaintext
[[56414   237]
 [   13    82]]
```

## Conclusion
ANN achieved the highest recall (86%) for fraud detection, critical for minimizing financial losses. Precision remains low due to class imbalance, highlighting the need for further tuning or alternative sampling techniques. The model is saved (neural_network_model.keras).

## Usage
### Train the model:
```python
from tensorflow.keras.models import load_model
model = load_model("neural_network_model.keras")
predictions = (model.predict(X_test) > 0.5).astype(int)
```

### Evaluate performance:
```python
print(classification_report(y_test, predictions))
print("AUPRC:", auc(recall, precision))
```

## License
MIT License