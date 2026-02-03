# Lecture 05 Dataset Plan

## Overview

Replace synthetic `make_classification()` data with established datasets. Use MedMNIST for instructor-led demos (health context can be explained), Fashion-MNIST for self-directed assignment (intuitive classes).

## Demos

| Demo | Dataset | Type | Samples | Classes | Install | Notes |
|:---|:---|:---|---:|---:|:---|:---|
| **Demo 1** | `load_breast_cancer()` | Tabular | 569 | 2 (malignant/benign) | sklearn built-in | Real tumor measurements, 30 features |
| **Demo 2** | BreastMNIST or PneumoniaMNIST | Image 28×28 | 780 / 5K | 2 (binary) | `pip install medmnist` | Model comparison + cross-validation |
| **Demo 3** | DermaMNIST | Image 28×28 | 10K | 7 (skin lesions) | `pip install medmnist` | Natural class imbalance, SMOTE demo |

### Demo 1: Binary Classification with Breast Cancer Data

**Goal**: Introduce classification workflow with real tabular medical data

**Methods covered**:
- `train_test_split` with `stratify`
- `LogisticRegression`
- `confusion_matrix`, `classification_report`
- `roc_curve`, `roc_auc_score`

**Code pattern**:
```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
# 0 = malignant, 1 = benign
```

### Demo 2: Model Comparison with MedMNIST

**Goal**: Compare classifiers using cross-validation on medical images

**Dataset options** (pick one):
- **PneumoniaMNIST**: 5,856 samples, binary (normal/pneumonia), chest X-rays
- **BreastMNIST**: 780 samples, binary (malignant/benign), breast ultrasound

**Methods covered**:
- `cross_val_score`, `StratifiedKFold`
- `LogisticRegression`, `RandomForestClassifier`, `XGBClassifier`
- `StandardScaler` (fit on train only)
- SHAP for interpretation

**Code pattern**:
```python
import medmnist
from medmnist import PneumoniaMNIST

train_dataset = PneumoniaMNIST(split='train', download=True)
test_dataset = PneumoniaMNIST(split='test', download=True)

# Images are 28x28, flatten to 784 for sklearn
X_train = train_dataset.imgs.reshape(len(train_dataset), -1)
y_train = train_dataset.labels.squeeze()
```

### Demo 3: Imbalanced Classification with DermaMNIST

**Goal**: Handle class imbalance with real skin lesion data

**Dataset**: DermaMNIST - 10,015 samples, 7 classes (skin lesion types), naturally imbalanced

**Methods covered**:
- `OneHotEncoder` (if adding synthetic categorical)
- `SMOTE` from `imblearn`
- `eli5` for model interpretation
- Focus on recall for minority classes

**Code pattern**:
```python
from medmnist import DermaMNIST

train_dataset = DermaMNIST(split='train', download=True)
# Class distribution is imbalanced - show with value_counts
```

## Assignment

| Component | Dataset | Type | Samples | Classes | Install |
|:---|:---|:---|---:|---:|:---|
| **Assignment** | Fashion-MNIST | Image 28×28 | 70K | 10 (clothing) | keras built-in |

### Why Fashion-MNIST for Assignment

1. **Intuitive classes**: T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot
2. **Self-debugging**: Students can visually verify predictions ("that's clearly a shoe, not a bag")
3. **Zero install issues**: Built into `keras.datasets`, no compatibility problems
4. **Well-documented**: Extensive tutorials, examples, benchmarks available
5. **Appropriate difficulty**: 10-class is harder than binary, good stretch

### Assignment Task

Binary classification: **Clothing vs Footwear**

- Class 0 (Clothing): T-shirt, trouser, pullover, dress, coat, shirt (labels 0,1,2,3,4,6)
- Class 1 (Footwear): Sandal, sneaker, ankle boot (labels 5,7,9)
- Bag (label 8): Exclude OR assign to one category

**Code pattern**:
```python
from tensorflow.keras.datasets import fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Create binary labels
clothing_labels = [0, 1, 2, 3, 4, 6]
footwear_labels = [5, 7, 9]
# Exclude bags (label 8) or assign to clothing

y_train_binary = np.isin(y_train, footwear_labels).astype(int)
```

**Required outputs**:
- `output/cv_results.csv` - cross-validation scores
- `output/test_results.csv` - final metrics
- `output/confusion_matrix.png`
- `output/roc_curve.png`

## Dependencies

### Demo requirements.txt
```
pandas
numpy
scipy
matplotlib
seaborn
scikit-learn
xgboost
shap
medmnist
imblearn
eli5
jupyter
jupytext
```

### Assignment requirements.txt
```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
tensorflow>=2.13.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## Migration Checklist

- [ ] Demo 1: Rewrite with `load_breast_cancer()`
- [ ] Demo 2: Rewrite with PneumoniaMNIST or BreastMNIST + cross-validation + SHAP
- [ ] Demo 3: Rewrite with DermaMNIST + SMOTE + eli5
- [ ] Assignment: Rewrite with Fashion-MNIST (clothing vs footwear)
- [ ] Update demo/requirements.txt to add `medmnist`
- [ ] Update assignment/requirements.txt to add `tensorflow`
- [ ] Regenerate .ipynb files from .md
- [ ] Test all demos execute without error
- [ ] Test assignment with solution, verify tests pass
