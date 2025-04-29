# Content Updates for Lecture 5

1. ✅ Add New Sections:
   - Automated feature engineering
     - Time series features (review)
     - Feature tools library
     - Domain-specific derivations
   - Model interpretation with tree-based models
     - SHAP values for feature importance
     - eli5 for model inspection
     - Interpreting feature interactions
   - Practical data preparation
     - OneHotEncoder for categorical variables
     - Handling imbalanced data with SMOTE
     - When/how to combine techniques

2. Demo Structure (in demo/01_xxx, demo/02_yyy):
   - Demo 1 (⅓): Binary classification with logistic regression (diabetes prediction, synthetic data, walkthrough model evaluation criteria)
   - Demo 2 (⅔): Sensor classification with derived features
     - Feature engineering from time series
     - RandomForest vs XGBoost comparison
     - Model interpretation with SHAP
   - Demo 3 (end): Imbalanced classification
     - Categorical feature handling
     - SMOTE for imbalanced classes
     - Model interpretation with eli5
     - (Optional) Additional feature engineering

3. ✅ Move to Neural Networks Lecture (if not moved already):
   - Image classification demo (animals)
   - EMNIST classification
   - StandardScaler and normalization
   - Hyperparameter tuning
   - GPU acceleration
   - Deep learning architectures

4. General Improvements:
   - Add speaking notes as HTML comments
   - For each topic include: conceptual overview, method reference card (function, params, definitions), short example (minimal context)
   - More beginner-friendly explanations
   - Add comprehension checkpoints
