---
lecture_number: 05
pdf: true
---

# Classification: Putting a label on things

- Quick example
- Classification model types and how to evaluate them
- How things can go wrong…
- … and how to fix it
- Hands-on with code

## Preamble

1. Example analytical technical interview - [Analytics Technical Interview](Analytics%20Technical%20Interview.md)
2. Example interview take-home - [https://github.com/christopherseaman/five_twelve](https://github.com/christopherseaman/five_twelve)
3. Data sources available on Physionet (some require registration) - [https://physionet.org/about/database/](https://physionet.org/about/database/)

## `git merge` conflicts

1. Working in branches for each exercise
2. Save to branch and sync (discarding commits)
3. (_Optional_) Add files from branch back to `main` through a Pull Request

![Git branches diagram](media/git_branches.png)

> [!info] Git merge conflicts | Atlassian Git Tutorial  
> What is a git merge conflict?  
> [https://www.atlassian.com/git/tutorials/using-branches/merge-conflicts](https://www.atlassian.com/git/tutorials/using-branches/merge-conflicts)  

## On the perils of ChatGPT

Not exactly what Linus was talking about, but the quote remains relevant…

![Meme about AI](media/wpvtr5pmskfc1.png.webp)

## 🤖**Accelerate Your Models with GPUs**

The models we'll build today require significant computational power, which might make them run slowly on your laptop. **Don't panic!** There are free cloud services you can use to leverage GPU-based computing:

- [**Google Colab**](https://colab.research.google.com) - easiest to use with excellent GitHub integration, but completely public (okay for today, but **NEVER USE COLAB WITH SENSITIVE/PHI DATA**)
- [**Paperspace**](https://www.paperspace.com) - (my weapon of choice) provides customizable private virtual computing with GPUs, but that customization comes at the cost of added complexity. Easy-ish to use, but not as easy as Colab
- **UCSF's Wynton** - File a ticket with IT to get access

### Example speed-up

From training the "Which animal is this" in the "Hands-on practice" section below

```Shell
## 2018 Macbook Pro
Epoch 1/25
63/63 [==============================] - 1177s 18s/step - loss: 0.7522 - accuracy: 0.5904 - val_loss: 0.8942 - val_accuracy: 0.4998

## Google Colab GPU
Epoch 1/25
63/63 [==============================] - 150s 2s/step - loss: 0.7323 - accuracy: 0.6080 - val_loss: 0.9307 - val_accuracy: 0.4998

## Paperspace RTX-5000
Epoch 1/25
63/63 [==============================] - 64s 989ms/step - loss: 0.7348 - accuracy: 0.5988 - val_loss: 0.8696 - val_accuracy: 0.4998
```

## 💥Crash course in classification

![XKCD classification](media/xkcd_classification.png)

The building blocks of ML are algorithms for **regression** and **classification:**

- **Regression**: predicting continuous quantities
- **Classification**: predicting _discrete class labels_ (categories)

### **Classification methods**

Classification algorithms aim to learn a function that maps input features to class labels. The most popular classification methods are:

- **Logistic Regression**: a simple linear model that models the probability of each class based on the input features. It's easy to interpret and works well for binary classification problems.
- **Decision Trees**: builds a tree-like model that maps features to class labels. It's easy to interpret, but prone to overfitting.
- **Random Forest**: an ensemble model that builds multiple decision trees and averages their predictions. It's more accurate than a single decision tree and less prone to overfitting.
- **Support Vector Machines (SVM)**: constructs a hyperplane that separates the classes with the maximum margin. It works well for high-dimensional data.
- **Naive Bayes**: applies Bayes' theorem with strong independence assumptions between the features. It's easy to train and performs well for small datasets.
- **Neural Networks**: models the mapping between inputs and outputs using an interconnected network of nodes. It can capture complex, non-linear relationships between features.

- Some links to dive deeper:
    - A nice tour of methods: [**https://github.com/bagheri365/ML-Models-for-Classification**](https://github.com/bagheri365/ML-Models-for-Classification)
    - [**Cancer classification**](https://www.kaggle.com/code/nandita711/cancer-classification-eda-pca-random-forest) (Kaggle)
    - [**Comparison of XGBoost, Random Forest, and Nomograph for Prediction of Disease Severity**](https://www.frontiersin.org/articles/10.3389/fcimb.2022.819267/full)
    - [**Prediction Method for Hypertension**](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6963807/) (Diagnostics Journal)
    - [**Guide to Predictive Lead Scoring using ML**](https://towardsai.net/p/l/a-guide-to-predictive-lead-scoring-using-machine-learning) (Towards AI)
    - [**True end-to-end ML example: Lead Scoring**](https://towardsdatascience.com/a-true-end-to-end-ml-example-lead-scoring-f5b52e9a3c80) (Towards Data Science)

### Model evaluation

There are many more classification approaches than data scientists, so choosing the best one for your application can be daunting. Thankfully, all of them output predicted classes for each data point. We can use this similarity to define objective performance criteria based on how often the predicted class matches the underlying truth.

I get in trouble with the data science police if I don't include something about confusion matrices:

![Evaluation metrics](media/evaluation.png)

- **Precision** (Positive Predictive Value) = $\frac{TP}{TP + FP}$

    > _How well it performs when it predicts positive_

- **Recall** (Sensitivity, True Positive Rate) = $\frac{TP}{TP+FN}$

    > _How well it performs among actual positives_

- **Accuracy** = **$\frac{(TP+TN)}{(TP+FP+FN+TN)}$**

    > _How well it performs among all known classes_

- **F1 score** = $2 \times \frac{Recall * Precision}{Recall + Precision}$

    > _Balanced score for overall model performance_

- **Specificity** (Selectivity, True Negative Rate) = $\frac{TN}{TN + FP}$

    > _Similar to_ **Recall**, _how well it performs among actual negatives_

- **Miss Rate** (False Negative Rate) = $\frac{FN}{TP + FN}$

    > _Proportion of positives that were incorrectly classified, good measure when missing a positive has a high cost_

- **Receiver-Operator Curve (ROC Curve) and Area Under the Curve (AUC)**

    > _Plot the True Positive vs. False Positive rates, which provides a scale-invariant measure of performance. A random model on balanced class data will have a score of 0.5, while a perfect model will always have a score of 1_

![AUROC example](media/auroc.png)

### ROC curve

An **ROC curve** (**receiver operating characteristic curve**) is a graph showing the performance of a classification model at all classification thresholds. This curve plots two parameters:

- True Positive Rate
- False Positive Rate

**True Positive Rate** (**TPR**) is a synonym for recall and is therefore defined as follows:

$$TPR = TP/(TP+FN)$$

**False Positive Rate** (**FPR**) is defined as follows:

$$FPR = FP/(FP+TN)$$

An ROC curve plots TPR vs. FPR at different classification thresholds. Lowering the classification threshold classifies more items as positive, thus increasing both False Positives and True Positives. The following figure shows a typical ROC curve.

[![ROC Curve](https://developers.google.com/static/machine-learning/crash-course/images/ROCCurve.svg)](https://developers.google.com/static/machine-learning/crash-course/images/ROCCurve.svg)

**Figure 4. TP vs. FP rate at different classification thresholds.**

To compute the points in an ROC curve, we could evaluate a logistic regression model many times with different classification thresholds, but this would be inefficient. Fortunately, there's an efficient, sorting-based algorithm that can provide this information for us, called AUC.

### AUC: Area Under the ROC Curve

**AUC** stands for "Area under the ROC Curve." That is, AUC measures the entire two-dimensional area underneath the entire ROC curve (think integral calculus) from (0,0) to (1,1).

[![AUC](https://developers.google.com/static/machine-learning/crash-course/images/AUC.svg)](https://developers.google.com/static/machine-learning/crash-course/images/AUC.svg)

**Figure 5. AUC (Area under the ROC Curve).**

AUC provides an aggregate measure of performance across all possible classification thresholds. One way of interpreting AUC is as the probability that the model ranks a random positive example more highly than a random negative example. For example, given the following examples, which are arranged from left to right in ascending order of logistic regression predictions:

[![AUC Predictions](https://developers.google.com/static/machine-learning/crash-course/images/AUCPredictionsRanked.svg)](https://developers.google.com/static/machine-learning/crash-course/images/AUCPredictionsRanked.svg)

**Figure 6. Predictions ranked in ascending order of logistic regression score.**

AUC represents the probability that a random positive (green) example is positioned to the right of a random negative (red) example.

AUC ranges in value from 0 to 1. A model whose predictions are 100% wrong has an AUC of 0.0; one whose predictions are 100% correct has an AUC of 1.0.

AUC is desirable for the following two reasons:

- AUC is **scale-invariant**. It measures how well predictions are ranked, rather than their absolute values.
- AUC is **classification-threshold-invariant**. It measures the quality of the model's predictions irrespective of what classification threshold is chosen.

However, both these reasons come with caveats, which may limit the usefulness of AUC in certain use cases:

- **Scale invariance is not always desirable.** For example, sometimes we really do need well calibrated probability outputs, and AUC won't tell us about that.
- **Classification-threshold invariance is not always desirable.** In cases where there are wide disparities in the cost of false negatives vs. false positives, it may be critical to minimize one type of classification error. For example, when doing email spam detection, you likely want to prioritize minimizing false positives (even if that results in a significant increase of false negatives). AUC isn't a useful metric for this type of optimization.

- [How to evaluate classification models](https://www.edlitera.com/en/blog/posts/evaluating-classification-models) (edlitera)

## 🦾 LIVE DEMO!

[demo/01_diabetes_prediction.md](demo/01_diabetes_prediction.md)

### Supervised vs. unsupervised

There are two-ish overarching categories of classification algorithms: **supervised** and **unsupervised**. There are many possible approaches in each category, and some that work well in both (deep learning, for example).

- **Supervised** - uses labeled datasets with known classes for the data points
- **Unsupervised** - uses unlabeled data to uncover organizational patterns
- **Semi-supervised** - some data with labels is used to extract relevant features, while others without can amplify that signal; e.g., medical images (x-ray, CT)

![[unsupervised.png]]

### Supervised models

To fairly evaluate each model, we must **test** its performance on different data than it was **train**ed on. So we split our dataset into two partitions: **test** and **train**:

- **Train** - the model is built using this data, which includes class labels
- **Test** - the model is tested using this data, withholding class labels

#### Quick supervised model review

Let's look at a few tools that you should get a lot of use out of:

- **Logistic Regression** shouldn't be overlooked! It's not as new as some other models, but it's simple and works.
- **Random Forest** is an ensemble model that makes many decision trees using bagging, then takes a simple vote across them to assign a class
- **XGBoost** is another ensemble and arguably the most widely used (and useful) algorithm in tabular ML (it can do regression, classification, and julienne fries!)
- **Deep Learning** uses artificial neural networks with multiple layers to learn complex patterns from data. These models have performed well in a variety of tasks: image recognition, speech recognition, and natural language processing.

    _Deep Learning models may also be used in unsupervised settings_

#### Logistic regression

Logistic regression works similarly to linear regression but uses a sigmoid curve that squeezes our straight line into an S-curve.

![Linear vs logistic regression](media/lin_vs_log.png)

Additionally, it uses **log loss** in place of our usual mean-squared error cost function. This provides a convex curve for approximating variable weights using gradient descent.

![Approximation optimization](media/approx_optimization.png)

- [Logistic regression](https://christophm.github.io/interpretable-ml-book/logistic.html) (interpretable ml)
- [Logistic Regression using Gradient descent](https://www.kaggle.com/general/192255) (kaggle)

#### Random forest

Each of the steps can be tweaked, but the general flow goes:

1. **Bagging** - create _k_ random samples from the data set
2. **Grow trees** - individual decision trees are constructed by choosing the best features and cutpoints to separate the classes
3. **Classify** - instances are run through all trees and assigned a class by majority vote

![Bagging diagram](media/bagging.png)

#### XGBoost

XGBoost stands for **Extreme Gradient Boosting**. Like other tree algorithms, XGBoost considers each instance with a series of `if` statements, resulting in a leaf with associated class assignment scores. Where XGBoost differs is that it uses gradient boosting to focus on weak-performing areas of the previous tree.

## Boosted trees and gradient boosting

> [!info] A Visual Guide to Gradient Boosted Trees  
> An intuitive visual guide and video explaining GBT and the MNIST database  
> [https://towardsdatascience.com/a-visual-guide-to-gradient-boosted-trees-8d9ed578b33](https://towardsdatascience.com/a-visual-guide-to-gradient-boosted-trees-8d9ed578b33)  

> [!info] Introduction to Boosted Trees — xgboost 2.0.3 documentation  
> XGBoost stands for "Extreme Gradient Boosting", where the term "Gradient Boosting" originates from the paper Greedy Function Approximation: A Gradient Boosting Machine, by Friedman.  
> [https://xgboost.readthedocs.io/en/stable/tutorials/model.html](https://xgboost.readthedocs.io/en/stable/tutorials/model.html)

- **Boosting** - sequentially choosing models by minimizing errors from previous models while increasing the influence of high-performing models; i.e., each model tries to improve where the last was wrong
- **Gradient boosting** - a stagewise additive algorithm sequentially adding trees to improve performance measured by a **loss function** until some threshold is met. It's a greedy algorithm prone to overfitting but often proves useful when focused on poor-performing areas

![XGBoost diagram](media/xgboost.png)

- [XGBoost vs Random Forest](https://medium.com/geekculture/xgboost-versus-random-forest-898e42870f30) (geek culture)
- [Interpretable machine learning with XGBoost](https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27) (towardsdatascience)

### Deep learning

**Deep learning** is a subfield of machine learning that uses artificial neural networks with multiple layers to learn complex patterns from data. These models use back-propagation to adjust the weights in each layer during training, allowing them to model very large and complex datasets.

Deep learning models are especially useful for handling large datasets with high dimensionality, and they can be used for both supervised and unsupervised learning tasks. However, they often require a large amount of data and computation power to train effectively.

These models have performed well in a variety of tasks such as image recognition, speech recognition, and natural language processing.

- **Artificial neural networks** - a computational model inspired by biological neural networks that learn by adjusting the weights between neurons through training data
- **Deep neural networks** - an artificial neural network with more than one hidden layer; these additional layers enable the model to learn more complex patterns from the input data
- **Convolutional neural networks** - a type of deep neural network designed for image and video recognition tasks that use convolutional layers to detect features in the input data
- **Recurrent neural networks** - a type of deep neural network designed for sequence data that uses recurrent connections to remember previous inputs and outputs
- **Popular frameworks** - TensorFlow, PyTorch, and Keras are commonly used deep learning frameworks for building and training deep learning models. Each framework maintains a list of tutorials/examples for getting started (and plenty more on the web + youtube):
    - **Keras**
        - [https://keras.io/getting_started/](https://keras.io/getting_started/)
        - _Deep Learning with Python_ (free pdf)
    - **Pytorch**
        - [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)
        - [https://github.com/ritchieng/the-incredible-pytorch](https://github.com/ritchieng/the-incredible-pytorch)
    - **Tensorflow**
        - [https://www.tensorflow.org/tutorials/quickstart/beginner](https://www.tensorflow.org/tutorials/quickstart/beginner)
        - Tensorflow [https://github.com/tensorflow/examples](https://github.com/tensorflow/examples)
    - **JAX**
        - [https://jax.readthedocs.io/en/latest/notebooks/quickstart.html](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)
        - [https://github.com/gordicaleksa/get-started-with-JAX](https://github.com/gordicaleksa/get-started-with-JAX)

### Unsupervised models

Unsupervised models come in a few flavors:

- **Clustering**: grouping points based on similarities/differences; e.g., proximity and separability of data, market segmentation, image compression
    - K-means (and Fuzzy K-means)
    - Hierarchical clustering (e.g., BIRCH)
    - Gaussian mixture
    - Affinity Propagation
    - Anomaly detection
        - Isolation Forest
        - Local Outlier Factor
        - Min Covariant Determinant
- **Association**: reveals relationships between variables; e.g., A goes up and B goes down, people who buy X also buy Y
    - Apriori
    - Equivalence Class Transformation (eclat)
    - Frequent-Pattern (F-P) Growth
- **Dimensionality** reduction: reduces the inputs to a smaller size while attempting to preserve predictive power; e.g., removing noise and collinearity
    - Principal Component Analysis
    - Manifold Learning — LLE, Isomap, t-SNE
    - Autoencoders

Links to learn more:

- Unsupervised Learning: Algorithms and Examples ([altexsoft](https://www.altexsoft.com/blog/unsupervised-machine-learning/))

### Topics cut for time

- Bias and fairness: How models can produce biased results and unfairly disadvantage certain groups of people. This could include a discussion on how to detect and mitigate bias in classification models.
- Interpretability and explainability: How to interpret and explain the decisions made by classification models. This could include a discussion on the methods used to create interpretable models, such as decision trees and rule-based systems.
- Handling imbalanced data: Techniques for dealing with datasets where one class is significantly more prevalent than others. This could include a discussion on methods such as oversampling, undersampling, and class weighting.
- Transfer learning: How to leverage pre-trained models for classification tasks where limited labeled data is available. This could include a discussion on fine-tuning pre-trained models and using transfer learning to improve classification performance.

## 📉 How models fail

### Labeling

Oh, labeling…

Labeling issues can arise when the data is not labeled correctly or consistently, which can lead to biased or inaccurate models. Examples of labeling issues include:

- **Mislabeling**: Labels that are assigned to data points are incorrect.
- **Ambiguous labeling**: Labels that are assigned to data points are not clear or specific.
- **Inconsistent labeling**: Labels that are assigned to similar data points are not the same

### Fit

A model may fail to fit the data in one of two ways: under-fitting or over-fitting:

- **Under-fitting**: The model fails to capture the the differences between the classes. The model may be too simple, lack the necessary features, or the classes may not easily divide based on existing data.
- **Over-fitting**: The model fits the training data too closely, leading to poor generalization. This can be the case when the model is overly complex or the data may have "too many features".

    > **Note**: _With enough variables you can build a perfect predictor for anything (at least in the training set). That doesn't mean the model will perform well in the wild_

### **Dataset Shift**

Dataset shift occurs when the distribution of the data changes between the training and test sets. Dataset shift can be divided into three types:

1. **Covariate Shift**: A change in the distribution of the independent variables between the training and test sets.
2. **Prior Probability Shift**: A change in the distribution of the target variable between the training and test sets.
3. **Concept Shift**: A change in the relationship between the independent and target variables between the training and test sets.

See: [https://d2l.ai/chapter_linear-classification/environment-and-distribution-shift.html](https://d2l.ai/chapter_linear-classification/environment-and-distribution-shift.html)

### Simpson's Paradox

**Simpson's paradox** occurs when a trend appears in several different groups of data, but disappears or reverses when these groups are combined. It is a common problem in statistics and machine learning that can occur when there are confounding variables that affect the relationship between the independent and dependent variables.

![Simpson's paradox](media/simpsons_paradox.png)

### Troublesome classes

Certain classes or categories in a dataset may be more difficult to classify accurately than others. This can be due to imbalanced class distribution, noisy data, or other factors. Identifying and addressing troublesome classes is an important step in building effective classification models.

Additional topics that could be added to this section include:

- Bias and fairness in classification models
- Lack of interpretability in black-box models
- Adversarial attacks and robustness of classification models
- Transfer learning and domain adaptation in classification models
- Active learning and semi-supervised learning for classification.

![Confused classes](media/confused_classes.png)
## 🏋️ LIVE DEMO!

[demo/02_sensor_classification.md](demo/02_sensor_classification.md)