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

> [!info] UCSF Wynton HPC Cluster  
> 2023-07-21: Rocky 8: Wynton will migrate from CentOS 7 to Rocky 8 at  
> [https://wynton.ucsf.edu/hpc/index.html](https://wynton.ucsf.edu/hpc/index.html)  

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

- **Precision** (Positive Predictive Value) = $\frac{TP}{TP + FP}$﻿
    
    > _How well it performs when it predicts positive_
    
- **Recall** (Sensitivity, True Positive Rate) = $\frac{TP}{TP+FN}$﻿
    
    > _How well it performs among actual positives_
    
- **Accuracy** = **$\frac{(TP+TN)}{(TP+FP+FN+TN)}$**﻿
    
    > _How well it performs among all known classes_
    
- **F1 score** = 2(Recall * Precision)/(Recall + Precision)
    
    > _Balanced score for overall model performance_
    
- **Specificity** (Selectivity, True Negative Rate) = $\frac{TN}{TN + FP}$﻿
    
    > _Similar to_ **Recall**, _how well it performs among actual negatives_
    
- **Miss Rate** (False Negative Rate) = $\frac{FN}{TP + FN}$﻿
    
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

## 🎛️Model tuning (teaser)

Fixing poor-performing models can take many forms

- Additional data or data preparation
    - **FEATURE ENGINEERING!** (worth a whole lecture or even a course)
- Change algorithm or parameters
- Train specialized model for poor-classes

## Process for model training

### Loss & accuracy over epochs

When training machine learning models, it's crucial to understand how loss and accuracy metrics evolve over multiple epochs.

- **epoch:** One full pass through the training data, often broken down into smaller **steps**
- **loss:** Quantifies the difference between the predicted output and the actual target values
    
    > In essence, it represents the error the model is making, and during training, the goal is to minimize this error:
    
- **accuracy:** Model evaluation metric as defined above (# correct)/(# total)
    
    > For certain models, it may be possible to specify different evaluation metrics
    

Monitoring these metrics can provide insights into the model's learning behavior. By observing these trends, we can identify potential challenges such as overfitting or under-fitting and how we might address those issues. We will visualize this with a graph depicting the training and validation loss/accuracy curves during the "Hands-on practice" section.

### Hyperparameters

Also written as "hyper-parameters", these are the arguments passed into models that may be adjusted. **They are set prior to training and adjusting them can significantly impact the model's behavior.**

Hyperparameters play a critical role in shaping the behavior and performance of machine learning models. Examples include learning rates, regularization strengths, and tree depths. Proper tuning of hyperparameters is essential for optimizing model performance.

#### Examples of hyperparameters

- **Learning Rate (LR)**
    - **Applicable Models:** Neural Networks, Gradient Boosted Trees, Support Vector Machines.
    - **Description:** Controls the step size during optimization. Too high can lead to overshooting, too low can result in slow convergence.
- **Regularization Strength**
    - **Applicable Models:** Linear Regression, Logistic Regression.
    - **Description:** Balances fitting the training data well while avoiding overfitting. Higher values increase regularization.
- **Tree Depth (Max Depth)**
    - **Applicable Models:** Decision Trees, Random Forest, Gradient Boosted Trees.
    - **Description:** Limits the maximum depth of individual trees, preventing overfitting.
- **Number of Estimators (Trees)**
    - **Applicable Models:** Random Forest, Gradient Boosted Trees.
    - **Description:** Determines how many trees are built in an ensemble model.
- **Batch Size**
    - **Applicable Models:** Neural Networks.
    - **Description:** Number of training examples utilized in one iteration. Larger batches may speed up training but require more memory.
- **Dropout Rate**
    - **Applicable Models:** Neural Networks.
    - **Description:** Fraction of input units to drop during training. Prevents overfitting by introducing redundancy.
- **Kernel Size**
    - **Applicable Models:** Convolutional Neural Networks (CNN).
    - **Description:** Specifies the size of the convolutional kernel in CNN layers.
- **C (Cost) in SVM**
    - **Applicable Models:** Support Vector Machines.
    - **Description:** Trade-off between smooth decision boundaries and classifying training points correctly.
- **Alpha**
    - **Applicable Models:** Ridge Regression.
    - **Description:** Regularization term in Ridge Regression to control the influence of high-degree polynomial terms.
- **Min Samples Split**
    - **Applicable Models:** Decision Trees, Random Forest.
    - **Description:** The minimum number of samples required to split an internal node.

### Model showdown process

We want to be rigorous when training multiple models and choosing a winner so, like in any research project, we should pre-define the methodology before beginning.

#### Example model selection flow

- **Load data**
    - **Import:** raw data into a performance layer (e.g., database)
    - **Validate:** import matches data dictionary
    - **Store:** in a persistent format (e.g., parquet)
- **Exploratory data analysis**
    - **Autoprofile:** Utilize autoprofiling (via `ydata-profiling` or similar)
        - **Column value counts:** Identify the distribution of values in each column
        - **Numerical distribution:** Measures of center and spread
    - **Autoprofile, split by Outcome/Class:** Generate separate autoprofiles by segment
    - **Identify problem variables:** Identify high missing values, unusual distributions, or potential data quality issues
- **Transform**
    - **Clean up problem variables:** Address missing values, outliers, or any other issues identified during exploratory analysis
    - **Recode categories:** Recode categorical variables as discussed earlier
    - **Manual feature engineering:** Use subject-matter expertise to combine/transform variables
    - **Automated feature engineering:** Use automated feature engineering tools (e.g., `featuretools` )
    - **Statistical feature engineering:** Utilize self-organizing maps, clustering, principle component analysis, etc. to
- **Model selection**
    - **Reserve validation set:** create a validation set that will be excluded from training/testing
    - **Evaluation criteria:** Define evaluation metrics based on the specifics of the problem being addressed
    - **Candidate models:** Which models should be included in the "showdown" based upon the type of problem and data available
    - **Hyperparameter tuning:** Search for good hyperparameters for applicable models, noting that at this stage we are only looking for decent approximations rather than optimal
    - **Crossfold training:** Assess model performance robustness across randomized train/test subsets. Create a random train/test split for each round and that same split across all models that round
    - **Distribution on unknown class:** Examine how well models generalize to the unknown class
    - **Evaluate & select winner:** Evaluate model performance on the validation set and select the best-performing model
    - **Feature importance:** Analyze feature importance scores from different models
- **Document! Document! Document!**
    - **Process:** What did you do and why
    - **Trade-offs:** Which choices were made, what would be better/worse if you had made others
    - **Technical information:** Anything necessary to reproduce your training
    - **Results**: (_duh_)

### Tips

- **Start with "babies":** Start with smaller, simpler models to identify issues early and streamline the debugging process.
- **Use validation sets**: A separate validation set allows you to evaluate model performance during training and prevent overfitting
- **Early stopping**: It is possible to halt training when improvements on the validation set plateau, preventing overfitting by stopping when loss increments fall below a threshold.
- **Logging and monitoring**: Log key metrics and monitor them during training, either through log files, print statements, or tools like TensorBoard for real-time visualization
- **Document :all_the_things:**: Clear documentation, including hyper-parameters and model architecture, are the only way to make your models reproducible and to collaboratively troubleshoot.
- **Cross-Validation**: Cross-validation (train/test on random subsets) is important for obtaining a robust estimate of model performance, particularly with limited data.

## 🏋️Hands-on practice

### Which animal is this?

Adapted from Google Keras code example [Image classification from scratch](https://keras.io/examples/vision/image_classification_from_scratch/). We'll try to classify dogs vs cats vs pandas:

![Dog example](media/dogs_00013.jpg)

![Cat example](media/cats_00016.jpg)

![Panda example](media/panda_00001.jpg)

1. Make sure we're in a virtual environment
2. Install and load libraries
3. Download data and split into train, test, and validation
4. Define model
5. Train model
6. Check train/test performance
7. Validate against new data

### Classify 0 vs 1 from `emnist`

#### Setup

**Install/import libraries**

As usual, remember to use a virtual environment!

**Download data**

The `emnist` library will download a copy of the dataset

#### Define helper functions, columns, subsets

It's a good idea to preprocess the data to make it easier to work with. You can create subsets of the data for training, validation, and testing. Also, since the labels in the original dataset are encoded as integers, it may be helpful to create a dictionary that maps the integer labels to their corresponding characters.

#### Pre-built models classifying 0/1

- Logistic regression
- RandomForest
- XGBoost
- Neural network

#### Evaluate/compare model performance

- Confusion matrix: A table that shows the number of true positives, true negatives, false positives, and false negatives for a binary classification problem.
- Accuracy: The proportion of correct predictions over the total number of predictions.
- Precision: The proportion of true positives over the total number of positive predictions.
- Recall: The proportion of true positives over the total number of actual positives.
- F1 score: The harmonic mean of precision and recall, which balances both metrics and gives equal weight to both.

## 🦾Exercise

### 1. Classify all symbols

#### Choose a model

Your choice of model! Choose wisely…

#### Train away!

Is do you need to tune any parameters? Is the model expecting data in a different format?

#### Evaluate the model

Evaluate the models on the test set, analyze the confusion matrix to see where the model performs well and where it struggles.

#### Investigate subsets

On which classes does the model perform well? Poorly? Evaluate again, excluding easily confused symbols (such as 'O' and '0').

#### Improve performance

Brainstorm for improving the performance. This could include trying different architectures, adding more layers, changing the loss function, or using data augmentation techniques.

  

## 🚶‍♀️Self-guided topics

### Awesome list of applications

- [https://github.com/ritchieng/the-incredible-pytorch](https://github.com/ritchieng/the-incredible-pytorch)

### More MNIST

Classifying hand-written digits is the "Hello, World!" of image ML.

- K-means - [https://github.com/sharmaroshan/MNIST-Using-K-means](https://github.com/sharmaroshan/MNIST-Using-K-means)
- MNIST, the Hello World of Deep Learning ([medium](https://medium.com/fenwicks/tutorial-1-mnist-the-hello-world-of-deep-learning-abd252c47709))

### Fashion MNIST

- `torchvision` [provides this dataset](https://pytorch.org/vision/stable/datasets.html) and is a great tool for image classification
- [Fashion MNIST](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb) example using Colab
- TensorFlow - [https://jobcollinsdulo.medium.com/part-one-image-classification-with-tensorflow-python-f92f94121ec1](https://jobcollinsdulo.medium.com/part-one-image-classification-with-tensorflow-python-f92f94121ec1)
- PyTorch - [https://medium.com/ml2vec/intro-to-pytorch-with-image-classification-on-a-fashion-clothes-dataset-e589682df0c5](https://medium.com/ml2vec/intro-to-pytorch-with-image-classification-on-a-fashion-clothes-dataset-e589682df0c5)

### Tabular data

- [https://github.com/ThisIsJohnnyLau/dirty_data_project](https://github.com/ThisIsJohnnyLau/dirty_data_project) (6 datasets for cleaning)
- Using Unsupervised Learning to optimise Children's T-shirt Sizing ([towardsdatascience](https://towardsdatascience.com/using-unsupervised-learning-to-optimise-childrens-t-shirt-sizing-d919d3cbc1f6))

### Panoramic dental x-rays

Example flow:

- [https://github.com/clemkoa/tooth-detection](https://github.com/clemkoa/tooth-detection)
- [https://github.com/Nirzu97/PROJECT-Dental-Disease-Detection](https://github.com/Nirzu97/PROJECT-Dental-Disease-Detection)
- X-ray imaging available at the [Tufts Dental Database](http://tdd.ece.tufts.edu)

### Data cleaning for images

- Introduction to Image Pre-processing | What is Image Pre-processing? ([Great Learning](https://www.mygreatlearning.com/blog/introduction-to-image-pre-processing/))
- [https://github.com/Nirzu97/PROJECT-Dental-Disease-Detection](https://github.com/Nirzu97/PROJECT-Dental-Disease-Detection) (see pptx for a good slide on this)

### Publications on X-ray classification

- Supervised and unsupervised language modelling in Chest X-Ray ([PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0229963))
- Unsupervised Clustering of COVID-19 Chest X-Ray Images with a Self-Organizing Feature Map ([IEEE Xplore](https://ieeexplore.ieee.org/document/9184493))
- A benchmark for comparison of dental radiography analysis algorithms ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1361841516000190))