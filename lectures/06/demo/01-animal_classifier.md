---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: '1.14.1'
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Animal Classifier Demo: Understanding Dataset Limitations

## Learning Objectives 🎯

By the end of this demo, you will be able to:

1. Understand the importance of diverse training data
2. Build and train a simple CNN for image classification
3. Evaluate model performance on unseen data categories

## Setup and Imports

```python
# Install required libraries
%pip install tensorflow numpy matplotlib --quiet
```

```python
# Data manipulation and visualization
import numpy as np
import matplotlib.pyplot as plt

# TensorFlow/Keras for building CNN
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
```

## 1. Generate and Visualize Animal Images

Let's create a synthetic dataset that mimics images of cats, dogs, and pandas:

```python
# Generate synthetic image data
# For simplicity, we'll use random noise images
import numpy as np

# Generate random images
x_train_cats = np.random.rand(100, 224, 224, 3)
x_train_dogs = np.random.rand(100, 224, 224, 3)
x_test_pandas = np.random.rand(20, 224, 224, 3)

# Create labels
y_train_cats = np.zeros(100)
y_train_dogs = np.ones(100)

# Combine data
x_train = np.concatenate((x_train_cats, x_train_dogs))
y_train = np.concatenate((y_train_cats, y_train_dogs))

print("Training set shape:", x_train.shape)
print("Test set shape:", x_test_pandas.shape)
```

## 2. Prepare Data for Modeling

```python
# Normalize pixel values to [0,1] range
x_train = x_train / 255.0
x_test_pandas = x_test_pandas / 255.0

# Convert labels to categorical
y_train_cat = keras.utils.to_categorical(y_train, num_classes=2)
```

## 3. Build and Train a Simple CNN Model

```python
# Define a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train_cat, epochs=10)
```

## 4. Test on Unseen Data (Pandas Images)

```python
# Make predictions on pandas images
predictions = model.predict(x_test_pandas)

# Print predictions
print(predictions)

# Expected outcome: The model will likely classify pandas as either cats or dogs
```

### Expected Outcome

The model will likely classify pandas as either cats or dogs, highlighting the importance of diverse training data.

```
# Define a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')  # Output layer for cat/dog classification
])

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)

# Test on pandas images
predictions = model.predict(x_test_pandas)
print(predictions)
```

### Expected Outcome

The model will likely classify pandas as either cats or dogs, highlighting the importance of diverse training data.
