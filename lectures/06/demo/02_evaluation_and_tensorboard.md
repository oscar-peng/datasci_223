# Demo 2: Model Evaluation and Monitoring 📊📈

## Goal
Learn how to evaluate model performance using various metrics and monitor training progress with TensorBoard.

## Setup
```python
# Install required packages
!pip install tensorflow tensorboard numpy matplotlib
```

## Data Preparation and Normalization
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape for the neural network
X_train = X_train.reshape(-1, 28 * 28)
X_test = X_test.reshape(-1, 28 * 28)

# Convert labels to one-hot encoding
y_train_onehot = tf.keras.utils.to_categorical(y_train)
y_test_onehot = tf.keras.utils.to_categorical(y_test)

def display_samples(X, y, n_samples=5):
    """Display sample images from the dataset."""
    plt.figure(figsize=(12, 4))
    for i in range(n_samples):
        plt.subplot(1, n_samples, i + 1)
        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        plt.title(f'Label: {y[i]}')
        plt.axis('off')
    plt.show()

# Display some training samples
display_samples(X_train, y_train)
```

## Create a Simple Model
```python
def create_model():
    """Create and compile a simple neural network model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC()
        ]
    )
    
    return model

model = create_model()
model.summary()
```

## Set Up TensorBoard
```python
import datetime

# Create a directory for TensorBoard logs
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Set up TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='epoch',
    profile_batch=2
)

# Set up early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
```

## Train and Monitor
```python
# Train the model
history = model.fit(
    X_train, y_train_onehot,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[tensorboard_callback, early_stopping]
)

# Evaluate on test set
test_loss, test_acc, test_precision, test_recall, test_auc = model.evaluate(
    X_test, y_test_onehot
)

print(f'\nTest accuracy: {test_acc:.2%}')
print(f'Test precision: {test_precision:.2%}')
print(f'Test recall: {test_recall:.2%}')
print(f'Test AUC: {test_auc:.2%}')
```

## Visualize Results
```python
def plot_metrics(history):
    """Plot training and validation metrics."""
    metrics = ['accuracy', 'precision', 'recall', 'auc']
    
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        plt.plot(history.history[metric], label=f'Training {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        plt.title(f'Training and Validation {metric.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
    plt.tight_layout()
    plt.show()

plot_metrics(history)
```

## Launch TensorBoard
```python
# Start TensorBoard
%load_ext tensorboard
%tensorboard --logdir logs/fit
```

## Exercise
1. Try different normalization techniques:
   - Min-max scaling
   - Standardization
   - Robust scaling
2. Experiment with different metrics:
   - F1 score
   - ROC curve
   - Precision-Recall curve
3. Adjust model architecture:
   - Add more layers
   - Change activation functions
   - Add dropout
4. Try different optimizers and learning rates

## Discussion Points
- How do different normalization techniques affect model performance?
- Which metrics are most important for your specific use case?
- How can TensorBoard help in debugging training issues?
- What patterns do you notice in the learning curves?

## Next Steps
- Implement custom metrics
- Add more sophisticated monitoring
- Try regularization techniques
- Experiment with learning rate scheduling

<!---
Speaking Notes:
- Explain the importance of proper data normalization
- Discuss the trade-offs between different metrics
- Highlight how TensorBoard can help identify issues
- Emphasize the importance of monitoring validation metrics
- Explain how to interpret different types of plots
--> 