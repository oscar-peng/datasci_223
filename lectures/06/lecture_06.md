Neural Networks: If I Only Had a Brain

![junior dev vs. NN](media/70wnk5kfr5hc1.jpeg)

# Neural Networks Overview

## Biological Inspiration

![biological neuron](media/Untitled.png)

A **neuron** has:

- Branching input (dendrites)
- Branching output (the axon)

The information circulates from the dendrites to the axon via the cell body. Axon connects to dendrites via synapses:

- Synapses vary in strength
- Synapses may be excitatory or inhibitory

### Pigeons as Art Experts (Watanabe et al. 1995)

Experiment:

- Pigeon in Skinner box
- Present paintings of two different artists (e.g. Chagall / Van Gogh)
- Reward for pecking when presented a particular artist (e.g. Van Gogh)

![pigeon in skinner box](media/Untitled%201.png)

![van gogh painting](media/Untitled%202.png)

![chagall painting](media/Untitled%203.png)

Pigeons were able to discriminate between Van Gogh and Chagall with 95% accuracy (when presented with pictures they had been trained on). Discrimination still 85% successful for previously unseen paintings of the artists.

Pigeons do not simply memorize the pictures!

- They can extract and recognize patterns (the 'style')
- They generalize from the already seen to make predictions

This is what neural networks (biological and artificial) are good at (unlike conventional computers).

## Artificial Neural Networks

Neural networks draw inspiration from the biological neural networks that constitute animal brains. Just as biological neurons transmit signals to each other via synapses, artificial neural networks (ANNs) consist of interconnected nodes or "neurons" that process and pass on information.

**Artificial neurons:** Non-linear, parameterized function with restricted output range

![simplified neuron](media/Screenshot_2024-02-26_at_1.02.39_PM.png)

![NN equation](media/ann.png)

![basic neural net](media/nn_overview.png)

## Famous Application: Tank or Not-a-Tank

In the 1980s (some say 60's?), the Pentagon wanted to harness computer technology to make their tanks harder to attack. The plan was to fit each tank with a digital camera hooked up to a computer that would scan the environment for threats.

The research team took 100 photographs of tanks hiding behind trees, and 100 photographs of trees with no tanks. They put 50 from each group in a vault for safe-keeping and trained a neural network on the remaining 100 photos.

![image of tank](media/Untitled%204.png)

![image of not-a-tank](media/Untitled%205.png)

### Success!

The neural network was fed each photo and asked if there was a tank. At first, answers were random. But each time it was wrong, it would adjust its weights until correct.

To their immense relief, the neural net correctly identified each photo from the vault as either having a tank or not having one.

### Testing with New Data

The Pentagon took another set of photos and ran them through the neural network.

**The results were completely random.**

After investigation, they discovered the problem: all tank photos were taken on sunny days, while tree-only photos were taken on cloudy days.

**The military was now the proud owner of a multi-million dollar mainframe computer that could tell you if it was sunny or not!**

> **Note:** The tank detector example is apocryphal - there's no evidence it actually happened. However, it serves as a useful parable about data bias and overfitting.

### Lesson Learned

Data bias can lead to unexpected model behavior. Ensuring diverse and representative training data is crucial.

## Applications in Machine Learning

Neural networks have revolutionized machine learning:

- **Image Recognition:** CNNs power facial recognition and medical imaging diagnostics
- **Natural Language Processing:** RNNs and Transformers enable translation, chatbots, and voice assistants
- **Autonomous Driving:** Neural networks interpret sensor data and make navigation decisions

## Universal Approximation Theorem

One of the most profound aspects of neural networks is their ability to approximate virtually any complex function. This theorem suggests that a feedforward network with a single hidden layer can approximate continuous functions, given appropriate activation functions.

The **layered composition** of neural networks allows them to learn hierarchies of features—initial layers recognize edges and textures, while deeper layers identify shapes and objects.

![universal approximation](media/universal_approx.gif)

# LIVE DEMO!

A hands-on CNN classification demo: Which animal is this? (cat, dog, or panda)

See: [demo/01_which_animal.md](demo/01_which_animal.md)

# Activation Functions

Activation functions introduce non-linearity into neural networks. Without them, a network would behave like a linear model, unable to capture complex patterns.

## Comparison of Activation Functions

> **Vanishing gradients:** When gradients become extremely small during backpropagation, early layers stop learning effectively. This is why activation function choice matters for deep networks.

| Function | Pros | Cons | Use Cases |
|---------|------|------|-----------|
| ReLU | Computationally efficient, mitigates vanishing gradients | Dying ReLU problem | Deep networks, hidden layers |
| Sigmoid | Outputs probability between 0 and 1 | Vanishing gradients, not zero-centered | Binary classification output layer |
| Tanh | Zero-centered, stronger gradients than sigmoid | Vanishing gradients | Hidden layers |
| Leaky ReLU | Mitigates dying ReLU problem | Non-zero gradient for negatives | When dying ReLU is a concern |

## Introducing: ReLU

The **Rectified Linear Unit (ReLU)** is the most widely used activation function in deep learning.

### Reference Card: ReLU

| Component | Details |
|:---|:---|
| **Function** | $f(x) = \max(0, x)$ |
| **Purpose** | Introduces non-linearity by replacing negative values with zero |
| **Strengths** | Computationally efficient, mitigates vanishing gradient |
| **Gradient** | 1 for positive inputs, 0 for negative |

### Code Snippet: ReLU

```python
import numpy as np

x = np.array([-1, 0, 1])
relu_output = np.maximum(0, x)
print(relu_output)  # Output: [0 0 1]
```

![ReLU graph](media/relu.png)

### Advantages of ReLU

- **Simplicity:** Computationally efficient max operation
- **Mitigates Vanishing Gradient:** Gradient of 1 for positive inputs
- **Sparse Activation:** Only positive values activate, leading to efficient models
- **Improved Gradient Flow:** Constant gradient enables deeper network training

# Preparing Inputs

Proper input preparation is crucial for efficient training.

## Why Normalize and Standardize?

1. **Improved Convergence:** Neural networks converge faster with normalized data
2. **Avoiding Saturation:** Prevents activation functions like sigmoid from saturating
3. **Equal Contribution:** All features contribute equally to learning
4. **Better Performance:** More stable and efficient training

## Techniques

### Reference Card: Data Normalization

| Technique | Description | When to Use |
|:---|:---|:---|
| **Normalization** | Scale to range [-1, 1] or [0, 1] | Data doesn't follow Gaussian distribution |
| **Standardization** | Transform to mean=0, std=1 | Data follows Gaussian distribution |

### Code Snippet: Standardization

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

data = np.array([[1, 200], [2, 300], [3, 400]])
scaler = StandardScaler()
standardized_data = scaler.fit_transform(data)
print(standardized_data)
```

## Input Shape Importance

- **Consistent Dimensions:** Neural networks require fixed input size
- **Batch Size:** Affects training speed and stability

## Feature Engineering

- **Feature Selection:** Use PCA or correlation analysis to identify relevant features
- **Feature Encoding:** One-hot encoding or embedding layers for categorical variables
- **Feature Construction:** Domain knowledge to create meaningful features

# Training Neural Networks

Training involves adjusting weights and biases to minimize the difference between predicted and actual outputs.

## Backpropagation

**Backpropagation** is the cornerstone of neural network training. It distributes error back through the network layers, providing insight into each weight's responsibility for the error.

![backpropagation](media/backpropagation.png)

## Gradient Descent

**Gradient descent** is the optimization algorithm used to minimize the cost function by adjusting weights in the direction that reduces cost.

## Cost Functions

### Reference Card: Cost Functions

| Function | Best For | Strengths | Weaknesses |
|:---|:---|:---|:---|
| **MSE** | Regression | Intuitive, penalizes large errors | Sensitive to outliers |
| **Cross-Entropy** | Classification | Works with probability outputs | Numerical instability risk |
| **Binary Cross-Entropy** | Binary classification | Aligns with probability outputs | Only for binary tasks |
| **Huber Loss** | Robust regression | Less sensitive to outliers | δ parameter needs tuning |

### Code Snippet: Cross-Entropy Loss

```python
import torch
import torch.nn.functional as F

predictions = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
labels = torch.tensor([0, 1])
loss = F.cross_entropy(predictions, labels)
print(loss)
```

## Regularization and Overfitting

Overfitting occurs when a model learns training data too well, including its noise.

**Regularization methods:**

- **L1/L2 regularization:** Add penalty for large weights
- **Dropout:** Randomly zero elements during training
- **Early stopping:** Stop training when validation performance plateaus

## Evaluation Metrics

| Metric | Strengths | Weaknesses |
|:---|:---|:---|
| Accuracy | Intuitive | Misleading on imbalanced data |
| Precision/Recall | Good for imbalanced data | Trade-off between them |
| F1 Score | Balances precision and recall | May hide nuances |
| AUC-ROC | Robust to imbalance | Less informative at extremes |

# LIVE DEMO!!

Training a neural network on handwritten digits (EMNIST).

See: [demo/02_handwritten_digits.md](demo/02_handwritten_digits.md)

# Model Architecture

The architecture of a neural network defines its ability to learn and solve problems.

## Network Depth

- **Shallow Networks:** Few layers, suited for simpler problems
- **Deep Networks:** Many layers, learn features at multiple levels of abstraction

### Depth Challenges

Deep networks face vanishing/exploding gradients. Solutions include:

- Residual connections (ResNets)
- Batch normalization
- Advanced optimizers

## Connectedness: Dense vs. Sparse

- **Fully Connected:** Every neuron connects to every neuron in adjacent layers
- **Sparse Connectivity:** Neurons connect only to a subset (e.g., CNNs)
- **Skip Connections:** Allow gradients to flow through the network directly
- **Recurrent Connections:** Form loops for sequential data (RNNs)

### Reference Card: Residual Connections

| Component | Details |
|:---|:---|
| **Purpose** | Allow gradients to flow directly by skipping layers |
| **Benefit** | Mitigates vanishing gradient, enables very deep networks |
| **Used In** | ResNet architectures |

### Code Snippet: Residual Block (PyTorch)

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)

    def forward(self, x):
        residual = x
        out = torch.relu(self.fc1(x))
        out = self.fc2(out)
        out += residual  # Residual connection
        return torch.relu(out)
```

## CNN vs. RNN Comparison

| Feature | CNN | RNN |
|---------|-----|-----|
| Use Case | Images, spatial data | Time series, sequential data |
| Key Layer | Convolutional | Recurrent |
| Strength | Local pattern detection | Sequence memory |
| Data Structure | Grid-like (pixels) | Sequential (words, time steps) |
| Parallelization | Highly parallelizable | Sequential processing |

## Convolutional Neural Networks (CNNs)

CNNs are specialized for processing grid-like data (images). They learn hierarchical features: early layers detect edges and textures, deeper layers recognize shapes and objects.

### Reference Card: `Conv2D`

| Component | Details |
|:---|:---|
| **Function** | `tensorflow.keras.layers.Conv2D()` |
| **Purpose** | Apply learnable filters to extract spatial features from images |
| **Key Parameters** | • `filters`: Number of output filters<br>• `kernel_size`: Size of convolution window (e.g., 3 or (3,3))<br>• `strides`: Step size for sliding window<br>• `padding`: 'valid' (no padding) or 'same' (preserve dimensions)<br>• `activation`: Activation function |
| **Output Shape** | (batch, height, width, filters) |

### Reference Card: `MaxPooling2D`

| Component | Details |
|:---|:---|
| **Function** | `tensorflow.keras.layers.MaxPooling2D()` |
| **Purpose** | Downsample by taking maximum value in each window |
| **Key Parameters** | • `pool_size`: Window size (e.g., (2,2))<br>• `strides`: Step size (defaults to pool_size)<br>• `padding`: 'valid' or 'same' |
| **Effect** | Reduces spatial dimensions, adds translation invariance |

### Reference Card: `BatchNormalization`

| Component | Details |
|:---|:---|
| **Function** | `tensorflow.keras.layers.BatchNormalization()` |
| **Purpose** | Normalize layer inputs to stabilize and speed up training |
| **When to Use** | After Conv2D or Dense layers, before activation |
| **Benefits** | Faster convergence, acts as regularizer, allows higher learning rates |

### Reference Card: `Dropout`

| Component | Details |
|:---|:---|
| **Function** | `tensorflow.keras.layers.Dropout(rate)` |
| **Purpose** | Randomly set inputs to zero during training to prevent overfitting |
| **Key Parameters** | • `rate`: Fraction of inputs to drop (e.g., 0.5 = 50%) |
| **Note** | Only active during training, not inference |

### Code Snippet: Simple CNN

```python
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

## Recurrent Neural Networks (RNNs)

RNNs handle sequential data where order matters—like time series, text, or ECG signals.

![RNN](media/rnn.png)

### Reference Card: `SimpleRNN`

| Component | Details |
|:---|:---|
| **Function** | `tensorflow.keras.layers.SimpleRNN()` |
| **Purpose** | Basic RNN layer for sequential data |
| **Key Parameters** | • `units`: Number of output units<br>• `activation`: Activation function (default 'tanh')<br>• `return_sequences`: Return full sequence or just last output<br>• `input_shape`: Tuple of (timesteps, features) |
| **Use Cases** | Short sequences, simple patterns, educational demos |

### Code Snippet: SimpleRNN

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Simple RNN for sequence classification
model = Sequential([
    SimpleRNN(32, input_shape=(time_steps, features)),
    Dense(num_classes, activation='softmax')
])
```

### Long Short-Term Memory (LSTM)

For longer sequences, LSTM addresses the **vanishing gradient problem** (where gradients become too small to update early layers effectively) and captures long-term dependencies.

**Three gates control information flow:**

1. **Input Gate:** Controls what new information enters the cell state
2. **Forget Gate:** Controls what information to discard
3. **Output Gate:** Controls what information to output

### Reference Card: `LSTM`

| Component | Details |
|:---|:---|
| **Function** | `tensorflow.keras.layers.LSTM()` |
| **Purpose** | Process sequential data with long-term memory |
| **Key Parameters** | • `units`: Dimensionality of output space<br>• `return_sequences`: Return full sequence (True) or just last output (False)<br>• `dropout`: Fraction of units to drop for inputs<br>• `recurrent_dropout`: Fraction to drop for recurrent state |
| **Use Cases** | Time series forecasting, text generation, clinical sequence data |

![LSTM vs GRU](media/LSTMvsGRU.png)

![LSTM vs GRU Detailed](media/lstmvsgru2.png)

### Code Snippet: LSTM for Time Series

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(64, input_shape=(time_steps, features)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # Predict next value
])
```

## Specialized Architectures

- **GANs:** Generator and discriminator trained together for data generation
- **Graph Neural Networks:** For graph-structured data
- **Autoencoders:** For dimensionality reduction and feature learning
- **Diffusion Models:** Gradually add/remove noise for generation
- **Large Language Models (LLMs):** Transformer-based models for text understanding and generation

# Training Callbacks

Callbacks let you hook into the training process to save checkpoints, stop early, or log metrics.

### Reference Card: `ModelCheckpoint`

| Component | Details |
|:---|:---|
| **Function** | `tf.keras.callbacks.ModelCheckpoint()` |
| **Purpose** | Save model weights or full model during training |
| **Key Parameters** | • `filepath`: Path to save model (can include `{epoch}`, `{val_loss}`)<br>• `save_best_only`: Only save when monitored metric improves<br>• `monitor`: Metric to monitor (e.g., 'val_loss', 'val_accuracy')<br>• `save_weights_only`: Save weights only or full model |
| **Use Case** | Keep best model for deployment, resume training after interruption |

### Reference Card: `EarlyStopping`

| Component | Details |
|:---|:---|
| **Function** | `tf.keras.callbacks.EarlyStopping()` |
| **Purpose** | Stop training when monitored metric stops improving |
| **Key Parameters** | • `monitor`: Metric to monitor (e.g., 'val_loss')<br>• `patience`: Epochs to wait before stopping<br>• `restore_best_weights`: Restore weights from best epoch<br>• `min_delta`: Minimum change to qualify as improvement |
| **Use Case** | Prevent overfitting, save training time |

### Code Snippet: Training Callbacks

```python
import tensorflow as tf

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        save_best_only=True,
        monitor='val_accuracy'
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
]

history = model.fit(X_train, y_train, 
                    validation_data=(X_val, y_val),
                    epochs=100,  # Will stop early if no improvement
                    callbacks=callbacks)
```

# Saving Models

### Reference Card: Model Saving

**Keras:**

```python
# Save entire model
model.save('my_model.keras')

# Load model
loaded_model = tf.keras.models.load_model('my_model.keras')
```

**PyTorch:**

```python
# Save model state
torch.save(model.state_dict(), 'model.pth')

# Save checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')
```

# TensorBoard for Monitoring

TensorBoard provides real-time visualization of training metrics.

### Reference Card: TensorBoard Setup

**Keras:**

```python
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir="./logs",
    histogram_freq=1,
    write_graph=True
)
model.fit(x_train, y_train, callbacks=[tensorboard_callback])
```

**PyTorch:**

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')
for epoch in range(num_epochs):
    writer.add_scalar('Loss/train', loss.item(), epoch)
    writer.add_scalar('Accuracy/train', accuracy, epoch)
```

# Implementing Custom Models

## Common Layer Types

| Layer Type | Purpose | When to Use |
|------------|---------|-------------|
| **Dense/Linear** | Fully connected | Final layers, tabular data |
| **Conv2D** | Spatial feature extraction | Images, medical imaging |
| **LSTM** | Sequential data with memory | Time series, clinical notes |
| **Embedding** | Map indices to vectors | Text, categorical data |
| **BatchNorm** | Normalize layer inputs | Deep networks, unstable training |
| **Dropout** | Prevent overfitting | Large networks |

## Keras Example

```python
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape

input_size = 784  # 28x28 images
hidden_size = 128
num_classes = 47

model = Sequential([
    Reshape((28, 28, 1), input_shape=(input_size,)),
    Flatten(),
    Dense(hidden_size, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

## PyTorch Example

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=47):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

## Training Loop (PyTorch)

```python
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
```

# LIVE DEMO!!!

Building a neural network from scratch on MNIST.

See: [demo/03_mnist_from_scratch.md](demo/03_mnist_from_scratch.md)

# Neural Networks in Practice

## Areas of Active Research

- **Few-Shot Learning:** Learning from very limited data
- **Generative Models:** GANs, diffusion models for content creation
- **Reinforcement Learning:** Autonomous systems and decision-making
- **Explainability and Ethics:** Understanding and ensuring ethical usage

## Current Limitations

- **Interpretability:** Black-box nature makes decisions hard to understand
- **Data Dependency:** High-performing networks require vast labeled data
- **Computational Resources:** Training requires specialized hardware (GPUs/TPUs)
- **Hallucination:** Generative models produce plausible but incorrect outputs

## Design Patterns

- **Ensemble Methods:** Combine multiple models for better performance
- **Attention Mechanisms:** Focus on relevant parts of input data
- **Normalization Techniques:** Batch, layer, and instance normalization
- **Transfer Learning:** Leverage pre-trained models for new tasks
- **Hybrid Models:** Combine CNNs and RNNs for complex tasks

# References

## Books

- _Deep Learning_, Goodfellow, Bengio & Courville - [free online](https://www.deeplearningbook.org/)
- _Deep Learning with Python_, Chollet - [Manning](https://www.manning.com/books/deep-learning-with-python-second-edition)
- _Dive into Deep Learning_ - [d2l.ai](https://d2l.ai)

## Tutorials

- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Keras Documentation](https://keras.io/)

## Health Data Science & Deep Learning

- Miotto et al. (2018). Deep learning for healthcare: review, opportunities and challenges. _Briefings in Bioinformatics_
- Esteva et al. (2017). Dermatologist-level classification of skin cancer with deep neural networks. _Nature_
- Rajpurkar et al. (2017). CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning
