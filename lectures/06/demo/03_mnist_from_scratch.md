# Demo 3: Building Neural Networks from Scratch 🏗️

## Goal
Build and train a CNN from scratch using both PyTorch and Keras, and implement an RNN for sequence data.

## Setup
```python
%pip install tensorflow torch numpy matplotlib tensorflow-datasets torchvision
```

## Part 1: CNN in PyTorch

### Data Loading and Preprocessing
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Set device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define preprocessing pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize using EMNIST statistics
])

# Load EMNIST byclass dataset (contains all digits and letters)
train_dataset = torchvision.datasets.EMNIST(
    root='./data', 
    split='byclass',  # Use byclass to get all characters
    train=True,
    download=True, 
    transform=transform
)

# Filter to keep only digits (classes 0-9)
digit_indices = [i for i, (_, label) in enumerate(train_dataset) if label < 10]
train_dataset = torch.utils.data.Subset(train_dataset, digit_indices)

# Same for test set
test_dataset = torchvision.datasets.EMNIST(
    root='./data', 
    split='byclass',
    train=False,
    download=True, 
    transform=transform
)
test_digit_indices = [i for i, (_, label) in enumerate(test_dataset) if label < 10]
test_dataset = torch.utils.data.Subset(test_dataset, test_digit_indices)

# Create data loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=64, 
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=1000, 
    shuffle=False
)

# Visualize some examples
def plot_examples(loader, num_examples=5):
    dataiter = iter(loader)
    images, labels = next(dataiter)
    
    fig = plt.figure(figsize=(12, 3))
    for i in range(num_examples):
        ax = fig.add_subplot(1, num_examples, i + 1)
        ax.imshow(images[i].squeeze().numpy(), cmap='gray')
        ax.set_title(f'Label: {labels[i].item()}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plot_examples(train_loader)
```

### CNN Architecture Definition
```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First convolutional layer: 1 input channel (grayscale), 32 output channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        # Second convolutional layer: 32 input channels, 64 output channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        # Dropout layer to prevent overfitting
        self.conv2_drop = nn.Dropout2d()
        # Fully connected layers
        self.fc1 = nn.Linear(1600, 100)  # 1600 = 64 * 5 * 5 (after two max pools)
        self.fc2 = nn.Linear(100, 10)    # 10 output classes (digits 0-9)

    def forward(self, x):
        # First conv block
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        
        # Second conv block
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.conv2_drop(x)
        
        # Flatten and fully connected layers
        x = x.view(-1, 1600)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

# Initialize model, loss function, and optimizer
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
```

### Training and Evaluation Functions
```python
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')

# Train and evaluate
for epoch in range(1, 3):
    train(epoch)
    test()
```

## Part 2: CNN in Keras

### Data Loading and Preprocessing
```python
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models

# Load EMNIST byclass dataset
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/byclass',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

# Filter function to keep only digits (classes 0-9)
def filter_digits(image, label):
    return tf.less(label, 10)

# Preprocess function
def preprocess(image, label):
    # Convert to float and normalize
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Configure dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 128  # Increased batch size for better GPU utilization

# Prepare training dataset with optimized pipeline
ds_train = (ds_train
    .filter(filter_digits)
    .cache()  # Cache the filtered dataset
    .shuffle(10000)  # Shuffle before batching
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE))

# Prepare test dataset
ds_test = (ds_test
    .filter(filter_digits)
    .cache()
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE))

# Create a copy of the dataset for visualization
# This prevents the "End of sequence" errors when we need to visualize multiple times
ds_train_viz = ds_train
ds_test_viz = ds_test

# Visualize some examples
def plot_tfds_examples(dataset, num_examples=5):
    # Get a single batch of data
    batch = next(iter(dataset))
    images, labels = batch
    
    # Create the plot
    plt.figure(figsize=(12, 3))
    for i in range(min(num_examples, len(images))):
        plt.subplot(1, num_examples, i + 1)
        plt.imshow(images[i].numpy(), cmap='gray')
        plt.title(f'Label: {labels[i].numpy()}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Use the visualization copy
plot_tfds_examples(ds_train_viz)
```

### CNN Architecture Definition
```python
# Define CNN model
model = models.Sequential([
    # First conv block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # Second conv block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Third conv block
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Dense layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes for digits
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
model.summary()
```

### Training and Evaluation
```python
# Train model
history = model.fit(
    ds_train,
    epochs=5,
    validation_data=ds_test
)

# Evaluate
test_loss, test_acc = model.evaluate(ds_test)
print(f'\nTest accuracy: {test_acc:.2%}')

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()
```

## Part 3: RNN for Health Data Classification

### Data Loading and Preprocessing
```python
# Load UCI Heart Disease dataset
(ds_train, ds_test), ds_info = tfds.load(
    'heart_disease',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

# Print dataset info
print("Dataset info:", ds_info)

# Preprocess function
def preprocess(features, label):
    # Convert features to float32 and normalize
    features = tf.cast(features, tf.float32)
    # Normalize each feature to [0, 1] range
    features = (features - tf.reduce_min(features)) / (tf.reduce_max(features) - tf.reduce_min(features))
    # Reshape for RNN input: (sequence_length, features)
    features = tf.reshape(features, [1, -1])  # Each sample is a sequence of length 1
    return features, label

# Simple dataset preparation
ds_train = ds_train.map(preprocess).batch(32)
ds_test = ds_test.map(preprocess).batch(32)

# Visualize some examples
def plot_heart_disease_examples(dataset, num_examples=5):
    # Get one batch
    features, labels = next(iter(dataset))
    
    # Create the plot
    plt.figure(figsize=(15, 10))
    for i in range(min(num_examples, len(features))):
        plt.subplot(num_examples, 1, i+1)
        plt.bar(range(len(features[i][0])), features[i][0])
        plt.title(f'Heart Disease Features - Class: {labels[i]}')
        plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_heart_disease_examples(ds_train)
```

### RNN Architecture Definition
```python
# Define RNN model
model = models.Sequential([
    layers.SimpleRNN(32, input_shape=(1, 13)),
    layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
model.summary()
```

### Training and Evaluation
```python
# Train model
history = model.fit(
    ds_train,
    epochs=5,
    validation_data=ds_test
)

# Evaluate
test_loss, test_acc = model.evaluate(ds_test)
print(f'\nTest accuracy: {test_acc:.2%}')

# Visualize predictions
def plot_heart_disease_prediction(model, ds_test):
    # Get one batch
    features, labels = next(iter(ds_test))
    
    # Make predictions
    predictions = model.predict(features)
    
    # Create the plot
    plt.figure(figsize=(15, 10))
    for i in range(min(5, len(features))):
        plt.subplot(5, 1, i+1)
        plt.bar(range(len(features[i][0])), features[i][0])
        plt.title(f'True: {labels[i]}, Pred: {predictions[i][0]:.2f}')
    plt.tight_layout()
    plt.show()

plot_heart_disease_prediction(model, ds_test)
```