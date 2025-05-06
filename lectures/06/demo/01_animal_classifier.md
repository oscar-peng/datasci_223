# Demo 1: Animal Classification with Pre-trained Models 🐾

## Goal
Learn how to use pre-trained models for image classification, focusing on the complete workflow from training to evaluation and prediction.

## Setup
```python
# Install required packages
!pip install tensorflow pillow numpy matplotlib
```

## Data Preparation
```python
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess an image for the model."""
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return img_array

def display_image(image, title=None):
    """Display an image with optional title."""
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()
```

## Load Pre-trained Model
```python
# Load MobileNetV2 pre-trained on ImageNet
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model
base_model.trainable = False

# Add classification layers
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(2, activation='softmax')  # 2 classes: dog and cat
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

## Training (Fine-tuning)
```python
# Prepare training data
# Note: In a real scenario, you would load your own dataset
# This is just for demonstration
X_train = np.random.random((100, 224, 224, 3))
y_train = np.random.randint(0, 2, (100,))

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=5,
    validation_split=0.2
)
```

## Evaluation
```python
# Prepare test data
X_test = np.random.random((20, 224, 224, 3))
y_test = np.random.randint(0, 2, (20,))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'\nTest accuracy: {test_acc:.2%}')
```

## Prediction
```python
def predict_image(model, image_path):
    """Make a prediction on a single image."""
    # Load and preprocess the image
    img = load_and_preprocess_image(image_path)
    
    # Make prediction
    pred = model.predict(np.expand_dims(img, axis=0))
    class_idx = np.argmax(pred[0])
    confidence = pred[0][class_idx]
    
    # Display results
    display_image(img, f'Predicted: {"Dog" if class_idx == 0 else "Cat"} ({confidence:.2%})')
    
    return class_idx, confidence

# Example usage
# predict_image(model, 'path/to/your/image.jpg')
```

## Testing with Out-of-Distribution Data
```python
def test_ood_image(model, image_path):
    """Test the model with an out-of-distribution image (e.g., a panda)."""
    img = load_and_preprocess_image(image_path)
    pred = model.predict(np.expand_dims(img, axis=0))
    
    # Display results
    display_image(img, f'Dog: {pred[0][0]:.2%}, Cat: {pred[0][1]:.2%}')
    
    return pred[0]

# Example usage
# test_ood_image(model, 'path/to/panda.jpg')
```

## Exercise
1. Try the model with different animal images
2. Compare predictions between similar animals
3. Test with non-animal images to see how the model handles out-of-distribution data

## Discussion Points
- How does the model perform with different poses?
- What happens with poor quality images?
- How does it handle animals not in the training set?

## Next Steps
- Fine-tune the model on your own dataset
- Try different pre-trained models
- Add data augmentation
- Implement transfer learning

<!---
Speaking Notes:
- Emphasize the importance of proper data preprocessing
- Explain why we freeze the base model
- Discuss the trade-offs of using pre-trained models
- Highlight the importance of testing with out-of-distribution data
- Explain how to interpret the model's confidence scores
--> 