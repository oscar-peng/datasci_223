---
jupyter:
  jupytext:
    formats: md,ipynb
    text_representation:
      extension: .md
      format_name: percent
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# %% [markdown]
# # Demo 2: Transfer Learning for Image Classification
#
# In this demo we'll use a pretrained ResNet-18 to classify images.
# We'll freeze the backbone, replace the classification head, and train
# a binary classifier — the same pipeline you'd use for medical imaging.
#
# **Dataset**: CIFAR-10 filtered to 2 classes (airplane vs automobile) for
# quick iteration. The workflow is identical for medical data — just swap
# in an `ImageFolder` pointing at your chest X-ray directory.

# %% [markdown]
# ## Setup

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## 1. Prepare the Data
#
# We'll use CIFAR-10 with just 2 classes for quick iteration.
# To use your own images, swap `Subset(CIFAR10(...))` with
# `ImageFolder("data/your_dataset/")` — everything else stays the same.

# %%
# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# %%
# Option A: Use CIFAR-10 subset (2 classes, quick download)
full_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
full_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=eval_transform)

# Filter to just 2 classes (airplane=0, automobile=1) for binary classification
def filter_classes(dataset, classes=(0, 1)):
    indices = [i for i, (_, label) in enumerate(dataset) if label in classes]
    return torch.utils.data.Subset(dataset, indices)

train_data = filter_classes(full_train)
test_data = filter_classes(full_test)

# Option B: Use ImageFolder for real medical data
# train_data = datasets.ImageFolder("data/montgomery/train", transform=train_transform)
# test_data = datasets.ImageFolder("data/montgomery/test", transform=eval_transform)

print(f"Training samples: {len(train_data)}")
print(f"Test samples:     {len(test_data)}")

# %%
# Split training into train + val
train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_set, val_set = random_split(train_data, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2)

print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_data)}")

# %% [markdown]
# ## 2. Load Pretrained ResNet-18
#
# We load a ResNet-18 pretrained on ImageNet, freeze the backbone, and
# replace the final classification layer.

# %%
# Load pretrained model
model = models.resnet18(weights="DEFAULT")

# Look at the original classification head
print(f"Original fc layer: {model.fc}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# %%
# Freeze all backbone parameters
for param in model.parameters():
    param.requires_grad = False

# Replace the classification head for binary classification
model.fc = nn.Sequential(
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 2),
)

# Count trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

model = model.to(device)

# %% [markdown]
# ## 3. Train the Model
#
# We'll train for a few epochs — only the classification head is learning.

# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

num_epochs = 5
history = {"train_loss": [], "val_loss": [], "val_acc": []}

for epoch in range(num_epochs):
    # --- Training ---
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    history["train_loss"].append(avg_train_loss)

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100.0 * correct / total
    history["val_loss"].append(avg_val_loss)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch+1}/{num_epochs} — "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, "
          f"Val Acc: {val_acc:.1f}%")

# %% [markdown]
# ## 4. Plot Training Curves

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(history["train_loss"], label="Train Loss")
ax1.plot(history["val_loss"], label="Val Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Loss Curves")
ax1.legend()

ax2.plot(history["val_acc"], label="Val Accuracy", color="green")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Validation Accuracy")
ax2.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Evaluate on Test Set

# %%
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Classification report
class_names = ["airplane", "automobile"]
print(classification_report(all_labels, all_preds, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Visualize Predictions

# %%
def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return t.clamp_(0, 1)

# Get a batch of test images
images, labels = next(iter(test_loader))
images_device = images.to(device)

model.eval()
with torch.no_grad():
    outputs = model(images_device)
    probs = torch.softmax(outputs, dim=1)
    _, preds = outputs.max(1)

# Display predictions
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for i, ax in enumerate(axes.flat):
    img = unnormalize(images[i].clone())
    ax.imshow(img.permute(1, 2, 0).numpy())
    color = "green" if preds[i] == labels[i] else "red"
    ax.set_title(
        f"Pred: {class_names[preds[i]]} ({probs[i][preds[i]]:.1%})\n"
        f"True: {class_names[labels[i]]}",
        color=color, fontsize=10
    )
    ax.axis("off")
plt.suptitle("Model Predictions (green=correct, red=wrong)", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Save the Model

# %%
torch.save(model.state_dict(), "resnet18_classifier.pt")
print("Model saved to resnet18_classifier.pt")

# To load later:
# model.load_state_dict(torch.load("resnet18_classifier.pt"))

# %% [markdown]
# ## Checkpoint
#
# You should now be able to:
# - Load a pretrained model and freeze its backbone
# - Replace the classification head for a new task
# - Write a PyTorch training loop (forward → loss → backward → step)
# - Evaluate with confusion matrix and classification report
# - Save and load model weights
