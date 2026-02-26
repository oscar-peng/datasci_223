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
# # Demo 1b: Torchvision Transforms & DataLoaders
#
# In this demo we'll build image preprocessing pipelines with
# `torchvision.transforms`, apply data augmentation, and create DataLoaders
# that feed batches to our models.

# %% [markdown]
# ## Setup

# %%
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from PIL import Image
import numpy as np

# %% [markdown]
# ## 1. Transform Pipelines
#
# Transforms are composable — chain them together with `Compose` to build a
# preprocessing pipeline.

# %%
# Training transforms: augmentation + normalization
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Evaluation transforms: deterministic resize + normalize only
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

print("Train transform pipeline:")
for i, t in enumerate(train_transform.transforms):
    print(f"  {i+1}. {t}")

# %% [markdown]
# ## 2. Visualizing Augmentations
#
# Let's see how augmentation transforms modify an image. Each application
# produces a different random variant.

# %%
# Create a sample image (or load one: Image.open("chest_xray.png"))
# Using CIFAR-10 for a quick demo — you'd use chest X-rays in practice
sample_dataset = datasets.CIFAR10(root="./data", train=True, download=True)
sample_img, sample_label = sample_dataset[0]
class_names = sample_dataset.classes
print(f"Sample image: {class_names[sample_label]}, size: {sample_img.size}")

# %%
# Augmentation-only transforms (no normalize, for visualization)
augment_viz = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
])

# Generate 8 augmented versions of the same image
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for i, ax in enumerate(axes.flat):
    if i == 0:
        # Show original
        ax.imshow(sample_img.resize((224, 224)))
        ax.set_title("Original", fontweight="bold")
    else:
        augmented = augment_viz(sample_img)
        ax.imshow(augmented)
        ax.set_title(f"Augmented #{i}")
    ax.axis("off")
plt.suptitle("Data Augmentation: Same image, different random transforms", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Creating a Dataset with ImageFolder
#
# `ImageFolder` expects a directory where each subdirectory is a class:
#
# ```
# data/chest_xrays/
# ├── normal/
# │   ├── img_001.png
# │   └── ...
# └── tuberculosis/
#     ├── img_101.png
#     └── ...
# ```
#
# For this demo, we'll use CIFAR-10 (it downloads automatically). In
# practice, you'd organize your chest X-rays into this folder structure.

# %%
# Load CIFAR-10 with our transform pipelines
train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=train_transform
)
test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=eval_transform
)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples:     {len(test_dataset)}")
print(f"Classes:          {train_dataset.classes}")

# %%
# Split training into train + validation
train_size = int(0.85 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(
    train_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)
print(f"Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_dataset)}")

# %% [markdown]
# ## 4. Building DataLoaders
#
# DataLoaders handle batching, shuffling, and parallel data loading.

# %%
train_loader = DataLoader(
    train_subset,
    batch_size=32,
    shuffle=True,        # randomize order each epoch
    num_workers=2,       # parallel data loading (set 0 for debugging)
    pin_memory=True,     # faster CPU→GPU transfer
)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

print(f"Batches per epoch: {len(train_loader)}")
print(f"Batch size: 32")

# %%
# Peek at a batch
images, labels = next(iter(train_loader))
print(f"Batch images shape: {images.shape}")  # (32, 3, 224, 224)
print(f"Batch labels shape: {labels.shape}")  # (32,)
print(f"Image dtype: {images.dtype}, range: [{images.min():.2f}, {images.max():.2f}]")

# %% [markdown]
# ## 5. Visualizing a Batch
#
# Let's display a batch of images. Since they're normalized, we need to
# "un-normalize" them for display.

# %%
def unnormalize(tensor, mean, std):
    """Reverse normalization for display."""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor.clamp_(0, 1)

# Display first 8 images from the batch
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for i, ax in enumerate(axes.flat):
    img = unnormalize(images[i].clone(), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # Convert from (C, H, W) to (H, W, C) for matplotlib
    ax.imshow(img.permute(1, 2, 0).numpy())
    ax.set_title(f"{train_dataset.classes[labels[i]]}")
    ax.axis("off")
plt.suptitle("Sample batch from DataLoader (normalized + augmented)", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Checkpoint
#
# You should now be able to:
# - Build train vs eval transform pipelines with `Compose`
# - Apply and visualize data augmentation
# - Create datasets with `ImageFolder` or built-in datasets
# - Build DataLoaders with batching, shuffling, and parallel loading
# - Iterate over batches and inspect tensor shapes
