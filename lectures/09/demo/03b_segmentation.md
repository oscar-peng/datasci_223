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
# # Demo 3b: Image Segmentation
#
# In this demo we'll explore image segmentation — classifying every pixel
# in an image. We'll use:
# 1. A pretrained **DeepLabV3** from torchvision for general segmentation
# 2. **segmentation_models_pytorch** (smp) to create a U-Net for medical use

# %% [markdown]
# ## Setup

# %%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_tensor, to_pil_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## Part 1: Pretrained DeepLabV3
#
# torchvision provides pretrained segmentation models. DeepLabV3 with a
# ResNet-50 backbone is trained on COCO for 21 classes (background + 20
# object categories).

# %%
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

# Load pretrained model
weights = DeepLabV3_ResNet50_Weights.DEFAULT
model = deeplabv3_resnet50(weights=weights)
model = model.to(device)
model.eval()

# Get preprocessing transforms
preprocess = weights.transforms()

# Class names
categories = weights.meta["categories"]
print(f"Segmentation classes ({len(categories)}): {categories}")

# %% [markdown]
# ## Run Segmentation

# %%
# Create or load an image
# In practice: img = Image.open("chest_xray.png").convert("RGB")
# For demo, create a simple color image or use a sample
try:
    from urllib.request import urlretrieve
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png"
    urlretrieve(url, "/tmp/sample_seg.png")
    img = Image.open("/tmp/sample_seg.png").convert("RGB")
except Exception:
    img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))

# Preprocess and run inference
img_tensor = preprocess(img).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img_tensor)["out"]  # shape: (1, num_classes, H, W)

# Get per-pixel class predictions
predictions = output.argmax(dim=1).squeeze(0).cpu().numpy()
print(f"Output shape: {output.shape}")
print(f"Prediction map shape: {predictions.shape}")
print(f"Unique classes found: {np.unique(predictions)}")

# %%
# Visualize original vs segmentation mask
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(predictions, cmap="tab20")
axes[1].set_title("Segmentation Mask")
axes[1].axis("off")

# Overlay
img_array = np.array(img.resize(predictions.shape[::-1]))
overlay = img_array.copy()
mask_colored = plt.cm.tab20(predictions / predictions.max())[:, :, :3] * 255
overlay = (0.6 * overlay + 0.4 * mask_colored).astype(np.uint8)
axes[2].imshow(overlay)
axes[2].set_title("Overlay")
axes[2].axis("off")

plt.suptitle("DeepLabV3 Semantic Segmentation", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Part 2: U-Net with segmentation_models_pytorch
#
# For medical segmentation tasks, `segmentation_models_pytorch` (smp) makes
# it easy to create a U-Net with a pretrained backbone.
#
# Install: `pip install segmentation-models-pytorch`

# %%
try:
    import segmentation_models_pytorch as smp

    # Create a U-Net with ResNet-18 encoder
    unet = smp.Unet(
        encoder_name="resnet18",       # backbone
        encoder_weights="imagenet",    # pretrained on ImageNet
        in_channels=1,                 # grayscale input (chest X-ray)
        classes=1,                     # binary: lung vs background
        activation="sigmoid",         # output probabilities
    )

    print(f"Model created: U-Net with ResNet-18 encoder")
    print(f"Input: 1 channel (grayscale)")
    print(f"Output: 1 channel (binary mask)")

    # Test with a dummy input
    dummy_input = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        dummy_output = unet(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {dummy_output.shape}")
    print(f"Output range: [{dummy_output.min():.3f}, {dummy_output.max():.3f}]")

except ImportError:
    print("segmentation_models_pytorch not installed.")
    print("Install with: pip install segmentation-models-pytorch")

# %% [markdown]
# ## Working with Segmentation Masks
#
# Segmentation masks are just images where each pixel value represents a
# class label. For binary segmentation (lung vs background), the mask has
# values 0 (background) and 1 (lung).

# %%
# Simulate a chest X-ray and its lung mask
# In practice, these come from annotated datasets
np.random.seed(42)
h, w = 256, 256

# Create a fake "chest X-ray" (dark background with lighter regions)
fake_xray = np.random.normal(80, 30, (h, w)).clip(0, 255).astype(np.uint8)
# Add brighter "lung" regions
fake_xray[60:200, 30:100] += 50  # left lung
fake_xray[60:200, 156:226] += 50  # right lung
fake_xray = fake_xray.clip(0, 255).astype(np.uint8)

# Create corresponding mask
fake_mask = np.zeros((h, w), dtype=np.uint8)
fake_mask[60:200, 30:100] = 1   # left lung
fake_mask[60:200, 156:226] = 1  # right lung

# Display
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(fake_xray, cmap="gray")
axes[0].set_title("Chest X-ray (simulated)")
axes[0].axis("off")

axes[1].imshow(fake_mask, cmap="gray")
axes[1].set_title("Ground Truth Mask")
axes[1].axis("off")

# Overlay
overlay = np.stack([fake_xray]*3, axis=-1)
overlay[fake_mask == 1] = [255, 100, 100]  # red overlay on lungs
axes[2].imshow(overlay)
axes[2].set_title("Mask Overlay")
axes[2].axis("off")

plt.suptitle("Segmentation: Image + Mask", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Computing Dice Score
#
# The Dice coefficient measures overlap between predicted and ground truth
# masks. It ranges from 0 (no overlap) to 1 (perfect match).

# %%
def dice_score(pred, target, threshold=0.5):
    """Compute Dice coefficient between predicted and target masks."""
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    return (2.0 * intersection) / (pred_binary.sum() + target.sum() + 1e-8)

# Simulate a model prediction (imperfect)
pred_mask = fake_mask.copy().astype(np.float32)
# Add some noise to simulate imperfect prediction
pred_mask += np.random.normal(0, 0.3, pred_mask.shape)
pred_mask = np.clip(pred_mask, 0, 1)

# Compute Dice
pred_tensor = torch.tensor(pred_mask)
target_tensor = torch.tensor(fake_mask, dtype=torch.float32)
score = dice_score(pred_tensor, target_tensor)
print(f"Dice Score: {score:.4f}")

# A perfect prediction would score 1.0
perfect_score = dice_score(target_tensor, target_tensor)
print(f"Perfect Dice Score: {perfect_score:.4f}")

# %% [markdown]
# ## Checkpoint
#
# You should now be able to:
# - Run pretrained DeepLabV3 for semantic segmentation
# - Create a U-Net with smp using a pretrained encoder
# - Understand segmentation masks (per-pixel labels)
# - Compute and interpret the Dice coefficient
# - Visualize segmentation results with overlays
