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
# # Demo 1a: Loading Medical Images (DICOM & Standard Formats)
#
# In this demo we'll load images in several formats — DICOM (the medical
# standard), PNG, and JPEG — and convert them into tensors that PyTorch can work
# with.

# %% [markdown]
# ## Setup

# %%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Medical imaging
import pydicom
from pydicom.data import get_testdata_file

# PyTorch
import torch
from torchvision import transforms

# %% [markdown]
# ## 1. Loading a DICOM File
#
# DICOM files contain both **pixel data** and **metadata** (patient info,
# acquisition parameters, etc.). Let's use one of pydicom's built-in test files.

# %%
# Load a DICOM test file
dcm_path = get_testdata_file("CT_small.dcm")
ds = pydicom.dcmread(dcm_path)

# Inspect metadata
print(f"Modality:        {ds.Modality}")
print(f"Patient Name:    {ds.PatientName}")
print(f"Image Size:      {ds.Rows} x {ds.Columns}")
print(f"Bits Allocated:  {ds.BitsAllocated}")
print(f"Pixel Spacing:   {getattr(ds, 'PixelSpacing', 'N/A')}")

# %%
# Extract pixel data
pixels = ds.pixel_array
print(f"Pixel array shape: {pixels.shape}")
print(f"Pixel dtype:       {pixels.dtype}")
print(f"Value range:       [{pixels.min()}, {pixels.max()}]")

# Display the image
plt.figure(figsize=(6, 6))
plt.imshow(pixels, cmap="gray")
plt.title(f"DICOM: {ds.Modality} — {ds.Rows}×{ds.Columns}")
plt.colorbar(label="Pixel intensity")
plt.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Normalizing DICOM Pixel Values
#
# DICOM pixel values can be in many ranges depending on the modality and bit
# depth. We need to normalize them to a standard range before using them with
# PyTorch.

# %%
# Normalize to [0, 1] range
pixels_float = pixels.astype(np.float32)
pixels_norm = (pixels_float - pixels_float.min()) / (pixels_float.max() - pixels_float.min())
print(f"Normalized range: [{pixels_norm.min():.2f}, {pixels_norm.max():.2f}]")

# Convert to PIL Image (expects uint8 or float32 in [0, 1])
img_pil = Image.fromarray((pixels_norm * 255).astype(np.uint8))
print(f"PIL Image mode: {img_pil.mode}, size: {img_pil.size}")

# %% [markdown]
# ## 3. Loading Standard Image Formats
#
# PNG and JPEG files are straightforward with Pillow.

# %%
# If you have a local chest X-ray image, load it like this:
# img = Image.open("chest_xray.png")

# For this demo, we'll create a sample image from the DICOM data
img = img_pil  # our DICOM converted to PIL

print(f"Image mode: {img.mode}")
print(f"Image size: {img.size}")  # (width, height) in PIL

# Convert grayscale to RGB (needed for pretrained models)
img_rgb = img.convert("RGB")
print(f"RGB image mode: {img_rgb.mode}")

# %%
# Display side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img, cmap="gray")
axes[0].set_title("Grayscale")
axes[0].axis("off")
axes[1].imshow(img_rgb)
axes[1].set_title("RGB (3 channels)")
axes[1].axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Converting to PyTorch Tensors
#
# PyTorch expects tensors in **(C, H, W)** format — channels first. The
# `ToTensor()` transform handles the conversion and scales pixel values from
# [0, 255] to [0.0, 1.0].

# %%
to_tensor = transforms.ToTensor()

# Grayscale → tensor
tensor_gray = to_tensor(img)
print(f"Grayscale tensor shape: {tensor_gray.shape}")  # (1, H, W)
print(f"Value range: [{tensor_gray.min():.3f}, {tensor_gray.max():.3f}]")

# RGB → tensor
tensor_rgb = to_tensor(img_rgb)
print(f"RGB tensor shape: {tensor_rgb.shape}")          # (3, H, W)
print(f"Value range: [{tensor_rgb.min():.3f}, {tensor_rgb.max():.3f}]")

# %% [markdown]
# ## 5. The Full Pipeline: DICOM → Tensor
#
# Here's the complete conversion path you'll use when working with DICOM
# files in a PyTorch pipeline.

# %%
def dicom_to_tensor(dicom_path, target_size=(224, 224)):
    """Load a DICOM file and return a normalized RGB tensor."""
    # Read DICOM
    ds = pydicom.dcmread(dicom_path)
    pixels = ds.pixel_array.astype(np.float32)

    # Normalize to [0, 255]
    pixels = (pixels - pixels.min()) / (pixels.max() - pixels.min()) * 255
    pixels = pixels.astype(np.uint8)

    # Convert to PIL and make RGB
    img = Image.fromarray(pixels).convert("RGB")

    # Apply transforms
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(img)

# Test it
tensor = dicom_to_tensor(dcm_path)
print(f"Output tensor shape: {tensor.shape}")
print(f"Output tensor dtype: {tensor.dtype}")
print(f"Ready for a pretrained model!")

