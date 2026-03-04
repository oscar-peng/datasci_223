---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Demo 1a: Loading Medical Images (DICOM & Standard Formats)

Medical images come in several formats — DICOM (the clinical standard),
PNG, and JPEG. This demo walks through loading each and converting to
tensors that PyTorch can work with.


## Setup

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Medical imaging
import pydicom
from pydicom.data import get_testdata_file

# PyTorch
import torch
from torchvision import transforms
```

## 1. Loading a DICOM File

DICOM files contain both **pixel data** and **metadata** (patient info,
acquisition parameters, etc.). Let's use one of pydicom's built-in test files.

```python
# Load a DICOM test file
dcm_path = get_testdata_file("CT_small.dcm")
ds = pydicom.dcmread(dcm_path)

# Inspect metadata
print(f"Modality:        {ds.Modality}")
print(f"Patient Name:    {ds.PatientName}")
print(f"Image Size:      {ds.Rows} x {ds.Columns}")
print(f"Bits Allocated:  {ds.BitsAllocated}")
print(f"Pixel Spacing:   {getattr(ds, 'PixelSpacing', 'N/A')}")
```

```python
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
```

## 2. Normalizing DICOM Pixel Values

DICOM pixel values can be in many ranges depending on the modality and bit
depth. We need to normalize them to a standard range before using them with
PyTorch.

```python
# Normalize to [0, 1] range
pixels_float = pixels.astype(np.float32)
pixels_norm = (pixels_float - pixels_float.min()) / (pixels_float.max() - pixels_float.min())
print(f"Normalized range: [{pixels_norm.min():.2f}, {pixels_norm.max():.2f}]")

# Convert to PIL Image (expects uint8 or float32 in [0, 1])
img_pil = Image.fromarray((pixels_norm * 255).astype(np.uint8))
print(f"PIL Image mode: {img_pil.mode}, size: {img_pil.size}")
```

## 3. Loading Standard Image Formats

PNG and JPEG files are straightforward with Pillow.

```python
# If you have a local chest X-ray image, load it like this:
# img = Image.open("chest_xray.png")

# For this demo, we'll create a sample image from the DICOM data
img = img_pil  # our DICOM converted to PIL

print(f"Image mode: {img.mode}")
print(f"Image size: {img.size}")  # (width, height) in PIL

# Convert grayscale to RGB (needed for pretrained models)
img_rgb = img.convert("RGB")
print(f"RGB image mode: {img_rgb.mode}")
```

```python
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
```

## 4. Converting to PyTorch Tensors

PyTorch expects tensors in **(C, H, W)** format — channels first. The
`ToTensor()` transform handles the conversion and scales pixel values from
[0, 255] to [0.0, 1.0].

```python
to_tensor = transforms.ToTensor()

# Grayscale → tensor
tensor_gray = to_tensor(img)
print(f"Grayscale tensor shape: {tensor_gray.shape}")  # (1, H, W)
print(f"Value range: [{tensor_gray.min():.3f}, {tensor_gray.max():.3f}]")

# RGB → tensor
tensor_rgb = to_tensor(img_rgb)
print(f"RGB tensor shape: {tensor_rgb.shape}")          # (3, H, W)
print(f"Value range: [{tensor_rgb.min():.3f}, {tensor_rgb.max():.3f}]")
```

## 5. Visualizing the Conversion Pipeline

Each step changes the data format and value range. Seeing them side by
side makes the pipeline concrete.

```python
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

# Step 1: Raw DICOM — integer pixel values, often 12- or 16-bit
axes[0].imshow(pixels, cmap="gray")
axes[0].set_title(f"1. Raw DICOM\n{pixels.dtype}, [{pixels.min()}, {pixels.max()}]",
                  fontsize=10)
axes[0].axis("off")

# Step 2: Normalized to [0, 1]
axes[1].imshow(pixels_norm, cmap="gray")
axes[1].set_title(f"2. Normalized\nfloat32, [0.0, 1.0]", fontsize=10)
axes[1].axis("off")

# Step 3: PIL Image (uint8, H×W or H×W×3)
axes[2].imshow(img_rgb)
axes[2].set_title(f"3. PIL RGB\nuint8, {img_rgb.size[0]}×{img_rgb.size[1]}×3",
                  fontsize=10)
axes[2].axis("off")

# Step 4: Tensor (float32, C×H×W)
axes[3].imshow(tensor_rgb.permute(1, 2, 0).numpy())
axes[3].set_title(f"4. PyTorch Tensor\nfloat32, {tuple(tensor_rgb.shape)}", fontsize=10)
axes[3].axis("off")

plt.suptitle("DICOM → Tensor: What changes at each step", fontsize=14)
plt.tight_layout()
plt.show()
```

## What's Next?

For more advanced medical imaging pipelines (3D volumes, CT windowing,
DICOM series handling), check out [MONAI](https://monai.io/) — a PyTorch
framework built specifically for healthcare imaging.

