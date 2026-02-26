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
# # Demo 3a: Object Detection with Pretrained Models
#
# In this demo we'll use a pretrained **Faster R-CNN** from torchvision to
# detect objects in images. The model is trained on COCO (80 everyday object
# classes). We'll also briefly show **Ultralytics YOLOv8** as the fastest
# path to detection.

# %% [markdown]
# ## Setup

# %%
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_tensor, to_pil_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## 1. Load Pretrained Faster R-CNN

# %%
# Load model and weights
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)
model = model.to(device)
model.eval()

# Get class labels from the weights metadata
categories = weights.meta["categories"]
print(f"Model trained on {len(categories)} COCO classes")
print(f"First 10 classes: {categories[:10]}")

# %% [markdown]
# ## 2. Prepare an Image
#
# We'll use a sample image. In practice, this could be a medical image,
# a photograph, or a frame from a video.

# %%
# Create a sample image (or load your own: Image.open("my_image.jpg"))
# Download a sample from torchvision
from torchvision.io import read_image

# Use a sample — you can replace with any image path
# For classroom, use a readily available image
try:
    from urllib.request import urlretrieve
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png"
    urlretrieve(url, "/tmp/sample_detection.png")
    img = Image.open("/tmp/sample_detection.png").convert("RGB")
except Exception:
    # Fallback: create a simple test image
    img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))

print(f"Image size: {img.size}")
plt.figure(figsize=(8, 6))
plt.imshow(img)
plt.title("Input Image")
plt.axis("off")
plt.show()

# %% [markdown]
# ## 3. Run Detection

# %%
# Preprocess
preprocess = weights.transforms()
img_tensor = preprocess(img).to(device)

# Run inference
with torch.no_grad():
    predictions = model([img_tensor])[0]

print(f"Detected {len(predictions['boxes'])} objects (before filtering)")
print(f"Prediction keys: {list(predictions.keys())}")

# %%
# Examine raw predictions
for i in range(min(5, len(predictions["boxes"]))):
    box = predictions["boxes"][i].cpu().numpy()
    label = categories[predictions["labels"][i]]
    score = predictions["scores"][i].item()
    print(f"  {label}: {score:.3f} — box: [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")

# %% [markdown]
# ## 4. Filter and Visualize Detections
#
# Keep only high-confidence detections.

# %%
confidence_threshold = 0.5

# Filter predictions
keep = predictions["scores"] > confidence_threshold
boxes = predictions["boxes"][keep]
labels_idx = predictions["labels"][keep]
scores = predictions["scores"][keep]

print(f"Detections above {confidence_threshold} confidence: {len(boxes)}")

# %%
# Draw bounding boxes
label_names = [f"{categories[l]}: {s:.2f}" for l, s in zip(labels_idx, scores)]

# draw_bounding_boxes expects uint8 tensor in (C, H, W) format
img_uint8 = (to_tensor(img) * 255).byte()
result = draw_bounding_boxes(img_uint8, boxes.cpu(), label_names, width=3, font_size=14)

plt.figure(figsize=(10, 8))
plt.imshow(to_pil_image(result))
plt.title(f"Faster R-CNN Detections (confidence > {confidence_threshold})")
plt.axis("off")
plt.show()

# %% [markdown]
# ## 5. Effect of Confidence Threshold
#
# The threshold controls the precision-recall tradeoff: lower threshold =
# more detections (higher recall) but more false positives (lower precision).

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, thresh in zip(axes, [0.3, 0.5, 0.8]):
    keep = predictions["scores"] > thresh
    boxes_t = predictions["boxes"][keep]
    labels_t = predictions["labels"][keep]
    scores_t = predictions["scores"][keep]

    names = [f"{categories[l]}: {s:.2f}" for l, s in zip(labels_t, scores_t)]
    result = draw_bounding_boxes(img_uint8, boxes_t.cpu(), names, width=2)

    ax.imshow(to_pil_image(result))
    ax.set_title(f"Threshold: {thresh} ({len(boxes_t)} detections)")
    ax.axis("off")

plt.suptitle("Effect of Confidence Threshold", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. (Optional) Ultralytics YOLOv8
#
# For the simplest path to detection, Ultralytics provides a one-liner:
#
# ```python
# from ultralytics import YOLO
#
# model = YOLO("yolov8n.pt")  # nano model, fast
# results = model("my_image.jpg")
# results[0].show()
#
# # Fine-tune on custom dataset
# model.train(data="dataset.yaml", epochs=50)
# ```
#
# Install with: `pip install ultralytics`

# %% [markdown]
# ## Discussion: Medical Detection
#
# The pretrained Faster R-CNN detects everyday objects (people, cars, animals).
# For medical detection tasks (nodule detection, cell counting), you would:
#
# 1. **Annotate** your medical images with bounding boxes (using CVAT or Label Studio)
# 2. **Fine-tune** the pretrained detector on your annotated dataset
# 3. **Evaluate** with mAP at appropriate IoU thresholds
#
# The pretrained backbone still helps — it knows about edges, textures, and
# shapes — even if the specific classes are completely different.

# %% [markdown]
# ## Checkpoint
#
# You should now be able to:
# - Load a pretrained detection model from torchvision
# - Run inference and extract boxes, labels, and scores
# - Filter predictions by confidence threshold
# - Visualize detections with `draw_bounding_boxes`
# - Understand the path to medical detection (annotate → fine-tune → evaluate)
