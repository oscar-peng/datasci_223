---
lecture_number: 08
pdf: true
---

# Computer Vision

## Outline

1. **Image Data & CNN Fundamentals (approx. 30 min)**
    * Briefly: What is Computer Vision? (Context: Enabling computers to interpret visual data).
    * Digital Image Representation:
        * Pixels, Resolution, Color Spaces (Grayscale, RGB).
        * Medical Image Specifics: Introduction to **DICOM** (metadata, series, pixel data, windowing).
    * Essential Python Libraries: Pillow and OpenCV for loading, basic properties (dimensions, mode), pixel access.
    * Introduction to Image Features: Why raw pixels are often not enough (briefly mention handcrafted ideas like edges/corners).
    * Convolutional Neural Networks (CNNs): Automated, hierarchical feature learning.
        * Core Components:
            * Convolutional Layer (Filters/Kernels, Shared Weights, Stride, Padding, Feature Maps, Receptive Field).
            * Activation Functions (ReLU).
            * Pooling Layer (Max Pooling, Average Pooling).
            * Fully Connected (Dense) Layer.
        * Building a CNN: Stacking layers, typical flow.
        * Hierarchical Feature Learning (Visual examples of features learned at different depths).
        * Ethical Note: Brief mention of how biases in training data (e.g., medical images from specific demographics) can affect CNN feature learning and outcomes.
    * **Reference Card:** Key CNN Layers (`Conv2D`, `MaxPool2D`, `Dense`, `ReLU`).
    * **Reference Card:** Pillow & OpenCV: Core functions for image loading, saving, property access.

2. **Demo 1: Image Ops & Convolution Demo (approx. 10 min)**
    * **Task:**
        * Part 1 (Pillow): Load an image. Basic manipulations: cropping, color space conversion (e.g., to grayscale), drawing shapes (e.g., rectangle for annotation).
        * Part 2 (OpenCV/Numpy): Visually demonstrate convolution by applying a simple kernel (e.g., edge detection, blur) to a small image/patch. Show input, kernel, output.
    * **Tools:** Python, Pillow, OpenCV, Numpy, Matplotlib.

3. **Object Detection (approx. 25 min)**
    * Task: Classification + Localization (Bounding Boxes).
    * Challenges: Scale variation, occlusion, multiple objects.
    * Approaches:
        * Two-Stage Detectors (e.g., **Faster R-CNN**: Region Proposal Networks).
        * One-Stage Detectors (e.g., **YOLO**: Grid-based, anchor boxes; SSD: Multi-scale features).
        * (Show examples of these applied to medical scenarios like tumor or instrument detection).
    * Key Concepts: Anchor Boxes, Non-Maximum Suppression (NMS).
    * Evaluation: Intersection over Union (IoU), Mean Average Precision (mAP).
    * **Reference Card:** IoU calculation and mAP definition.

4. **Demo 2: Pre-trained Object Detection (approx. 10 min)**
    * **Task:** Use a pre-trained object detection model (e.g., YOLO via OpenCV DNN or Ultralytics, or SSD MobileNet via TF Hub/PyTorch Hub).
    * **Example:** Detect objects in a general image (for clarity and robustness) or a health-related image if a simple, effective pre-trained model is available (e.g., detecting cells, medical equipment).
    * **Focus:** Loading model, image prep, inference, interpreting output (boxes, classes, scores), NMS, visualizing results.
    * **Tools:** Python, OpenCV (DNN) / Ultralytics, or TensorFlow/Keras / PyTorch.

5. **Image Segmentation with U-Net (approx. 20 min)**
    * Task: Pixel-level classification (assigning a class to every pixel).
    * Types: Semantic Segmentation (classifying regions) vs. Instance Segmentation (identifying individual objects).
    * **U-Net Architecture:**
        * Significance in medical image segmentation.
        * Encoder-Decoder Structure: Contracting path (feature extraction) and Expansive path (localization).
        * Skip Connections: Combining deep, semantic features with shallow, fine-grained features. (Illustrate with a U-Net diagram).
    * Applications: Delineating organs, segmenting tumors/lesions for measurement, cell segmentation in microscopy.
    * Loss Functions for Segmentation (briefly): Dice Loss, Jaccard Loss (IoU Loss).
    * **Reference Card:** U-Net architecture diagram & Dice/Jaccard Loss.

6. **Demo 3: Transfer Learning for Classification (approx. 10 min)**
    * **Task:** Adapt a pre-trained CNN for a medical image classification task.
    * **Dataset:** MedMNIST subset (e.g., PathMNIST, DermaMNIST, ChestMNIST).
    * **Method:** Load pre-trained model (e.g., MobileNetV2) without top, add new classifier, freeze base, train new top.
    * **Tools:** Python, TensorFlow/Keras or PyTorch.

7. **Transfer Learning & Other Advanced Concepts (approx. 10 min)**
    * **Transfer Learning Strategies (Recap & Detail):**
        * Feature Extraction.
        * Fine-tuning (unfreezing layers, learning rates).
    * Briefly:
        * Vision Transformers (ViT): Images as sequences, self-attention.
        * Generative Models (GANs, Diffusion): Synthetic medical data, image enhancement.
        * Explainable AI (XAI) in CV (Grad-CAM, LIME, SHAP for medical model trust).

8. **CV Toolkit & Further Learning (approx. 5 min)**
    * Key Libraries: OpenCV, Pillow, Scikit-image, TensorFlow/Keras, PyTorch.
    * Data Augmentation: Albumentations.
    * Medical CV: MONAI, SimpleITK.
    * Annotation Tools: ITK-SNAP, 3D Slicer, LabelMe/CVAT.
    * Datasets: TCIA, Grand Challenge, MedMNIST, Kaggle.
    * Learning Resources: Courses, conferences (MICCAI, CVPR), blogs.
    * Recap & Q&A.
