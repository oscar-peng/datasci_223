---
lecture_number: 08
pdf: true
---

# Computer Vision for Health Data Science - Lecture 08 Outline

## 1. **Introduction to Computer Vision & Image Data (25 min)**

* What is Computer Vision? Brief overview and relevance to health data science
* Digital Image Representation:
    * Pixels, Resolution, Color Spaces (Grayscale, RGB)
    * Medical Image Formats: DICOM overview (metadata, pixel data, windowing)
* Essential Python Libraries for Medical Imaging:
    * Pillow: Basic image loading and manipulation
    * OpenCV: Advanced image processing
    * Medical-specific: Brief intro to SimpleITK, pydicom
* Image Features: Why raw pixels aren't enough
    * Traditional approaches (brief mention of handcrafted features)
    * Deep learning approach: Automated feature learning
* **Reference Card:** Image loading and basic operations with Pillow/OpenCV

## 2. **Demo 1: Image Loading & Basic Processing (10 min)**

* **Task:** Load and preprocess medical images (DICOM and standard formats)
* **Operations:**
    * Loading images with appropriate libraries
    * Basic inspection (dimensions, pixel values, metadata)
    * Preprocessing: Cropping, normalization, color conversion
    * Visualization with matplotlib
* **Tools:** Python, Pillow, OpenCV, matplotlib, pydicom (optional)

## 3. **Convolutional Neural Networks for Medical Imaging (20 min)**

* CNN Architecture (building on previous neural network knowledge):
    * Convolutional Layer (Filters/Kernels, Stride, Padding, Feature Maps)
    * Activation Functions (ReLU)
    * Pooling Layer (Max Pooling, Average Pooling)
    * Fully Connected (Dense) Layer
* Hierarchical Feature Learning (visual examples from medical images)
* Common CNN Architectures in Medical Imaging:
    * VGG, ResNet, DenseNet (brief overview)
* Ethical Considerations: Bias in medical image datasets
* **Reference Card:** Key CNN Layers (`Conv2D`, `MaxPool2D`, `Dense`, `ReLU`)
* **Minimal Example:** Simple CNN for medical image classification

## 4. **Demo 2: Transfer Learning for Medical Image Classification (10 min)**

* **Task:** Adapt a pre-trained CNN for a medical image classification task
* **Dataset:** MedMNIST subset (e.g., ChestMNIST, DermaMNIST)
* **Method:**
    * Load pre-trained model (e.g., ResNet, MobileNet)
    * Freeze base layers
    * Add new classification head
    * Train and evaluate
* **Tools:** Python, TensorFlow/Keras or PyTorch

## 5. **Object Detection in Medical Imaging (15 min)**

* Task: Classification + Localization (Bounding Boxes)
* Challenges: Scale variation, occlusion, multiple objects
* Approaches:
    * Two-Stage Detectors (Faster R-CNN)
    * One-Stage Detectors (YOLO)
* Key Concepts:
    * Anchor Boxes
    * Non-Maximum Suppression (NMS)
* Evaluation: Intersection over Union (IoU), Mean Average Precision (mAP)
* Medical Applications: Tumor detection, instrument tracking, anatomical landmark detection
* **Reference Card:** IoU calculation and visualization

## 6. **Image Segmentation with U-Net (15 min)**

* Task: Pixel-level classification
* Types: Semantic vs. Instance Segmentation
* U-Net Architecture:
    * Significance in medical image segmentation
    * Encoder-Decoder Structure
    * Skip Connections
* Applications: Organ segmentation, tumor delineation, cell segmentation
* Loss Functions for Segmentation: Dice Loss, Jaccard Loss
* **Reference Card:** U-Net architecture diagram

## 7. **Demo 3: Pre-trained Object Detection or Segmentation (10 min)**

* **Option A: Object Detection**
    * Use a pre-trained YOLO model to detect objects in medical images
    * Focus on model loading, inference, result visualization
* **Option B: Segmentation**
    * Use a pre-trained U-Net model for medical image segmentation
    * Focus on model loading, inference, visualization of segmentation masks
* **Tools:** Python, Ultralytics YOLO or TensorFlow/Keras

## 8. **Advanced Topics & Future Directions (15 min)**

* Vision Transformers (ViT) in Medical Imaging:
    * Self-attention mechanism for images
    * Advantages over CNNs for certain medical tasks
* Generative Models in Healthcare:
    * GANs and Diffusion Models for synthetic medical data
    * Image enhancement and reconstruction
* Explainable AI (XAI) for Medical CV:
    * Grad-CAM, LIME, SHAP for model interpretability
    * Importance in clinical applications
* Practical Tools & Resources:
    * Medical CV Libraries: MONAI, SimpleITK
    * Datasets: TCIA, Grand Challenge, MedMNIST
    * Data Augmentation: Albumentations
    * Annotation Tools: ITK-SNAP, 3D Slicer
* **Reference Card:** Key resources for medical computer vision
