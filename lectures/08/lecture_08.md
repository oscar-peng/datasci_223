---
lecture_number: 08
pdf: true
title: "Computer Vision for Health Data Science"
---

# Computer Vision for Health Data Science - Lecture 08

## Overall Learning Objectives

* Understand fundamental concepts of digital image representation and processing.
* Grasp the architecture and core components of Convolutional Neural Networks (CNNs) for image analysis.
* Learn about key computer vision tasks: image classification, object detection, and image segmentation, with a focus on medical applications.
* Gain hands-on experience loading, processing, and analyzing medical images using Python libraries.
* Be aware of transfer learning techniques and their utility in medical imaging.
* Recognize advanced topics and resources for further learning in medical computer vision, including an introduction to video object tracking.

## Outline

1. Introduction to Computer Vision & Image Data (20 min)
    * **What is Computer Vision?**
        * Brief overview and its significance in health data science.
        * Examples: Radiograph analysis, pathology slide interpretation, robotic surgery guidance.
    * **Digital Image Representation:**
        * Pixels, Resolution, Aspect Ratio.
        * Color Spaces: Grayscale, RGB.
    * **Medical Image Formats:**
        * **DICOM (Digital Imaging and Communications in Medicine):**
            * Overview: Structure, metadata (tags), pixel data.
            * Importance of Series, Study, Instance.
            * Concept of Windowing/Leveling for visualization.
        * Other common formats (PNG, JPEG, TIFF) and their use cases.
    * **Essential Python Libraries for Imaging:**
        * **Pillow (PIL Fork):** Basic image loading, manipulation, and saving.
        * **OpenCV (`cv2`):** Advanced image processing, feature detection, and CV algorithms.
        * **Pydicom & SimpleITK:** For reading, writing, and manipulating DICOM files.
        * **Matplotlib:** For image visualization.
    * **Reference Card:** Pillow & OpenCV: Core functions for image loading, saving, property access.
        * `Image.open()`, `Image.save()`, `image.size`, `image.mode` (Pillow)
        * `cv2.imread()`, `cv2.imwrite()`, `image.shape`, `cv2.cvtColor()` (OpenCV)
        * `pydicom.dcmread()` (Pydicom)
2. Convolutional Neural Networks (CNNs) for Image Analysis (20 min)
    * **Why CNNs for Images?**
        * Limitations of traditional NNs for image data.
        * Local connectivity, parameter sharing, and hierarchical feature learning.
    * **Core CNN Components (Building on previous NN knowledge):**
        * **Convolutional Layer:**
            * Filters/Kernels (and their role in feature detection).
            * Shared Weights & Parameter Efficiency.
            * Stride and Padding.
            * Feature Maps (Activation Maps).
        * **Activation Functions (e.g., ReLU):** Introducing non-linearity.
        * **Pooling Layer (e.g., Max Pooling):** Down-sampling and invariance.
        * **Fully Connected (Dense) Layer:** For classification based on learned features.
    * **Hierarchical Feature Learning:**
        * Visual examples: Early layers learn simple features (edges, corners), deeper layers learn more complex patterns (textures, parts of objects, objects).
    * **Building a CNN:** Stacking layers, typical architectural patterns (e.g., Conv-ReLU-Pool blocks).
    * **Ethical Note:** Brief mention of how biases in training data (e.g., medical images from specific demographics) can affect CNN feature learning and outcomes.
    * **Reference Card:** Key CNN Layers (TensorFlow/Keras syntax).
        * `tf.keras.layers.Conv2D()`
        * `tf.keras.layers.MaxPooling2D()`
        * `tf.keras.layers.Dense()`
        * `tf.keras.layers.ReLU()` / `activation='relu'`
    * **Minimal Example:** Conceptual structure of a simple CNN for image classification.

        ```python
        # Conceptual Keras model
        # model = Sequential([
        #   Conv2D(filters, kernel_size, activation='relu', input_shape=(...)),
        #   MaxPooling2D(),
        #   Conv2D(filters, kernel_size, activation='relu'),
        #   MaxPooling2D(),
        #   Flatten(),
        #   Dense(units, activation='relu'),
        #   Dense(num_classes, activation='softmax')
        # ])
        ```

3. Demo 1: Image Loading, Preprocessing & Basic CNN (10 min)

    * **File:** `lectures/08/demo/01_image_cnn_basics.md`
    * **Content:** Covers loading DICOM/standard images, inspection, preprocessing (resize, normalize, color convert), visualization, and optionally building a basic CNN structure.

4. Key Computer Vision Tasks & Techniques (25 min)
    1. Image Classification
        * Task: Assigning a label to an entire image (e.g., "normal" vs. "abnormal" chest X-ray).
        * Recap: Using CNNs as feature extractors followed by a classifier.
        * Common CNN Architectures (brief overview of their significance):
            * LeNet, AlexNet (historical context)
            * VGG, ResNet, Inception, DenseNet, MobileNet (key ideas like depth, residual connections, efficiency).

    2. Transfer Learning for Medical Image Classification

        * Concept: Using knowledge from models pre-trained on large datasets (e.g., ImageNet) for new, often smaller, medical datasets.
        * Why it's crucial for medical imaging (limited annotated data).
        * Strategies:
            * **Feature Extraction:** Use pre-trained CNN as a fixed feature extractor, train a new classifier on top.
            * **Fine-Tuning:** Unfreeze some of the later layers of the pre-trained CNN and train them on the new dataset with a small learning rate.
        * **Reference Card:** Transfer Learning Steps.
            1. Load pre-trained model (e.g., `tf.keras.applications.ResNet50(weights='imagenet', include_top=False)`).
            2. Freeze base model layers (`base_model.trainable = False`).
            3. Add new custom classification layers.
            4. Compile and train the new model on your data.
            5. (Optional) Fine-tune by unfreezing some top layers of the base model and re-training with a very low learning rate.

    3. Object Detection

        * Task: Classification + Localization (drawing bounding boxes around detected objects).
            * Examples: Finding nodules in lung CTs, identifying cells in microscopy.
        * Key Concepts:
            * Bounding Boxes (representation: x, y, width, height or x_min, y_min, x_max, y_max).
            * Anchor Boxes (predefined boxes of various scales/ratios).
            * Non-Maximum Suppression (NMS) (to remove redundant overlapping boxes).
        * Approaches (briefly):
            * Two-Stage Detectors (e.g., Faster R-CNN: region proposal then classification).
            * One-Stage Detectors (e.g., YOLO, SSD: direct prediction of boxes and classes).
        * Evaluation Metrics:
            * Intersection over Union (IoU).
            * Mean Average Precision (mAP).
        * **Reference Card:** IoU Calculation.
            * `IoU = Area_of_Overlap / Area_of_Union`

5. Demo 2: Transfer Learning for Medical Image Classification (10 min)

    * **File:** `lectures/08/demo/02_transfer_learning_classification.md`
    * **Content:** Demonstrates adapting a pre-trained CNN (e.g., MobileNetV2) for medical image classification using a MedMNIST subset, including model modification, training, and evaluation.

6. Image Segmentation (15 min)

    * Task: Pixel-level classification (assigning a class label to each pixel in an image).
        * Examples: Segmenting organs, delineating tumors, identifying different tissue types.
    * Types:
        * **Semantic Segmentation:** Labeling all pixels belonging to a class (e.g., all "tumor" pixels).
        * **Instance Segmentation:** Differentiating instances of the same class (e.g., "tumor_1", "tumor_2"). (Mention briefly, focus on semantic).
    * **U-Net Architecture:**
        * Significance and widespread use in medical image segmentation.
        * Encoder-Decoder Structure:
            * Contracting Path (Encoder): Captures context, extracts features (similar to a classification CNN).
            * Expanding Path (Decoder): Recovers spatial resolution, localizes features.
        * **Skip Connections:** Crucial for combining high-level semantic features from the encoder with low-level fine-grained features from the decoder, leading to precise segmentation. (Illustrate with a U-Net diagram).
    * Loss Functions for Segmentation (briefly):
        * Pixel-wise Cross-Entropy.
        * Dice Loss, Jaccard (IoU) Loss (better for imbalanced classes, common in medical segmentation).
    * **Reference Card:** U-Net Architecture Diagram (conceptual blocks).
    * **Minimal Example:** Conceptual U-Net structure.

        ```python
        # Conceptual U-Net structure
        # inputs = Input(...)
        # # Encoder
        # c1 = Conv2D(...)(inputs) ... p1 = MaxPooling2D(...)(c1)
        # ...
        # # Bottleneck
        # bn = Conv2D(...)(...)
        # # Decoder
        # u1 = Conv2DTranspose(...)(bn) ... u1 = concatenate([u1, c_corresponding]) ... u1 = Conv2D(...)(u1)
        # ...
        # outputs = Conv2D(num_classes, 1, activation='softmax')(...)
        # model = Model(inputs=[inputs], outputs=[outputs])
        ```

7. Demo 3: Pre-trained Object Detection or Segmentation (10 min)

    * **File:** `lectures/08/demo/03_pretrained_detection_or_segmentation.md`
    * **Content:** Offers options for using pre-trained models for either object detection (e.g., YOLO) or image segmentation (e.g., DeepLabV3 from TF Hub), focusing on inference and visualization.

8. Advanced Topics & Future Directions (10 min)

    * **Briefly touch upon:**
        * **Video Data:** Images over time, introducing temporal dimension. Challenges: motion, appearance changes.
        * **Vision Transformers (ViTs):** Applying Transformer architectures (from NLP) to images. How they work (patches as tokens, self-attention). Potential advantages in medical imaging.
        * **Generative Models (GANs, Diffusion Models):**
            * Synthetic medical data generation (for data augmentation, privacy).
            * Image enhancement, super-resolution, modality translation.
        * **Explainable AI (XAI) for Medical CV:**
            * Importance of model interpretability in clinical settings (trust, debugging, discovery).
            * Methods like Grad-CAM, LIME, SHAP to visualize what the model "looks at".
        * **Self-Supervised Learning:** Learning representations from unlabeled data, reducing reliance on large annotated datasets.
    * **Data Augmentation:**
        * Techniques: Rotation, scaling, flipping, brightness/contrast changes, elastic deformations.
        * Libraries: Albumentations (powerful for image augmentation).
    * **Key Libraries & Toolkits for Medical CV:**
        * **MONAI:** PyTorch-based framework for deep learning in healthcare imaging.
        * **SimpleITK:** Powerful library for medical image analysis, including registration and segmentation.

9. Mini-Demo: Introduction to Video Object Tracking (5 min)

    * **File:** `lectures/08/demo/04_video_object_tracking_mini_demo.md`
    * **Content:** Introduces video object tracking concepts with brief visual demonstrations of detection-based (NN) and optical flow-based approaches.

10. References & Further Learning

    * **Key Review Papers / Comprehensive Textbooks:** (Self-search encouraged for latest comprehensive resources and foundational texts in medical computer vision).
    * **Datasets:** TCIA (The Cancer Imaging Archive), Grand Challenge, MedMNIST, Kaggle datasets.
    * **Annotation Tools:** ITK-SNAP, 3D Slicer, LabelMe, CVAT.
    * **Conferences:** MICCAI (Medical Image Computing and Computer Assisted Intervention), CVPR (Conference on Computer Vision and Pattern Recognition), RSNA (Radiological Society of North America).
    * **Online Courses & Communities:** Platforms like Coursera, edX, Fast.ai; communities like Kaggle, PapersWithCode.

----

#FIXME: Lecture content here