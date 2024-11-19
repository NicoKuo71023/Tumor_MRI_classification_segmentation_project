# **Tumor MRI Classification and Segmentation Project Proposal**

---

## **1. Project Overview**
This project aims to develop an end-to-end deep learning pipeline for tumor MRI analysis, including classification and segmentation tasks, to assist medical professionals in diagnosis. The project is divided into two primary tasks:
- **Classification**: Predict tumor types from MRI scans (e.g., benign vs. malignant or specific tumor types).
- **Segmentation**: Accurately delineate tumor regions in MRI images.

---

## **2. Methodology**

### **2.1 Dataset**
- **Sources**:
  - Classificaiton: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection
  - Segmentation: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
- **Data Preprocessing**:
  - **Normalization**: Rescale pixel intensity to `[0, 1]`.
  - **Resizing**:
    - Classification: `128 × 128`.
    - Segmentation: `256 × 256`.
  - **Data Augmentation**:
    - Geometric transformations: rotation, flipping, cropping.
    - Intensity adjustments: contrast enhancement, Gaussian noise. (not sure)
  - **Splitting**: Divide data into training, validation, and test sets (e.g., `70% / 15% / 15%`).

---

### **2.2 Classification**

#### **2.2.1 Baseline Model**
- **Simple CNN**:
  - A small convolutional network to establish baseline performance.
  - Architecture: `Conv2D → ReLU → MaxPooling → Fully Connected → Sigmoid/Softmax`.
  - **Metrics**:
    - Accuracy, precision, recall, and F1 score.

#### **2.2.2 Advanced Architectures**
- **ResNet**:
  - Deep architectures (e.g., `ResNet-18`, `ResNet-50`) to address vanishing gradient issues.
  - Use pretrained weights for transfer learning?
- **Vision Transformers (ViT)**:
  - Capture global dependencies in MRI images.
  - Experiment with fine-tuning pretrained ViT models on medical datasets.

#### **2.2.3 Other Potential Models**
- **DenseNet**:
  - Efficient feature reuse for better performance.
- **EfficientNet**:
  - Balances accuracy and model size effectively.

---

### **2.3 Segmentation**

#### **2.3.1 Baseline Model**
- **U-Net**:
  - A standard architecture for medical image segmentation.
  - Combines an encoder (downsampling) and decoder (upsampling) with skip connections.

#### **2.3.2 Advanced Architectures**
- **Attention U-Net**:
  - Incorporates attention mechanisms to focus on tumor regions.
- **DeepLabV3+**:
  - Uses atrous spatial pyramid pooling (ASPP) for multi-scale feature extraction.
- **Swin Transformer**:
  - Combines the power of transformers with efficient hierarchical design.

#### **2.3.3 Other Potential Models**
- **SegNet**:
  - Encoder-decoder-based architecture for segmentation.
- **nnU-Net**:
  - A self-adaptive U-Net tailored for biomedical datasets.

---

### **2.4 Training Strategy**
- **Loss Functions**:
  - **Classification**:
    - Binary Cross-Entropy (BCE) for binary classification.
    - Cross-Entropy Loss for multi-class classification.
  - **Segmentation**:
    - Dice Loss for overlapping region precision.
    - Binary Cross-Entropy + Dice Loss for balanced optimization.
- **Optimizers and Learning Rate Scheduling**:
  - Use Adam or SGD with momentum.
  - Dynamic learning rate adjustments: ReduceLROnPlateau or Cosine Annealing.
- **Evaluation Metrics**:
  - **Classification**: Accuracy, precision, recall, F1 score, and ROC-AUC.
  - **Segmentation**: Dice coefficient, Intersection over Union (IoU), and Hausdorff Distance.

---

### **2.5 Tools and Frameworks**
- **Programming Language**: Python.
- **Deep Learning Frameworks**: PyTorch
- **Data Augmentation**: Albumentations, MONAI (for medical imaging).
- **Visualization Tools**:
  - Grad-CAM: Explainability for classification models.
  - Overlay segmentation results on MRI images.
