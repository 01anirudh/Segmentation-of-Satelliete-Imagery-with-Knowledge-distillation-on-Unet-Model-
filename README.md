# CO2 Segmentation using Knowledge Distillation

A deep learning project implementing knowledge distillation for CO2 segmentation using a teacher-student model architecture. The teacher model employs UNet++ with EfficientNet encoder and attention mechanisms, while the student model uses a lightweight UNet with MobileNet encoder.

## 🎯 Project Overview

This project demonstrates knowledge distillation techniques to transfer knowledge from a complex, high-performance teacher model to a lightweight, efficient student model suitable for deployment in resource-constrained environments.

### Key Features

- **Advanced Teacher Model**: UNet++ architecture with EfficientNet encoder
- **Attention Mechanisms**: Self-attention and cross-attention modules for enhanced feature learning
- **Efficient Student Model**: Lightweight UNet with MobileNet encoder
- **Knowledge Distillation**: Transfer learning from teacher to student for improved performance
- **CO2 Segmentation**: Specialized for segmenting CO2 regions in images

## 🏗️ Architecture

### Teacher Model
- **Architecture**: UNet++
- **Encoder**: EfficientNet (pre-trained on ImageNet)
- **Special Components**:
  - Self-Attention Modules: Capture long-range dependencies
  - Cross-Attention Modules: Enable feature interaction between encoder and decoder
- **Purpose**: High-accuracy segmentation model serving as knowledge source

### Student Model
- **Architecture**: UNet
- **Encoder**: MobileNet (lightweight and efficient)
- **Design**: Simplified architecture for faster inference
- **Purpose**: Deployable model learning from teacher's knowledge

## 📊 Knowledge Distillation Strategy

The knowledge distillation process involves:

1. **Soft Target Distillation**: Student learns from teacher's soft predictions
2. **Feature Matching**: Intermediate feature alignment between teacher and student
3. **Combined Loss Function**:
   - Distillation Loss (KL Divergence)
   - Feature Matching Loss (L2 Distance)
   - Ground Truth Segmentation Loss

```python
Total_Loss = α × Distillation_Loss + β × Feature_Loss + γ × Segmentation_Loss
