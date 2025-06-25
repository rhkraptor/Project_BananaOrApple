# Banana or Apple or Other – Image Classifier (ResNet18 + Gradio)

This deep learning project classifies uploaded fruit images into three categories: **apple**, **banana**, or **other**.  
It utilizes a fine-tuned **ResNet18** model implemented with **PyTorch**, featuring real-time image classification through a **Gradio** interface. The project includes a complete training pipeline and evaluation toolkit.

---

## Live Demo

Try the classifier on Hugging Face Spaces:  
[https://huggingface.co/spaces/rhkraptor/BananaOrApple](https://huggingface.co/spaces/rhkraptor/BananaOrApple)

---

## Dataset

The dataset used for training and validation is available here:  
[Google Drive – Banana/Apple/Other Dataset](https://drive.google.com/drive/folders/1xII1yoYWo1aEtlMhOMg6zVclPXfi22Rb)

---

## Model Overview

- **Architecture**: ResNet18 backbone with a custom classification head
- **Framework**: PyTorch
- **Validation Accuracy**: ~94%
- **Techniques Used**:
  - Data augmentation
  - Early stopping
  - Class balancing
- **Classes**:
  - `apple`
  - `banana`
  - `other`
- **Deployment**:
  - Gradio app
  - Hugging Face Spaces
  - GitHub Actions for CI/CD

---

## Features

- End-to-end image classification pipeline
- Interactive Gradio web interface for real-time predictions
- Optimized model training with robust evaluation strategy
- Easy deployment and reproducibility

---

## Getting Started

1. Clone the repository  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
