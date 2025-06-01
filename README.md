# 🍌🍎 Banana or Apple or Other – Image Classifier (ResNet18 + Gradio)

This deep learning project classifies uploaded fruit images into three categories: **apple**, **banana**, or **other**.  
It uses a fine-tuned **ResNet18** model built with **PyTorch**, features real-time classification via **Gradio**, and includes a training pipeline and evaluation toolkit.

---

## 🚀 Live Demo
🔗 Try the classifier on **Hugging Face Spaces**:  
👉 [https://huggingface.co/spaces/your_username/BananaOrApple](#) *(replace with real URL once deployed)*

---

## 🧠 Model Overview

- ✅ **Architecture**: ResNet18 backbone + custom classification head
- ✅ **Training Framework**: PyTorch
- ✅ **Accuracy**:  
  - Validation accuracy: **~97%**  
  - Trained with **data augmentation**, **early stopping**, and **class balancing**
- ✅ **Classes**: `apple`, `banana`, `other`
- ✅ **Deployment**: Gradio app with Hugging Face Spaces & GitHub Actions

---

## 🗂️ Project Structure

BananaOrApple_Project/
│
├── hf_app/ # Gradio web app
│ ├── app.py # Main Gradio interface
│ ├── model.py # Inference model (ResNet18)
│ ├── banana_or_apple.pt # Trained model weights
│ └── requirements.txt # HF Spaces dependencies
│ 
│
├── training/ # Training pipeline
│ └── src/
│ ├── model.py # ResNet18 classifier class
│ ├── train.py # Training script with early stopping, timing, overfit checks
│ └── data.py # Data loader with augmentations
│
├── dataset/ # Training & validation data (not included in repo)
│
├── exploration.ipynb # Optional notebook for analysis & testing
├── .gitignore
└── README.md