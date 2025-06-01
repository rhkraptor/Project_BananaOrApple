import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import gradio as gr
import os

# Load model
class BananaOrAppleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.model(x)

# Initialize and load weights
model = BananaOrAppleClassifier()
model_path = os.path.join(os.path.dirname(__file__), "banana_or_apple.pt")
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# Class labels
classes = ['apple', 'banana']

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Prediction function
def predict(image):
    image = Image.fromarray(image).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)  # Shape: [1, 3, 128, 128]
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        label = classes[predicted.item()]
        return f"{label} ({confidence.item() * 100:.2f}%)"

# Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload an image of an apple or banana"),
    outputs=gr.Label(label="Prediction"),
    title="🍌 Banana or 🍎 Apple Classifier",
    description="Upload an image to see whether it's a banana or an apple."
)

if __name__ == "__main__":
    interface.launch()