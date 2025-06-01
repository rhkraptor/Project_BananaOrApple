import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import BananaOrAppleClassifier

# Load model
model = BananaOrAppleClassifier()
model.load_state_dict(torch.load("banana_or_apple.pt", map_location="cpu"))
model.eval()

# Class labels
class_names = ['apple', 'banana', 'other']
threshold = 0.8  # 80% confidence threshold

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Prediction function
def classify_image(img):
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        label = class_names[pred.item()]
        if conf.item() < threshold:
            label = "❓ unknown"
        return {c: float(probs[0][i]) for i, c in enumerate(class_names)}, label

# Gradio interface
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=3, label="Confidence"),
        gr.Text(label="Predicted Class")
    ],
    title="🍌🍎 Banana or Apple or Other?",
    description="Upload an image of a fruit or something else. The model will predict if it's an apple, banana, or unknown.",
    examples=["banana.jpg", "apple.jpg", "car.jpg"]
)

if __name__ == "__main__":
    demo.launch(share=True)