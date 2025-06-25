import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import tempfile
import os, sys

#  Add path to the training/src directory to import the model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training', 'src')))
from model import BananaOrAppleClassifier

#  Load model
model = BananaOrAppleClassifier()
model.load_state_dict(torch.load("banana_or_apple.pt", map_location="cpu"))
model.eval()

#  Class labels
class_names = ['apple', 'banana', 'other']

#  Image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

#  Prediction function (no threshold rejection anymore)
def classify_image(img):
    # Re-save to avoid PermissionError (especially on Windows)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        img.save(tmp.name)
        img = Image.open(tmp.name).convert("RGB")

    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        label = class_names[pred.item()]
        return {c: float(probs[0][i]) for i, c in enumerate(class_names)}, label

#  Gradio interface
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=3, label="Confidence"),
        gr.Text(label="Predicted Class")
    ],
    title=" Banana or Apple or Other?",
    description="Upload an image of a fruit or something else. The model will predict if it's an apple, banana, or other."
)

#  Launch app
if __name__ == "__main__":
    demo.launch(share=True)