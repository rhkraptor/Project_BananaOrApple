import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import BananaOrAppleClassifier

# ---- Hyperparameters ----
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Data Transformations ----
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ---- Load Dataset ----
base_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset')

train_data = datasets.ImageFolder(os.path.join(base_dir, 'train'), transform=transform)
val_data   = datasets.ImageFolder(os.path.join(base_dir, 'val'), transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=BATCH_SIZE)

# ---- Model, Loss, Optimizer ----
model = BananaOrAppleClassifier().to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ---- Training Loop ----
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct = 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        preds = model(imgs)
        loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (preds.argmax(1) == labels).sum().item()

    acc = correct / len(train_loader.dataset)

    # ---- Validation ----
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs)
            val_correct += (preds.argmax(1) == labels).sum().item()
    val_acc = val_correct / len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {total_loss:.3f} | Train Acc: {acc:.2%} | Val Acc: {val_acc:.2%}")

# ---- Save Model ----
model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'hf_app', 'banana_or_apple.pt')
torch.save(model.state_dict(), model_path)
print(f"\n✅ Model saved to: {model_path}")
# ---- End of Training ----