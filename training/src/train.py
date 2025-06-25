import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import BananaOrAppleClassifier
from data import get_dataloaders
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# Full trainer with timing, patience, overfitting check, and confusion matrix saving
def main():
    data_dir = "../dataset"  # Adjust if needed

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.3),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Loaders
    train_loader, val_loader, _ = get_dataloaders(data_dir, train_transform, val_transform, batch_size=32, num_workers=0)

    # Model setup
    model = BananaOrAppleClassifier()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss and optimizer
    class_weights = torch.tensor([1.0, 1.0, 0.8]).to(device)   # Adjust weights for class imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

    #  NEW: Add scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)


    best_val_acc = 0.0
    patience = 30
    patience_counter = 0

    all_y_true, all_y_pred = [], []

    for epoch in range(100):
        start_time = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        model.eval()
        val_correct, val_total = 0, 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                y_true += labels.cpu().tolist()
                y_pred += predicted.cpu().tolist()

        val_acc = val_correct / val_total
        end_time = time.time()
        print(f"Epoch {epoch+1} |  {end_time - start_time:.2f}s | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        
        #  NEW: Adjust LR based on validation performance
        scheduler.step(val_acc)


        acc_diff = abs(train_acc - val_acc)
        if train_acc > 0.85 and val_acc < 0.75:
            print(" Potential overfitting: training high, validation low")
        elif train_acc < 0.6 and val_acc < 0.6:
            print(" Likely underfitting: model not learning enough")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "../../hf_app/banana_or_apple.pt")
            print(" Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(" Early stopping")
                break

        # Save labels for confusion matrix later
        all_y_true = y_true
        all_y_pred = y_pred

    # Confusion Matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Apple', 'Banana', 'Other'], yticklabels=['Apple', 'Banana', 'Other'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    # Save in project root
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    save_path = os.path.join(root_dir, "confusion_matrix.png")
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    main()

