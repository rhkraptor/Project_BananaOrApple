import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from model import BananaOrAppleClassifier
from data import get_dataloaders

def main():
    # Parameters
    EPOCHS = 100
    BATCH_SIZE = 32
    PATIENCE = 15
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
    train_loader, val_loader, _ = get_dataloaders(data_dir, batch_size=BATCH_SIZE)

    # Initialize model, loss, optimizer
    model = BananaOrAppleClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0
    epochs_no_improve = 0

    print()
    for epoch in range(EPOCHS):
        start_time = time.time()

        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        # Fit diagnosis
        if train_acc < 0.7 and val_acc < 0.7:
            status = "⚠️ Underfitting"
        elif train_acc > 0.85 and (train_acc - val_acc) > 0.15:
            status = "⚠️ Overfitting"
        elif val_acc > 0.8:
            status = "✅ Good fit"
        else:
            status = "🟡 Needs tuning"

        # Time per epoch
        epoch_time = time.time() - start_time

        # Print summary
        print(f"Epoch {epoch+1}/{EPOCHS} | ⏱ {epoch_time:.1f}s | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | {status}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "..", "..", "hf_app", "banana_or_apple.pt"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
                break

    print(f"✅ Final model saved to: hf_app/banana_or_apple.pt")

if __name__ == "__main__":
    main()