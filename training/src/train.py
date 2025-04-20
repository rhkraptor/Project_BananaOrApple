# Dummy Model for testing purposes
# This script creates a dummy model and saves it to the specified path.

import torch
import torch.nn as nn

class DummyNet(nn.Module):
    def __init__(self):
        super(DummyNet, self).__init__()
        self.fc = nn.Linear(3 * 128 * 128, 2)  # Fake classifier for 2 classes

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

# Create a dummy model
model = DummyNet()

# Save it to hf_app so Hugging Face can load it
torch.save(model, "../hf_app/banana_or_apple.pt")

print("Dummy model saved successfully.")