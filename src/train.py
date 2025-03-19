import sys
import os
import torch
import torch.optim as optim
import torch.nn as nn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.ViT_model import CustomViT  # Import the ViT model class
from main import train_loader, test_loader  # Load DataLoaders

# Initialize model and move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomViT(num_labels=2).to(device)

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-5)

# Training Loop
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)  # Already in (3, 224, 224)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Print progress update every 10 batches
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    # Print average loss for the epoch
    print(f"Epoch [{epoch+1}/{num_epochs}] completed. Avg Loss: {running_loss / len(train_loader):.4f}")

# Save Model
torch.save(model.state_dict(), "models/vit_model.pth")
print("Model training complete. Saved as vit_model.pth")
