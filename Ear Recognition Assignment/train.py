import os
import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from dataloader import create_dataloaders
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Load DataLoaders
base_path = os.path.join('datasets', 'ears', 'images-cropped')
train_dir = os.path.join(base_path, 'train')
val_dir = os.path.join(base_path, 'val')
test_dir = os.path.join(base_path, 'val')
train_loader, val_loader, test_loader, num_classes = create_dataloaders(train_dir, val_dir, test_dir)

# Model
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Move the model to the device (GPU or CPU)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001, betas=(0.8, 0.999), eps=1e-08)
# optimizer = optim.Adam(model.parameters())

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# Training
best_val_accuracy = 0.0
val_loss = 0
model.train()
for epoch in range(100):  # number of epochs
    # Training loop
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    scheduler.step(val_loss)
    
    # Validation loop
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            # Inside the training loop
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    print(f'Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%. ', end="")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        # Save the model
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"New best model saved.")
    else:
        print("")
