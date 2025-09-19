import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets, models
import os

# Dataset path
DATASET_PATH = os.path.join("Datasets", "FERdataset")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # Debug print

# Define transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load dataset with error handling
try:
    train_dataset = datasets.ImageFolder(os.path.join(DATASET_PATH, "train"), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(DATASET_PATH, "test"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}")  # Debug print
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Define the model
def create_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 5)  # Modify for 5 classes
    return model  # DO NOT move model to GPU here

# Train the model
def train_model(model, train_loader, epochs=6, lr=0.001):
    model.to(device)  # Move model to GPU **inside** training function
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        correct, total = 0, 0

        print(f"Starting Epoch {epoch+1}/{epochs}...")  # Debug print

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    # Save the trained model
    os.makedirs("Models", exist_ok=True)
    torch.save(model.state_dict(), "Models/emotion_model.pth")
    print("Model saved successfully!")

if __name__ == "__main__":
    model = create_model()
    train_model(model, train_loader, epochs=6)
