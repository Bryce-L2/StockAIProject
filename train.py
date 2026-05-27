import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Settings
IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Load images from bullish/bearish folders
dataset = datasets.ImageFolder(
    "dataset",
    transform=transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
)
print("Class labels:", dataset.class_to_idx)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Step 2: Build the CNN
class StockClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

model = StockClassifier().to(DEVICE)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Step 3: Train the model
print("Training...")
for epoch in range(EPOCHS):
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# Step 4: Check accuracy
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
        output = model(images)
        predicted = (output > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Training Accuracy: {100 * correct / total:.2f}%")

# Step 5: Save the model
torch.save(model.state_dict(), "model.pt")
print("Model saved!")