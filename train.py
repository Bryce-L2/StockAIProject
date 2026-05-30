import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reads subfolders, resizes, and converts pixels to numbers
dataset = datasets.ImageFolder(
    "dataset",
    transform=transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
)
print("Class labels:", dataset.class_to_idx)

# Feeds the images into the model in batches of 32
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# It learns to recognize visual patterns in candlestick charts
class StockClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Scans the image for basic shapes like lines and edges
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            # Looks for more complex patterns built from what layer 1 found
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            # Looks for the most complex patterns like trends and structures
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            # Turns the 2D image into a single list of numbers for the final layers
            nn.Flatten(),
            # Processes all the features found by the conv layers
            nn.Linear(64 * 16 * 16, 128), nn.ReLU(),
            # Squishes everything to a number between 0(bearish) and 1(bullish)
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = StockClassifier().to(DEVICE)

#  Measures how wrong the model's prediction was compared to the real label
loss_fn = nn.BCELoss()

# Adam optimizer adjusts the model after each batch to reduce the loss
optimizer = optim.Adam(model.parameters(), lr=0.0001)

print("Training...")
for epoch in range(EPOCHS):
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
        optimizer.zero_grad()
        # Runs the images through the model to get predictions
        output = model(images)
        # Calculates how wrong the predictions were
        loss = loss_fn(output, labels)
        # Works backwards through the model to find what caused the error
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Prints the total loss for this epoch
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# Switchs model to evaluation mode so it stops updating weights during the accuracy check
model.eval()
correct = 0
total = 0

# Tells PyTorch not to track calculations since we are just testing not training
with torch.no_grad():
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
        output = model(images)

        # If output is above 0.5 predict bullish, below 0.5 predict bearish
        predicted = (output > 0.5).float()

        # Count how many predictions were correct
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Training Accuracy: {100 * correct / total:.2f}%")

# Saves everything the model learned to model.pt so predict.py can load it later
torch.save(model.state_dict(), "model.pt")
print("Model saved!")
