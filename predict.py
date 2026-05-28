import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Settings - must match train.py exactly
IMAGE_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Define the exact same model architecture as train.py
# The structure must be identical or the saved weights won't load correctly
class StockClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # First conv layer - detects basic shapes and edges
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # Shrink image by half

            # Second conv layer - detects more complex patterns
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # Shrink image by half again

            # Third conv layer - detects the most complex patterns
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # Shrink image by half again

            # Flatten and make final decision
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()  # Output between 0 and 1
        )
    def forward(self, x):
        return self.model(x)

# Load the saved model weights from model.pt
model = StockClassifier()
model.load_state_dict(torch.load("model.pt", map_location=DEVICE))
model.eval()  # Put model in evaluation mode - stops it from updating weights

# Step 2: Same image transformations as train.py
# Every image must be processed the same way the training images were
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize to 128x128
    transforms.ToTensor()  # Convert image to numbers
])

# Step 3: Ask user for image path
print("=== ES/NQ Futures Daily Bias Predictor ===")
image_path = input("Enter chart image path: ")

try:
    # Open the image and convert to RGB (removes alpha channel if present)
    image = Image.open(image_path).convert("RGB")

    # Transform the image and add a batch dimension
    # unsqueeze(0) adds an extra dimension because the model expects batches
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Step 4: Run the image through the model
    with torch.no_grad():  # Don't update the model during prediction
        output = model(image_tensor)
        confidence = output.item()  # Extract the raw number from the tensor

    # Step 5: Convert output to a readable result
    # Output close to 1 = bullish, close to 0 = bearish
    if confidence > 0.5:
        label = "BULLISH"
        score = confidence * 100
    else:
        label = "BEARISH"
        score = (1 - confidence) * 100

    # Print the final prediction
    print(f"\nResult:     {label}")
    print(f"Confidence: {score:.2f}%")

except FileNotFoundError:
    # If the image path doesn't exist
    print("Error: Image file not found. Check the path and try again.")
except Exception as e:
    print(f"Error: {e}")