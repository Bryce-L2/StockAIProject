import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

IMAGE_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# The structure must be identical or the saved weights won't load correctly
class StockClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # First conv layer
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # Shrink image by half

            # Second conv layer
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # Shrink image by half again

            # Third conv layer 
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # Shrink image by half again

            # Flattens and make final decision
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

# Loads the saved model weights from model.pt
model = StockClassifier()
model.load_state_dict(torch.load("model.pt", map_location=DEVICE))
model.eval()

# Every image must be processed the same way the training images were
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# Asks user for image path
print("=== ES/NQ Futures Daily Bias Predictor ===")
image_path = input("Enter chart image path: ")

try:
    # Opens the image and convert to RGB
    image = Image.open(image_path).convert("RGB")

    # Transforms the image and add a batch dimension
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    #Runs the image through the model
    with torch.no_grad():  
        output = model(image_tensor)
        confidence = output.item()

    # Converts output to a readable result
    if confidence > 0.5:
        label = "BULLISH"
        score = confidence * 100
    else:
        label = "BEARISH"
        score = (1 - confidence) * 100

    # Prints the final prediction
    print(f"\nResult:     {label}")
    print(f"Confidence: {score:.2f}%")

except FileNotFoundError:
    print("Error: Image file not found. Check the path and try again.")
except Exception as e:
    print(f"Error: {e}")
