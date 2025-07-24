import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image transforms (must match training transforms)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load class-to-index mapping
class_to_idx = torch.load("class_to_idx.pth")
idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = len(class_to_idx)

# Load trained model (ResNet18)
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("indian_food_classifier.pth", map_location=device))
model = model.to(device)
model.eval()

# Load calorie data from Excel
df = pd.read_excel(r"C:\Users\ashwi\OneDrive\Pictures\Documents\indian_food_calories.csv.xlsx")  # Replace with your actual file name
df['FoodItem'] = df['FoodItem'].str.lower()  # Normalize for matching

# Prediction function
def predict_image(image_path, model, idx_to_class, df):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        label = idx_to_class[predicted.item()]
        label_lower = label.lower()

        # Fetch calorie info from DataFrame
        row = df[df['FoodItem'] == label_lower]
        if not row.empty:
            calories = row.iloc[0]['Calories']
        else:
            calories = "Unknown"

    return label, calories

# Run prediction
image_path = r"C:\Users\ashwi\Downloads\lassi.jpeg" # Update path as needed
label, cal = predict_image(image_path, model, idx_to_class, df)
print(f"🥘 Food: {label} | 🔥 Calories: {cal}")
