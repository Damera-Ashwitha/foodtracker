from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from PIL import Image
import torch
from torchvision import models, transforms
import pandas as pd
import io
import os
from datetime import datetime
from pymongo import MongoClient

# ------------------- Init ---------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ------------------- Base Path ---------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("Running in directory:", BASE_DIR)

# ------------------- MongoDB ---------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["calorie_db"]
collection = db["food_logs"]

# ------------------- Model & Data ---------------------
# Load class-to-idx
class_idx_path = os.path.join(BASE_DIR, "class_to_idx.pth")
class_to_idx = torch.load(class_idx_path)
idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = len(class_to_idx)

# Load model
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

model_path = os.path.join(BASE_DIR, "indian_food_classifier.pth")
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load Excel calorie file
excel_path = os.path.join(BASE_DIR, "indian_food_calories.csv.xlsx")
df = pd.read_excel(excel_path)
df.columns = df.columns.str.strip()
df['FoodItem'] = df['FoodItem'].str.lower()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------- Routes ---------------------

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


from datetime import datetime

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        print(f"[INFO] Received file: {file.filename}")

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            print(f"[INFO] Model output: {outputs}")
            _, predicted = torch.max(outputs, 1)
            predicted_idx = predicted.item()
            label = idx_to_class[predicted_idx]
            print(f"[INFO] Predicted label: {label}")

        row = df[df['FoodItem'] == label.lower()]
        if row.empty:
            print(f"[WARN] No calorie entry found for: {label}")
            calories = "Unknown"
        else:
            calories = int(row.iloc[0]['Calories'])  # ✅ Fix: convert to int

        # Insert into MongoDB (only if calories is not unknown)
        entry = {
            "food": label,
            "timestamp": datetime.now()
        }
        if calories != "Unknown":
            entry["calories"] = calories

        collection.insert_one(entry)
        print(f"[INFO] Inserted: {entry}")

        return JSONResponse({
            "food": label,
            "calories": calories
        })

    except Exception as e:
        print("[ERROR] Prediction failed:")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": "Prediction failed"})
