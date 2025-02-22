import io
import os
import base64
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from facenet_pytorch import MTCNN
import torch.nn as nn
import uvicorn
from torchvision.models import resnet50, ResNet50_Weights

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================
# Setup face detection with MTCNN
# ============================
mtcnn = MTCNN(keep_all=True, device=device)

# ============================
# Define the enhanced mask classifier using ResNet50 with new weights API
# ============================
class MaskDetector(nn.Module):
    def __init__(self):
        super(MaskDetector, self).__init__()
        weights = ResNet50_Weights.DEFAULT  # Use the best available weights (IMAGENET1K_V2)
        self.resnet = resnet50(weights=weights)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        
    def forward(self, x):
        return self.resnet(x)

# Initialize mask detector
mask_detector = MaskDetector().to(device)
mask_detector.eval()

# Load trained weights
mask_model_path = "mask_detector_model.pth"
if os.path.exists(mask_model_path):
    state_dict = torch.load(mask_model_path, map_location=device)
    if 'model_state_dict' in state_dict:
        mask_detector.load_state_dict(state_dict['model_state_dict'])
    else:
        mask_detector.load_state_dict(state_dict)
    print("Loaded mask detection model successfully!")
else:
    raise FileNotFoundError(f"Mask detector model not found at {mask_model_path}")

# Define image transforms for the mask detector using the weights transforms
weights = ResNet50_Weights.DEFAULT
transform = weights.transforms()

@app.get("/")
def home():
    return {
        "message": "Face Detection and Mask Classification API",
        "status": "running",
        "models_loaded": {
            "face_detection": "MTCNN",
            "mask_classification": "ResNet50"
        }
    }

@app.post("/face_predict")
async def face_predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    try:
        # Read and process image
        contents = await file.read()
        image_pil = Image.open(io.BytesIO(contents)).convert("RGB")

        # Detect faces using MTCNN
        boxes, probs = mtcnn.detect(image_pil)
        faces_results = []
        cropped_faces = []  # to store cropped face images for the collage

        if boxes is not None:
            for box, prob in zip(boxes, probs):
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image_pil.width, x2), min(image_pil.height, y2)
                
                # Extract and process face image
                face_img = image_pil.crop((x1, y1, x2, y2))
                cropped_faces.append(face_img)
                input_tensor = transform(face_img).unsqueeze(0).to(device)

                # Get mask prediction
                with torch.no_grad():
                    outputs = mask_detector(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)
                    mask_prob = float(probabilities[0][0])  # Probability of wearing mask
                    label = "mask" if predicted.item() == 0 else "no_mask"

                faces_results.append({
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "mask_status": label,
                    "mask_confidence": float(mask_prob),
                    "face_detection_confidence": float(prob)
                })

            # Create a horizontal collage of the cropped faces
            if cropped_faces:
                widths, heights = zip(*(face.size for face in cropped_faces))
                total_width = sum(widths)
                max_height = max(heights)
                collage = Image.new("RGB", (total_width, max_height))
                x_offset = 0
                for face in cropped_faces:
                    collage.paste(face, (x_offset, 0))
                    x_offset += face.width

                # Encode collage image as base64 string so it can be returned in JSON
                buffered = io.BytesIO()
                collage.save(buffered, format="JPEG")
                collage_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            else:
                collage_str = None

            return JSONResponse(content={
                "status": "success",
                "faces": faces_results,
                "total_faces": len(faces_results),
                "collage": collage_str
            })
        else:
            return JSONResponse(content={
                "status": "success",
                "faces": [],
                "total_faces": 0,
                "message": "No faces detected in the image"
            })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("mask_api:app", host="0.0.0.0", port=8000, reload=True)