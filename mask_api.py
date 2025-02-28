import io
import os
import base64
import torch
import torch.nn as nn
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision.models import resnet50, ResNet50_Weights
from cachetools import TTLCache, LRUCache
import hashlib
import math
from typing import List, Dict, Tuple
from dataclasses import dataclass
from contextlib import asynccontextmanager
import uvicorn
import logging

class MaskDetector(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT
        self.resnet = resnet50(weights=weights)
        num_features = self.resnet.fc.in_features
        # Replace final fully connected layer with our custom head.
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        return self.resnet(x)

logging.basicConfig(
    level=logging.DEBUG,  # Detailed logging enabled
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

BATCH_SIZE = 8  # Not used anymore.
IMAGE_SIZE = 224
CACHE_TTL = 5.0
CACHE_SIZE = 1000
MODEL_PATH = "mask_detector_model.pth"
AVERAGE_FACE_HEIGHT_MM = 57.33  # Average of provided face heights
SAFETY_DISTANCE_MM = 1800  # 1.8 meters in millimeters

@dataclass
class ModelResources:
    device: torch.device
    mtcnn: MTCNN
    mask_detector: nn.Module
    transform: torch.nn.Module
    thread_pool: ThreadPoolExecutor

resources = None

def compute_image_hash(image_data: bytes) -> str:
    return hashlib.md5(image_data).hexdigest()

def calculate_distances(faces: List[Dict]) -> List[Dict]:
    """Calculate distances between all faces and determine if they're safe"""
    if len(faces) < 2:
        return []
    
    distances = []
    for i in range(len(faces)):
        for j in range(i+1, len(faces)):
            face1 = faces[i]
            face2 = faces[j]
            
            # Calculate centers of face boxes
            x1_center = (face1["box"][0] + face1["box"][2]) / 2
            y1_center = (face1["box"][1] + face1["box"][3]) / 2
            x2_center = (face2["box"][0] + face2["box"][2]) / 2
            y2_center = (face2["box"][1] + face2["box"][3]) / 2
            
            # Calculate pixel distance between faces
            pixel_distance = math.sqrt((x2_center - x1_center) ** 2 + (y2_center - y1_center) ** 2)
            
            # Estimate real-world scale using face heights
            face1_height_px = face1["box"][3] - face1["box"][1]
            face2_height_px = face2["box"][3] - face2["box"][1]
            avg_face_height_px = (face1_height_px + face2_height_px) / 2
            
            # Calculate mm per pixel
            mm_per_pixel = AVERAGE_FACE_HEIGHT_MM / avg_face_height_px
            
            # Calculate distance in mm
            distance_mm = pixel_distance * mm_per_pixel
            
            # Determine if distance is safe
            is_safe = distance_mm >= SAFETY_DISTANCE_MM
            
            distances.append({
                "face1_index": i,
                "face2_index": j,
                "distance_mm": distance_mm,
                "distance_meters": distance_mm / 1000,
                "is_safe": is_safe,
                "points": [
                    int(x1_center), 
                    int(y1_center), 
                    int(x2_center), 
                    int(y2_center)
                ]
            })
    
    return distances

@asynccontextmanager
async def lifespan(app: FastAPI):
    global resources
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thread_pool = ThreadPoolExecutor(max_workers=4)
    logger.debug(f"Using device: {device}")

    mtcnn = MTCNN(
        keep_all=True,
        device=device,
        min_face_size=60,
        thresholds=[0.6, 0.7, 0.7],
        post_process=True
    )

    mask_detector = MaskDetector().to(device)
    mask_detector.eval()
    logger.debug("Mask detector model architecture:")
    logger.debug(str(mask_detector))
    
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=device)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            mask_detector.load_state_dict(state_dict)
            logger.info("Loaded mask detection model successfully!")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    else:
        raise FileNotFoundError(f"Mask detector model not found at {MODEL_PATH}")

    weights = ResNet50_Weights.DEFAULT
    transform = weights.transforms()
    
    resources = ModelResources(
        device=device,
        mtcnn=mtcnn,
        mask_detector=mask_detector,
        transform=transform,
        thread_pool=thread_pool
    )
    
    try:
        yield
    finally:
        resources.thread_pool.shutdown()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

prediction_cache = TTLCache(maxsize=CACHE_SIZE, ttl=CACHE_TTL)
face_detection_cache = LRUCache(maxsize=CACHE_SIZE)

async def process_frame(image: Image.Image, ws: WebSocket):
    """
    Process a single image frame: run face detection and mask recognition immediately,
    then send the result over the WebSocket.
    """
    # Optionally compute hash to log if frames are unique.
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    image_bytes = buf.getvalue()
    image_hash = compute_image_hash(image_bytes)
    logger.debug(f"Processing frame hash: {image_hash}")
    
    # Run face detection.
    boxes, probs = resources.mtcnn.detect(image)
    logger.debug(f"Detection output (boxes): {boxes}, (probs): {probs}")
    if boxes is None:
        result = {
            "status": "success",
            "faces": [],
            "total_faces": 0,
            "distances": [],
            "message": "No faces detected"
        }
    else:
        faces_results = []
        face_tensors = []
        for box, prob in zip(boxes, probs):
            x1, y1, x2, y2 = map(int, box)
            face_img = image.crop((x1, y1, x2, y2))
            face_tensor = resources.transform(face_img)
            face_tensors.append(face_tensor)
        if face_tensors:
            face_batch = torch.stack(face_tensors).to(resources.device)
            with torch.no_grad():
                outputs = resources.mask_detector(face_batch)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.max(outputs, 1)[1]
            for i, (box, prob, pred, mask_prob) in enumerate(zip(boxes, probs, predictions, probabilities)):
                faces_results.append({
                    "box": [int(coord) for coord in box],
                    "mask_status": "mask" if pred.item() == 0 else "no_mask",
                    "mask_confidence": float(mask_prob[0]),
                    "face_detection_confidence": float(prob)
                })
                
        # Calculate distances between faces
        distances = calculate_distances(faces_results)
        
        result = {
            "status": "success",
            "faces": faces_results,
            "total_faces": len(faces_results),
            "distances": distances
        }
    logger.debug(f"Sending result: {result}")
    await ws.send_json({
        "type": "result",
        "result": result
    })

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    try:
        while True:
            data = await websocket.receive_json()
            logger.debug(f"Received data on WebSocket: {list(data.keys())}")
            if data["type"] == "image":
                img_bytes = base64.b64decode(data["image"])
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                logger.debug(f"Image size after decoding: {img.size}")
                # Process each frame immediately and send result.
                await process_frame(img, websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

@app.post("/face_predict")
async def face_predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    try:
        contents = await file.read()
        image_hash = compute_image_hash(contents)
        if image_hash in prediction_cache:
            return JSONResponse(content=prediction_cache[image_hash])
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        boxes, probs = resources.mtcnn.detect(image)
        if boxes is None:
            result = {
                "status": "success",
                "faces": [],
                "total_faces": 0,
                "distances": [],
                "message": "No faces detected"
            }
        else:
            faces_results = []
            face_tensors = []
            for box, prob in zip(boxes, probs):
                x1, y1, x2, y2 = map(int, box)
                face_img = image.crop((x1, y1, x2, y2))
                face_tensor = resources.transform(face_img)
                face_tensors.append(face_tensor)
            if face_tensors:
                face_batch = torch.stack(face_tensors).to(resources.device)
                with torch.no_grad():
                    outputs = resources.mask_detector(face_batch)
                    probabilities = torch.softmax(outputs, dim=1)
                    predictions = torch.max(outputs, 1)[1]
                for i, (box, prob, pred, mask_prob) in enumerate(zip(boxes, probs, predictions, probabilities)):
                    faces_results.append({
                        "box": [int(coord) for coord in box],
                        "mask_status": "mask" if pred.item() == 0 else "no_mask",
                        "mask_confidence": float(mask_prob[0]),
                        "face_detection_confidence": float(prob)
                    })
            
            # Calculate distances between faces
            distances = calculate_distances(faces_results)
            
            result = {
                "status": "success",
                "faces": faces_results,
                "total_faces": len(faces_results),
                "distances": distances
            }
        prediction_cache[image_hash] = result
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

if __name__ == "__main__":
    uvicorn.run("mask_api:app", host="0.0.0.0", port=8000, reload=True)