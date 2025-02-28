Collecting workspace information# Face Mask Detection with Safety Distance Monitoring System

This repository contains a computer vision solution that detects face masks and monitors safety distances between people in images, videos, and real-time webcam feeds.

## Project Overview

This project implements a complete computer vision pipeline that:
1. Detects faces in images/videos/webcam feeds
2. Determines if detected faces are wearing masks
3. Calculates the distance between detected faces
4. Visualizes safety compliance through a web interface

The solution uses a PyTorch-based face mask detection model with a ResNet50 backbone and communicates with a frontend interface through WebSockets for real-time processing.

## Features

- **Multiple Input Sources**:
  - Real-time webcam processing
  - Video file processing
  - Static image processing
  
- **Detection Capabilities**:
  - Face detection using MTCNN
  - Mask detection using a custom ResNet50 model
  - Safety distance calculation and monitoring
  
- **Visualization**:
  - Real-time overlay of detection results
  - Bird's eye view for distance visualization
  - Distance measurements between people

## Requirements

- Python 3.8+ 
- PyTorch
- FastAPI
- facenet-pytorch
- OpenCV
- Uvicorn
- PIL (Pillow)
- Other dependencies in the Python files

## Installation

1. Clone this repository:
```bash
git clone https://github.com/firas-ben-thayer/covid-conformity.git
cd covid-conformity
```

2. Install the required packages:
```bash
pip install torch torchvision fastapi uvicorn python-multipart pillow opencv-python facenet-pytorch cachetools
```

3. Make sure you have the pre-trained model file mask_detector_model.pth in the root directory. The link of the model is here: https://ydray.com/get/t/u17407548356501kLwbbfc9b7bd42a6eK

## Running the Application

### Main API Server (Face Mask Detection)

Run the main API server:

```bash
python mask_api.py
```

This starts the FastAPI server on http://localhost:8000 that handles:
- Face detection
- Mask detection
- Safety distance calculations
- WebSocket connections for real-time processing

### Live Feed API (Optional - for multiple camera support)

If you want to use the multiple camera feature:

```bash
python live_feed_api.py
```

This starts the secondary API on http://localhost:8001 that handles multiple camera feeds.

## Usage

Once the server(s) are running, open your browser and navigate to http://localhost:8000 to access the web interface.

### 1. Home Page

The home page (`index.html`) presents three options:
- Webcam: For real-time analysis
- Video: For uploading and analyzing video files
- Image: For uploading and analyzing static images

### 2. Webcam Mode

In webcam mode:
1. Click "Start Camera" to activate your webcam
2. Use the toggle buttons to enable/disable:
   - Face Detection
   - Mask Detection
   - Safety Distance

The system will display:
- Face bounding boxes
- Mask status (mask/no mask)
- Distance measurements between people
- Bird's eye view visualization

### 3. Video Mode

In video mode:
1. Upload a video file (supports MP4, WebM, MOV)
2. The video will appear in the player
3. Use the toggle buttons to enable the detection features
4. Results are shown in real-time as the video plays

### 4. Image Mode

In image mode:
1. Upload an image file (supports JPG, PNG)
2. Use the toggle buttons to analyze the image
3. View detection results overlaid on the image

## Technical Implementation

### Backend

- mask_api.py: Main FastAPI server that handles face detection, mask detection and safety distance calculation
  - Uses MTCNN for face detection
  - Uses a custom ResNet50 model for mask detection
  - Calculates distances between people based on face sizes
  - Communicates via WebSocket for real-time processing

- live_feed_api.py: Secondary FastAPI server for handling multiple camera feeds (optional)

### Frontend

- HTML/CSS/JavaScript implementation
- WebSocket communication for real-time updates
- Canvas overlay for visualization
- Responsive design for various screen sizes

### Model

The system uses a pre-trained face mask detection model (`mask_detector_model.pth`) based on ResNet50 architecture, fine-tuned for mask detection with the following architecture:
- ResNet50 backbone
- Custom head with:
  - Linear(2048, 512) → ReLU → Dropout(0.5)
  - Linear(512, 256) → ReLU → Dropout(0.5)
  - Linear(256, 2) for binary classification

## Performance Considerations

- The system uses a caching mechanism to improve performance
- For real-time video, there's a frame rate limiter (30 FPS)
- The image size can be adjusted in the code for performance optimization
- WebSocket communication minimizes latency compared to traditional HTTP requests

## Limitations and Improvements

- **Performance**: The detection speed depends on the hardware capabilities, especially the GPU availability
- **Multiple People**: Performance might degrade with many faces in frame
- **Distance Calculation**: Uses an approximation method based on face size
- **Scaling**: For large-scale deployment, consider using a load balancer and multiple instances

## Running in a Cloud Environment

This solution can be deployed to cloud platforms like AWS, Azure, or GCP:

1. Create a virtual machine with GPU support for better performance
2. Install the dependencies as shown above
3. Configure the firewall to allow traffic on ports 8000 and 8001
4. Run the application using a production ASGI server:

```bash
uvicorn mask_api:app --host 0.0.0.0 --port 8000
uvicorn live_feed_api:app --host 0.0.0.0 --port 8001
```

For production deployment, consider using tools like Docker and Kubernetes for containerization and orchestration.

## Acknowledgements

This project was created as a solution for the DeepNeuronic ML Test, inspired by the CovidSight system. The face detection is powered by MTCNN from facenet-pytorch, and the mask detection model is a custom implementation using PyTorch.
