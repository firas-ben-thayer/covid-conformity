import cv2
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import contextmanager
import asyncio
import base64
from typing import Dict, Set, List
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(
    title="Live Feed API",
    description="API for streaming live video feed from server-connected webcams.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global resources
camera_instances = {}
thread_pool = ThreadPoolExecutor(max_workers=4)

# Add available cameras - this is the list of camera IDs we support
AVAILABLE_CAMERAS = [0, 2]  # Camera 0 and 2

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[int, Set[WebSocket]] = {0: set(), 2: set()}
        self.camera_tasks: Dict[int, asyncio.Task] = {}
        
    async def connect(self, websocket: WebSocket, camera_id: int):
        await websocket.accept()
        self.active_connections[camera_id].add(websocket)
        
        # Start camera stream if it's the first connection
        if camera_id not in self.camera_tasks:
            self.camera_tasks[camera_id] = asyncio.create_task(stream_camera(camera_id))

    async def disconnect(self, websocket: WebSocket, camera_id: int):
        self.active_connections[camera_id].remove(websocket)
        
        # Stop camera stream if no more connections
        if not self.active_connections[camera_id] and camera_id in self.camera_tasks:
            self.camera_tasks[camera_id].cancel()
            self.camera_tasks.pop(camera_id)

    async def broadcast_frame(self, camera_id: int, frame: str):
        if not self.active_connections[camera_id]:
            return
            
        dead_connections = set()
        for connection in self.active_connections[camera_id]:
            try:
                await connection.send_text(frame)
            except:
                dead_connections.add(connection)
        
        for dead in dead_connections:
            await self.disconnect(dead, camera_id)

manager = ConnectionManager()

def process_frame(frame):
    """Process frame in a separate thread"""
    # Resize frame to reduce data
    frame = cv2.resize(frame, (640, 480))
    
    # Compress frame
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    frame_data = base64.b64encode(buffer).decode('utf-8')
    
    return frame_data

# New endpoint to list available cameras
@app.get("/cameras")
async def get_cameras():
    """Get a list of all available cameras and currently active ones"""
    # Check which cameras can be opened
    active_cameras = []
    
    for camera_id in AVAILABLE_CAMERAS:
        try:
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                active_cameras.append(camera_id)
                cap.release()
        except Exception as e:
            print(f"Error checking camera {camera_id}: {str(e)}")
    
    return {
        "available": AVAILABLE_CAMERAS,
        "active": active_cameras
    }

@contextmanager
def get_camera(camera_index: int):
    """Context manager to handle camera resources safely"""
    try:
        if camera_index in camera_instances:
            old_cap = camera_instances[camera_index]
            old_cap.release()
            camera_instances.pop(camera_index, None)
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Cannot open camera with index {camera_index}")
            yield None
        else:
            # Set lower resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera_instances[camera_index] = cap
            try:
                yield cap
            finally:
                if camera_index in camera_instances:
                    cap.release()
                    camera_instances.pop(camera_index, None)
    except Exception as e:
        print(f"Error with camera {camera_index}: {str(e)}")
        yield None

async def stream_camera(camera_id: int):
    """Stream camera feed using WebSocket"""
    with get_camera(camera_id) as cap:
        if cap is None:
            return

        last_frame_time = 0
        frame_interval = 1/30  # 30 FPS
        
        try:
            while True:
                current_time = asyncio.get_event_loop().time()
                if current_time - last_frame_time < frame_interval:
                    await asyncio.sleep(0.001)  # Small sleep to prevent CPU overload
                    continue

                success, frame = cap.read()
                if not success:
                    break

                # Process frame in thread pool
                frame_data = await asyncio.get_event_loop().run_in_executor(
                    thread_pool, 
                    process_frame,
                    frame
                )
                
                message = json.dumps({
                    "type": "frame",
                    "data": frame_data
                })
                
                await manager.broadcast_frame(camera_id, message)
                last_frame_time = current_time
                
        except asyncio.CancelledError:
            print(f"Stream cancelled for camera {camera_id}")
        except Exception as e:
            print(f"Error streaming camera {camera_id}: {str(e)}")

@app.websocket("/ws/{camera_id}")
async def websocket_endpoint(websocket: WebSocket, camera_id: int):
    if camera_id not in [0, 2]:
        await websocket.close(code=4000)
        return
    
    await manager.connect(websocket, camera_id)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except:
        await manager.disconnect(websocket, camera_id)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup all resources when the server shuts down"""
    for cap in camera_instances.values():
        cap.release()
    camera_instances.clear()
    thread_pool.shutdown()

if __name__ == "__main__":
    uvicorn.run("live_feed_api:app", host="0.0.0.0", port=8001, reload=True)