import cv2
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import contextmanager

app = FastAPI(
    title="Live Feed API",
    description="API for streaming live video feed from server-connected webcams.",
    version="1.0.0"
)

# Add CORS middleware to handle browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global dictionary to track camera instances
camera_instances = {}

@contextmanager
def get_camera(camera_index: int):
    """
    Context manager to handle camera resources safely
    """
    try:
        # Check if camera is already in use
        if camera_index in camera_instances:
            # Release the existing instance
            old_cap = camera_instances[camera_index]
            old_cap.release()
            camera_instances.pop(camera_index, None)  # Safely remove the instance
        
        # Create new camera instance
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Cannot open camera with index {camera_index}")
            yield None
        else:
            camera_instances[camera_index] = cap
            try:
                yield cap
            finally:
                # Cleanup when the generator is closed
                if camera_index in camera_instances:
                    cap.release()
                    camera_instances.pop(camera_index, None)  # Safely remove the instance
    except Exception as e:
        print(f"Error with camera {camera_index}: {str(e)}")
        yield None

def gen_frames(camera_index: int):
    """
    Generate MJPEG frames from the webcam at the specified index.
    """
    with get_camera(camera_index) as cap:
        if cap is None:
            return
        
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break
                    
                ret, buffer = cv2.imencode(".jpg", frame)
                frame_bytes = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
                time.sleep(0.033)  # Approximately 30 FPS
        except GeneratorExit:
            # This will be caught by the context manager's finally block
            pass
        except Exception as e:
            print(f"Error generating frames: {str(e)}")

@app.get("/video0", summary="Live video feed from Camera 0")
async def video_feed0():
    """
    Provides MJPEG stream for webcam at index 0 (/dev/video0)
    """
    return StreamingResponse(
        gen_frames(0),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/video2", summary="Live video feed from Camera 2")
async def video_feed2():
    """
    Provides MJPEG stream for webcam at index 2 (/dev/video2)
    """
    return StreamingResponse(
        gen_frames(2),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup all camera instances when the server shuts down
    """
    for cap in camera_instances.values():
        cap.release()
    camera_instances.clear()

if __name__ == "__main__":
    uvicorn.run("live_feed_api:app", host="0.0.0.0", port=8001, reload=True)