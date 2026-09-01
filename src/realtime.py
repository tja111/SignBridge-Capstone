import cv2
import os
import torch
from torch import load
from model import DETR
import albumentations as A
from utils.boxes import rescale_bboxes
from utils.setup import get_classes, get_colors
from utils.logger import get_logger
from utils.rich_handlers import DetectionHandler, create_detection_live_display
import sys
import time
from pathlib import Path


# Initialize logger and handlers
logger = get_logger("realtime")
detection_handler = DetectionHandler()

logger.print_banner()
logger.realtime("Initializing real-time sign language detection...")

transforms = A.Compose(
        [   
            A.Resize(224,224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.ToTensorV2()
        ]
    )

def resource_path(relative_path: str) -> str:
    base = getattr(sys, "_MEIPASS", None) or Path(__file__).resolve().parent.parent
    return str(Path(base, relative_path))

model = DETR(num_classes=26)
model.eval()
model.load_pretrained(resource_path('checkpoints/99_model.pt'))
CLASSES = get_classes() 
COLORS = get_colors() 

logger.realtime("Starting camera capture...")
# Camera index can be provided as the first CLI argument
cam_index = 0
if len(sys.argv) > 1:
    try:
        cam_index = int(sys.argv[1])
    except Exception:
        pass
try:
    backend = cv2.CAP_DSHOW if os.name == "nt" else None
    if backend is not None:
        cap = cv2.VideoCapture(cam_index, backend)
    else:
        cap = cv2.VideoCapture(cam_index)
except Exception:
    cap = cv2.VideoCapture(cam_index)
try:
    # Resolution and FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # Low latency buffer
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Force MJPG where available
    try:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    except Exception:
        pass
    # Attempt to reduce flicker
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    if hasattr(cv2, "CAP_PROP_AUTOFOCUS"):
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    if hasattr(cv2, "CAP_PROP_AUTO_WB"):
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
except Exception:
    pass

# Window name for OpenCV preview
WINDOW_NAME = 'Frame'

# Initialize performance tracking
frame_count = 0
fps_start_time = time.time()

try:
    while cap.isOpened(): 
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame from camera")
            break
        # Mirror the camera frame horizontally
        frame = cv2.flip(frame, 1)

        # Time the inference
        inference_start = time.time()
        transformed = transforms(image=frame)
        result = model(torch.unsqueeze(transformed['image'], dim=0))
        inference_time = (time.time() - inference_start) * 1000  # Convert to ms

        probabilities = result['pred_logits'].softmax(-1)[:,:,:-1] 
        max_probs, max_classes = probabilities.max(-1)
        keep_mask = max_probs > 0.5  # Set IoU threshold

        batch_indices, query_indices = torch.where(keep_mask)

        # Limit to max_det = 1 (top confidence)
        if len(query_indices) > 0:
            top_idx = max_probs[batch_indices, query_indices].argmax()
            batch_indices = batch_indices[top_idx:top_idx+1]
            query_indices = query_indices[top_idx:top_idx+1]

        height, width = frame.shape[:2]
        bboxes = rescale_bboxes(result['pred_boxes'][batch_indices, query_indices,:], (width, height))
        classes = max_classes[batch_indices, query_indices]
        probas = max_probs[batch_indices, query_indices]

        # Prepare detection results for logging
        detections = []
        for bclass, bprob, bbox in zip(classes, probas, bboxes): 
            bclass_idx = bclass.detach().numpy()
            bprob_val = bprob.detach().numpy() 
            x1, y1, x2, y2 = map(int, bbox.detach().numpy())
            color = COLORS[bclass_idx]

            # Draw a thinner bounding box
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # Prepare label text
            label = f"{CLASSES[bclass_idx]}: {round(float(bprob_val), 2)}"

            # Calculate text size
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            # Draw filled rectangle for label background (with some transparency)
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
            alpha = 0.6  # Transparency factor
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Put label text
            cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        # Calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:  # Log every 30 frames
            elapsed_time = time.time() - fps_start_time
            fps = 30 / elapsed_time
            
            # Log detection results and performance
            if detections:
                detection_handler.log_detections(detections, frame_id=frame_count)
            detection_handler.log_inference_time(inference_time, fps)
            
            # Reset FPS counter
            fps_start_time = time.time()

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        # Quit on 'q' or ESC
        if key == ord('q') or key == 27:
            logger.realtime("Stopping real-time detection...")
            break

        # Handle window close (X button)
        try:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                logger.realtime("Window closed — stopping camera...")
                break
        except cv2.error:
            # If window no longer exists, stop
            break
finally:
    try:
        cap.release()
    except Exception:
        pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
