import os
from ultralytics import YOLO

MODEL_PATH = os.path.join("models", "yolov8n.pt")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"YOLO model not found at {MODEL_PATH}. Please download manually.")

class ObjectDetector:
    def __init__(self, confidence=0.4):
        self.model = YOLO(MODEL_PATH)
        self.confidence = confidence

    def detect(self, frame):
        results = self.model(frame, conf=self.confidence, verbose=False, device="cpu")[0]
        detections = []
        if results.boxes is None:
            return detections

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "label": label,
                "confidence": round(conf, 2),
                "bbox": (x1, y1, x2, y2)
            })
        return detections
