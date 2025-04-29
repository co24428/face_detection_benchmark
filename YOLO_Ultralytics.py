import time
import cv2
from ultralytics import YOLO

# load a pretrained model
model = YOLO("yolov8n-face.pt") 

# connect to webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    start_time = time.time()

    
    # inference
    results = model.predict(frame, imgsz = 640, conf = 0.5)
    
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # ms
    fps = 1 / (end_time - start_time)
    
    # show bbox in frame
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

   # Inference time, FPS display
    cv2.putText(frame, f"Inference: {inference_time:.2f} ms", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    
    # show the frame
    cv2.imshow("Yolov8 Face Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
            