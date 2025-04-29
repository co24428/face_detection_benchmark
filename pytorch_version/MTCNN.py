import torch
import cv2
from facenet_pytorch import MTCNN
import time

# load model & set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, probs = mtcnn.detect(rgb_frame)

    end_time = time.time()
    inference_time = (end_time - start_time) * 1000
    fps = 1 / (end_time - start_time)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(frame, f"Inference: {inference_time:.2f} ms", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("PyTorch MTCNN Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()