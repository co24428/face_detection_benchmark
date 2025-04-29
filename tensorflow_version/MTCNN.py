import time
import cv2
from mtcnn import MTCNN

detector = MTCNN()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # fps calculation
    start_time = time.time()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_frame)
    
    # fps calculation
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # ms
    fps = 1 / (end_time - start_time)
    
    for face in faces:
        x, y, width, height = face["box"]
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)


    cv2.putText(frame, f"Inference: {inference_time:.2f} ms", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("TensorFlow MTCNN Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()