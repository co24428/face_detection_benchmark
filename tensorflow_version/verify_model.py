import tensorflow as tf
import cv2
import numpy as np

# TFLite 모델 로드
interpreter = tf.lite.Interpreter(model_path="mtcnn_tflite_model_v2.tflite")
interpreter.allocate_tensors()

# 입력/출력 텐서 정보 확인
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Input Details:", input_details)
print("Output Details:", output_details)

# 웹캠 연결
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지 전처리 (224x224, RGB)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_input = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0

    # TFLite 모델 실행
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Feature Map을 이용해 얼굴 감지 (임의로 강도 기준)
    feature_map = np.squeeze(output_data)  # (1280,) 형태
    height, width, _ = frame.shape
    threshold = 0.2  # 얼굴로 인식할 최소값 (0.2~0.5 조정 가능)

    # Feature Map을 강도 기반으로 얼굴 탐지 (임의 예제)
    detected_faces = []
    for i in range(0, len(feature_map), 4):  # Feature Map의 4개 단위로 (x, y, w, h)
        x, y, w, h = feature_map[i:i+4]
        if x > threshold and y > threshold:
            # 좌표 변환 (224x224 → 실제 프레임 크기)
            x1 = int(x * width)
            y1 = int(y * height)
            x2 = int((x + w) * width)
            y2 = int((y + h) * height)
            detected_faces.append((x1, y1, x2, y2))

    # 얼굴 박스 그리기
    for (x1, y1, x2, y2) in detected_faces:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # FPS 및 화면 출력
    cv2.imshow("TFLite MTCNN Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()