import tensorflow as tf
import cv2
import numpy as np

# TFLite 모델 로드
interpreter = tf.lite.Interpreter(model_path="mobilenetv2_model.tflite")
interpreter.allocate_tensors()

# 입력/출력 텐서 정보 확인
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]  # (224, 224)

# 출력 클래스 수 확인 (1000개가 맞는지 확인)
output_size = output_details[0]['shape'][1]
print(f"TFLite 모델 출력 클래스 수: {output_size}")

# ImageNet 클래스 이름 로드 (1000개 확인)
class_names = tf.keras.applications.mobilenet_v2.decode_predictions(np.zeros((1, 1000)))[0]
class_names = [item[1] for item in class_names]
print(f"ImageNet 클래스 수: {len(class_names)}")

# 웹캠 연결
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지 전처리 (224x224, RGB)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, input_shape)
    img_input = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0

    # TFLite 모델 예측
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # 상위 3개 예측 결과 확인
    top_indices = np.argsort(output_data[0])[-3:][::-1]
    predictions = [(class_names[i], output_data[0][i]) for i in top_indices if i < len(class_names)]

    # 예측 결과 좌상단에 표시
    y_offset = 20
    for idx, (label, score) in enumerate(predictions):
        text = f"{label}: {score:.2f}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25  # 각 줄 간격

    # 화면 출력
    cv2.imshow("TFLite MobileNetV2 Image Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()