import tensorflow as tf

# MobileNetV2 모델 로드 (사전 학습된 ImageNet)
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# HDF5 (.h5) 형식으로 저장
model.save("mobilenetv2_model.h5")
print("MobileNetV2 모델이 'mobilenetv2_model.h5' 파일로 저장되었습니다.")

# H5 모델 로드
model = tf.keras.models.load_model("mobilenetv2_model.h5")

# TFLite 변환
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 최적화 적용 (성능 향상)
tflite_model = converter.convert()

# TFLite 파일 저장
with open("mobilenetv2_model.tflite", "wb") as f:
    f.write(tflite_model)

print("MobileNetV2 TFLite 모델이 'mobilenetv2_model.tflite' 파일로 저장되었습니다.")