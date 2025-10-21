import tensorflow as tf
import coremltools as ct

# TensorFlow MobileNetV2 모델 로드
model = tf.keras.applications.MobileNetV2(weights='imagenet', input_shape=(224, 224, 3))

# Core ML로 변환 (소스를 명시적으로 지정)
mlmodel = ct.convert(
    model,
    source="tensorflow",  # 여기서 소스 프레임워크를 명시적으로 지정
    inputs=[ct.ImageType(name="image", shape=(1, 224, 224, 3))]
)

# Core ML 모델 저장
mlmodel.save("MobileNetV2.mlmodel")