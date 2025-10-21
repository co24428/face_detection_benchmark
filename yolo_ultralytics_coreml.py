from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolov8n.pt")

# Export the model to CoreML format
model.export(format="coreml")  # creates 'yolo11n.mlpackage'

# Load the exported CoreML model
coreml_model = YOLO("yolov8n.mlpackage")

# Run inference
results = coreml_model("https://ultralytics.com/images/bus.jpg")