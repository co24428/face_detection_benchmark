# Face Detection Benchmark with TensorFlow and Pytorch

## Test Setup
- Environment: Mac, CPU-only
- Webcam: 720p
- Face counts: Single face vs Multiple faces
---

## Environment

**Python**: 3.12.5
- opencv-python: 4.11.0.86
- torch: 2.2.2
- torchvison: 0.17.2
- tensorflow: 2.19.0
- mtcnn: 1.0.0
- facenet-pytorch: 2.6.0
- ultralytics: 8.3.119

---

## Results

| Model                | Backend           | Faces      | FPS       | Inference Time (ms) | Notes |
|:---------------------|:------------------|:-----------|:----------|:--------------------|:------|
| YOLOv8n-face          | PyTorch (Ultralytics) | 1       | 27–28     | 34–36               | Very fast, stable |
| MTCNN                 | PyTorch (facenet-pytorch) | 1     | ~5        | 200                 | Moderate latency |
| MTCNN                 | TensorFlow        | 1         | 4.0–4.3   | 230                 | Slower |
| MTCNN                 | PyTorch (facenet-pytorch) | Multiple(~20)     | ~3        | 350                 | Significant slowdown |
| MTCNN                 | TensorFlow        | Multiple(~20)  | ~3.5      | 300                 | Significant slowdown |

---
---
## Models
### MTCNN (Multi-task Cascaded Convolutional Networks)

**MTCNN** is a deep learning model designed for face detection, with the following key characteristics:

- **Three-stage architecture** consisting of P-Net, R-Net, and O-Net.  
  Each network progressively filters and refines face candidate regions.

- Due to its **multi-stage structure,** inference speed may decrease significantly when multiple objects (faces) are present in the input image.

- The primary function of MTCNN is to **detect and return the location of faces**  
  (i.e., bounding boxes) within an image.  
  It can also optionally return facial landmarks such as eyes, nose, and mouth positions.

- **MTCNN alone does not support face similarity comparison or search.**  
  - To perform face recognition or search, additional feature extraction models like **Inception-ResNet V1** are required.
  - These models generate embeddings that can be compared using similarity metrics (e.g., cosine similarity).

> MTCNN is specialized in **detecting** faces, while **comparing or recognizing** faces requires an additional embedding model.
---
---
### YOLOv8n-face

**YOLOv8n-face** is a lightweight version (nano) of the YOLOv8 model optimized for face detection, with the following key characteristics:

- It is a **one-stage detection model** based on the YOLO (You Only Look Once) architecture.  
  - By passing the image through the network only once, it achieves fast inference speeds.

- Thanks to its **anchor-free design and optimized architecture,** performance degradation is minimal even when detecting multiple faces.

- Its primary function is to **detect and return the location of faces (bounding boxes).**  
  - It outputs precise bounding box coordinates along with a confidence score,
  - making it highly suitable for real-time applications.

- **YOLOv8n-face itself does not support face recognition (identification) or similarity comparison.**  
  - To enable these features, an additional embedding model must be used.

> YOLOv8n-face is optimized for **fast, lightweight, real-time face detection**,  
> but **additional models are required** for **face comparison and recognition** tasks.

---



## Analysis
- **MTCNN** is available with pre-trained models in both **PyTorch** and **TensorFlow**,  
  making it flexible for integration into different deep learning environments.

- However, due to its **multi-stage architecture (P-Net → R-Net → O-Net)**,  
  its performance degrades noticeably when detecting multiple faces,  
  regardless of the framework used. ( Pytorch is slightly better than TensorFlow)

- **YOLOv8n-face**, on the other hand, is implemented through the **Ultralytics** library,  
  which is built on **PyTorch** but not a pure PyTorch implementation.

- As a **one-stage detector**, YOLOv8n-face offers significantly better performance  
  in terms of both **FPS (Frames Per Second)** and **inference time**,  
  especially in real-time and multi-face scenarios.