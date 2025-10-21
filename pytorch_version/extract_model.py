import torch
from facenet_pytorch import MTCNN
import numpy as np

# Load MTCNN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device).eval()  # evaluation mode

# 예제 이미지 (더미 이미지, 실제 테스트 시 필요)
dummy_image = torch.randn(1, 3, 224, 224).to(device)  # 3채널, 224x224 이미지

# Saved model as torchscript, using tracing
traced_mtcnn = torch.jit.trace(mtcnn, dummy_image)
traced_mtcnn.save("mtcnn_traced.pt")



# # TorchScript로 모델 변환 (Trace 방식)
# traced_mtcnn = torch.jit.trace(mtcnn, dummy_image)
# traced_mtcnn.save("mtcnn_traced.pt")

# print("MTCNN TorchScript 모델이 'mtcnn_traced.pt' 파일로 저장되었습니다.")