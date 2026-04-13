import cv2
import torch
import numpy as np
from depth_anything_v3.dpt import DepthAnythingV3

# 1. Load Model (Use "small" for better real-time performance)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DepthAnythingV3.from_pretrained("depth-anything/DA3-SMALL").to(device).eval()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # 2. Pre-process frame for DA3
    # Resize and normalize according to model requirements (usually handled by a provided transform)
    image_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    with torch.no_grad():
        # 3. Perform Inference
        # DA3 outputs multiple heads; we'll focus on depth for now
        output = model(image_tensor)
        depth = output['depth'] # Extract depth map

    # 4. Post-process and Colorize for Visualization
    depth_np = depth.squeeze().cpu().numpy()
    depth_rescaled = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_rescaled, cv2.COLORMAP_MAGMA)

    # 5. Display both feeds
    cv2.imshow('Original Feed', frame)
    cv2.imshow('DA3 Depth Map', depth_colored)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
