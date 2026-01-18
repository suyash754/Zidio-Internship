from captioning.infer import generate_caption
from segmentation.model import UNet
import torch
from segmentation.infer import predict_mask
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np

# --- CONFIG ---
image_path = r"C:\Users\suyas\OneDrive\Desktop\image_caption_segmentation_project/data/val2017/000000000139.jpg"  # Sample image path
mask_model_path = "segmentation_model.pth"    # Trained U-Net model path

# --- STEP 1: Generate Caption ---
caption = generate_caption(image_path)

# --- STEP 2: Predict Segmentation Mask ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load(mask_model_path, map_location=device))
model.eval()

image = Image.open(image_path).convert("RGB")
img_tensor = torch.tensor(np.array(image).transpose(2, 0, 1) / 255.0, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img_tensor)
    mask = torch.sigmoid(output).squeeze().cpu().numpy()

# --- STEP 3: Visualize Everything ---
plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.axis('off')
plt.title("Original Image")

# Predicted Mask
plt.subplot(1, 3, 2)
plt.imshow(mask, cmap='gray')
plt.axis('off')
plt.title("Predicted Mask")

# Caption
plt.subplot(1, 3, 3)
plt.imshow(image)
plt.axis('off')
plt.title("Caption:\n" + caption)

plt.tight_layout()
plt.show()
