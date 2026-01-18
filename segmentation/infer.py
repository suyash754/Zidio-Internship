import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from segmentation.model import UNet
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet().to(device)
model.load_state_dict(torch.load("segmentation_model.pth", map_location=device))
model.eval()

def predict_mask(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.Resize((256, 256)),T.ToTensor()])
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        output = torch.sigmoid(output).squeeze().cpu().numpy()

    return image, output

def visualize(image, mask):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Predicted Mask")
    plt.show()

# Example
if __name__ == "__main__":
    img_path = r"C:\Users\suyas\OneDrive\Desktop\image_caption_segmentation_project\data\val2017\000000000885.jpg"
    img, msk = predict_mask(img_path)
    visualize(img, msk)
