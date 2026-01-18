from torch.utils.data import DataLoader
from model import UNet
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataset_loader import CocoSegmentationDataset  # type: ignore
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = CocoSegmentationDataset(
    img_dir=r"C:\Users\suyas\OneDrive\Desktop\image_caption_segmentation_project\data\train2017",
    ann_file=r"C:\Users\suyas\OneDrive\Desktop\image_caption_segmentation_project\data\annotations\instances_train2017.json"
)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(2):  # Reduced from 5
    total_loss = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "segmentation_model.pth")
