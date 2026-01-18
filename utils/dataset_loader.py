from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import torch
import os
from torchvision import transforms

class CocoSegmentationDataset(Dataset):
    def __init__(self, img_dir, ann_file, image_size=(128, 128)):
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())[:5000]
        self.img_dir = img_dir
        self.image_size = image_size

        self.resize_image = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        self.resize_mask = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        image = Image.open(img_path).convert("RGB")
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in anns:
            mask += self.coco.annToMask(ann)
        mask = np.clip(mask, 0, 1)

        # Convert to numpy → torch and resize both image and mask
        image = np.array(image)
        mask = mask.astype(np.uint8) * 255  # make it proper binary mask

        image = self.resize_image(image)
        mask = self.resize_mask(mask)

        return image, mask
