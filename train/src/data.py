import numpy as np
from PIL import Image
import cv2
import torch
from torch import nn
from torchvision import transforms as T

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

def imagenet_denorm(x):
    """x: array-like with shape (..., H, W, C)"""
    return x * imagenet_std + imagenet_mean

class Dataset(torch.utils.data.Dataset): 
    def __init__(self, fo_dataset, is_train=True):
        super().__init__()
        self.fo_dataset = fo_dataset
        self.is_train = is_train
        self.fns = fo_dataset.values("filepath")
        self.tfms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
            T.Normalize(imagenet_mean, imagenet_std)
        ])

    def __len__(self):
        return len(self.fo_dataset)
    def __getitem__(self, index):
        fn = self.fns[index]
        x = self.fo_dataset[fn]
        im = Image.open(fn).convert("RGB")
        inpt = self.tfms(im)
        maskAll = np.zeros(inpt.shape[1:], np.uint8)
        height, width = inpt.shape[1:]
        for det in x.ground_truth.detections:
            if det.mask is None:
                continue
            relX, relY, relW, relH = det.bounding_box
            x1 = int(width * relX)
            x2 = int(width * (relX + relW))
            y1 = int(height * relY)
            y2 = int(height * (relY + relH))
            mask = np.where(det.mask, 255, 0).astype(np.uint8)
            maskAll[y1:y2, x1:x2] |= cv2.resize(mask, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
        target = torch.Tensor(np.where(maskAll > 0, 1.0, 0.0).astype(np.float32)).unsqueeze(0)
        if self.is_train:
            if np.random.rand() < 0.5:
                inpt = torch.fliplr(inpt)
                target = torch.fliplr(target)
        return inpt, target
