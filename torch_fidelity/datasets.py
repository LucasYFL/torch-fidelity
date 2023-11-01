import sys
from contextlib import redirect_stdout

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, STL10, CIFAR100
import torchvision.transforms.functional as F
import albumentations
from torch_fidelity.helpers import vassert
import numpy as np

class TransformPILtoRGBTensor:
    def __call__(self, img):
        vassert(type(img) is Image.Image, 'Input is not a PIL.Image')
        return F.pil_to_tensor(img)


class ImagesPathDataset(Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = TransformPILtoRGBTensor() if transforms is None else transforms
        self.npf = files[0].strip().endswith(".npy")
        if self.npf:
            self.size = 256
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            if True:#not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        if self.npf:
            img = self.preprocess_image(path)
        else:
            img = Image.open(path).convert('RGB')
            img = self.transforms(img)
        return img
    
    def preprocess_image(self, image_path):
        image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        #image = (image/127.5 - 1.0).astype(np.float32)
        image = np.transpose(image, (2,0,1))
        image = torch.from_numpy(image)
        return image

class Cifar10_RGB(CIFAR10):
    def __init__(self, *args, **kwargs):
        with redirect_stdout(sys.stderr):
            super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img
    

class Cifar100_RGB(CIFAR100):
    def __init__(self, *args, **kwargs):
        with redirect_stdout(sys.stderr):
            super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img


class STL10_RGB(STL10):
    def __init__(self, *args, **kwargs):
        with redirect_stdout(sys.stderr):
            super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img


class RandomlyGeneratedDataset(Dataset):
    def __init__(self, num_samples, *dimensions, dtype=torch.uint8, seed=2021):
        vassert(dtype == torch.uint8, 'Unsupported dtype')
        rng_stash = torch.get_rng_state()
        try:
            torch.manual_seed(seed)
            self.imgs = torch.randint(0, 255, (num_samples, *dimensions), dtype=dtype)
        finally:
            torch.set_rng_state(rng_stash)

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, i):
        return self.imgs[i]
