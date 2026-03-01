import random
from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image

from .utils import MEANs, STDs

def default_transform(img_size=224):
    return T.Compose([
        T.Resize(max(img_size, 256)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=MEANs, std=STDs)
    ])

def aug_transform(img_size=224, jitter_strength=.5):
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(
            0.4*jitter_strength, 
            0.4*jitter_strength, 
            0.4*jitter_strength, 
            0.2*jitter_strength
        )], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize(mean=MEANs, std=STDs),
    ])


class ImageDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = [Path(p) for p in paths]
        self.transform = transform or default_transform()

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, str(path)


class PositiveImagesDataset(Dataset):
    def __init__(self, images, transform=None):
        mean = torch.tensor(MEANs).view(1, 3, 1, 1)
        std = torch.tensor(STDs).view(1, 3, 1, 1)
        self.images = (images * std + mean).clamp(0., 1.)
        self.transform = transform or aug_transform()
        self.to_pil = T.ToPILImage()

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        return self.transform(
            self.to_pil(img)
        )



class NegativeImageDataset(Dataset):
    def __init__(
            self, 
            paths, 
            n_samples, 
            transform=None, 
            seed=12
    ):
        rng = random.Random(seed)
        self.paths = rng.sample(paths, min(n_samples, len(paths)))
        self.transform = transform or aug_transform()

    def __len__(self):
        return len(self.paths)
    

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img)
    

class ContrastiveDataset(Dataset):
    def __init__(self, pos_dataset, neg_dataset, neg_ratio=8, seed=12):
        rng = random.Random(seed)
        self.pos = pos_dataset
        self.neg = neg_dataset
        self.neg_ratio = neg_ratio

        self.indices = [
            ("pos", i) for i in range(len(self.pos))
        ]
        
        neg_count = len(self.pos) * neg_ratio
        self.indices += [
            ("neg", rng.randint(0, len(self.neg)-1)) 
            for _ in range(neg_count)
        ]

        rng.shuffle(self.indices)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        type_, i = self.indices[idx]
        if type_ == "pos":
            view1 = self.pos[i]
            view2 = self.pos[i]
            label = 1
        else:
            view1 = self.neg[i]
            view2 = self.neg[i]
            label = 0
        return view1, view2, torch.tensor(label, dtype=torch.long)
