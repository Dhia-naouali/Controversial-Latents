import os
import glob
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from .models import SimCLRModel
from .data import (
    PositiveImagesDataset,
    NegativeImageDataset,
    ContrastiveDataset,
)
from .losses import nt_xent_loss

def train_contrastive(
    optimized_images,
    images_dir,
    backbone="resnet50",
    layer_name="layer4",
    proj_dim=128,
    hidden_dim=512,
    epochs=12,
    batch_size=128,
    lr=1e-3,
    wd=1e-2,
    temperature=7e-2,
    neg_ratio=8,
    output_dir=None,
    num_workers=os.cpu_count()//2,
    run=None
):
    pos_dataset = PositiveImagesDataset(optimized_images)
    
    images_paths = (
        glob.glob(f"{images_dir}/*.jpg") + 
        glob.glob(f"{images_dir}/*.JPEG") + 
        glob.glob(f"{images_dir}/*.png")
    )
    n_neg = len(optimized_images) * neg_ratio * 2 # sampling from a larger pool for higher samples var
    neg_dataset = NegativeImageDataset(images_paths, n_samples=n_neg)
    contrastive_dataset = ContrastiveDataset(pos_dataset, neg_dataset, neg_ratio=neg_ratio)
    loader = DataLoader(
        contrastive_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    model = SimCLRModel(
        backbone,
        proj_dim,
        hidden_dim,
        layer_name
    ).cuda()

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=wd
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(loader), eta_min=lr*.1)

    best_loss = float("inf")
    ckpt_path = None

    if output_dir:
        ckpt_path = Path(output_dir) / "SimCLR_model.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    pb = tqdm(range(1, epochs+1))
    for epoch in pb:
        model.train()
        epoch_loss = 0.

        for v1, v2, labels in loader:
            v1 = v1.cuda(non_blocking=True)
            v2 = v2.cuda(non_blocking=True)
            optimizer.zero_grad()

            z1, z2 = torch.chunk(
                model(
                    torch.cat([
                        v1, v2
                    ], dim=0)
                ),
                2, dim=0
            )

            loss = nt_xent_loss(z1, z2, temperature)
            loss.backward()
            nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=1.)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

            pb.set_postfix(epoch=epoch, loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.4f}")

        epoch_loss /= len(loader)        
        if epoch_loss < best_loss:
            if ckpt_path:
                torch.save(model.head.state_dict(), ckpt_path)
            best_loss = epoch_loss

    if ckpt_path and ckpt_path.exists():
        model.head.load_state_dict(torch.load(ckpt_path, map_location="cuda"))

    return model.eval()