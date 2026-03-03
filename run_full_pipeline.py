import os
import sys
from pathlib import Path

import hydra 
import torch
from omegaconf import OmegaConf


from src import (
    build_extractor, 
    build_generator,
    optimize_images, 
    train_contrastive,
    retrieve
)


def _extract_contrastive_training_config(config, run=None):
    return dict(
        images_dir=config.data.images_dir,
        output_dir=config.output.dir,
        config=config.contrastive,
        proj_dim=config.proj_dim,
        hidden_dim=config.hidden_dim,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        wd=config.wd,
        temperature=config.temperature,
        neg_ratio=config.neg_ratio,
        num_workers=config.data.get("num_workers", os.cpu_count() // 2),
        run=run
    )


@hydra.main(config_path="configs", config_name="config")
def main(config):
    images_dir = config.data.images_dir
    out_dir = Path(config.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_save_path = out_dir / "config.yaml"
    OmegaConf.save(config, config_save_path)
    print(OmegaConf.to_yaml(config, resolve=True))

    run = None

    extractor = build_extractor(config)
    generator = build_generator(config)

    images, cross_eval = optimize_images(config, extractor, run=run, generato=generator)
    del extractor, generator
    torch.cuda.empty_cache()

    
    contrastive_config = _extract_contrastive_training_config(config, run=run)
    model, _ = train_contrastive(
        images, **contrastive_config
    )

    paths, scores = retrieve(
        model, 
        images, 
        images_dir, 
        config.retrieval.topk,
        config.retrieval.batch_size,
        num_workers=config.data.num_workers,
        output_dir=out_dir
    )

    
