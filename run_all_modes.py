import os, sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import hydra
from omegaconf import OmegaConf


import wandb
from src import (
    build_extractor, 
    build_generator,
    optimize_images
)

from huggingface_hub import login

HF_TOKEN = os.environ["HF_TOKEN"]
login(token=HF_TOKEN)


MODES = [
    ("pixels_ensemble",  "ensemble"),
    ("pixels_clip",      "clip"),
    ("pixels_kl",        "classifier"),
    ("flux",            "dino"),
]




def run_mode(mode_name, extractor_name, group=None):
    with hydra.initialize(config_path="configs"):
        config = hydra.compose(
            "default", overrides=[
                f"mode={mode_name}",
                f"extractor={extractor_name}",
                f"output.dir=outputs/{mode_name}"
            ]
        )

    run = wandb.init(
        "controversial-latents", 
        # group=run_group, 
        name=mode_name, 
        config=OmegaConf.to_container(config, resolve=True),
        reinit=True
    )

    extractor = build_extractor(config.extractor)
    generator = build_generator(config.mode) if mode_name == "flux" else None

    images, cross_eval = optimize_images(
        config, extractor, run=run, generator=generator
    )

    del extractor, generator
    torch.cuda.empty_cache()

    return {
        "mode": mode_name, "images": images, "cross_eval": cross_eval
    }



def main():
    parser = argparse.ArgumentParser()
    run_group = f"all_modes_{datetime.now()}"

    Path(f"outputs/{run_group}").mkdir(parents=True, exist_ok=True)
    results = []

    for mode_name, extractor_name in MODES:
        result = run_mode(mode_name, extractor_name, None)
        results.append(result)
        

if __name__ == "__main__":
    main()