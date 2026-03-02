import os
import sys
import json
import argparse
import pandas as pd
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


MODES2EXT = {
    "pixels_ensemble": "ensemble",
    "pixels_clip": "clip",
    "pixels_kl": "classifier",
    "flux": "dino",
}


def run_mode(mode_name, extractor_name, group=None):
    with hydra.initialize(config_path="configs"):
        config = hydra.compose(
            "default", overrides=[
                f"mode={mode_name}",
                f"extractor={extractor_name}",
                f"output.dir=outputs/{mode_name}"
            ]
        )

    # run = wandb.init(
    #     project="controversial-latents", 
    #     # group=run_group, 
    #     name=mode_name, 
    #     config=OmegaConf.to_container(config, resolve=True),
    #     reinit=True
    # )
    run = None

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

def cross_eval_matrix(_):
    return {}


def main():
    parser = argparse.ArgumentParser()
    run_group = f"all_modes_{datetime.now()}"

    Path(f"outputs/{run_group}").mkdir(parents=True, exist_ok=True)
    results = []

    for mode_name, extractor_name in MODES2EXT.items():
        result = run_mode(mode_name, extractor_name, None)
        results.append(result)
        
    # if len(results) > 1:
    #     val_matrix = cross_eval_matrix(results)
    #     df = pd.DataFrame(val_matrix).T
    #     df = df.round(4)
    #     print(df)
        
        # for mode_name, extractor_score in val_matrix.items():



if __name__ == "__main__":
    main()