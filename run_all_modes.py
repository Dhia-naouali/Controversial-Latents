import os
import argparse
from pathlib import Path
from datetime import datetime

import torch
import hydra

import wandb
from src import (
    build_extractor, 
    build_generator,
    optimize_images,
    save_heatmap,
    run_cross_eval
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


def run_mode(mode_name, extractor_name, run=None):
    with hydra.initialize(config_path="configs"):
        config = hydra.compose(
            "default", overrides=[
                f"mode={mode_name}",
                f"extractor={extractor_name}",
            ]
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
    }, config


def cross_eval_matrix(_):
    return {}


def main():
    # parser = argparse.ArgumentParser()
    run_group = f"all_modes_{datetime.now()}"

    run = wandb.init(
        project="controversial-latents", 
    )


    Path(f"outputs/{run_group}").mkdir(parents=True, exist_ok=True)
    results = []

    for mode_name, extractor_name in MODES2EXT.items():
        result, config = run_mode(mode_name, extractor_name, run)
        results.append(result)

    if len(results) > 1:
        df = run_cross_eval({r["mode"]: r["images"] for r in results})
        print(df)
        if config.output.dir:
            save_heatmap(df, config.output.dir)


if __name__ == "__main__":
    main()