from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

from .extractors import build_all_extractors
from .losses import compute_mpcd

def run_cross_eval(
    image_sets,
    run=None,
    save_path=None
):
    extractors = build_all_extractors()
    results = {}

    for mode_name, images in image_sets.items():
        results[mode_name] = {}
        row_vals = []

        for extractor_name, extractor in extractors.items():
            with torch.no_grad():
                feats = extractor(images.cuda())
                if extractor_name == "classifier":
                    feats = F.softmax(feats.float(), dim=1)
                mpcd = compute_mpcd(feats)
            
            results[mode_name][extractor_name] = mpcd
            row_vals.append(mpcd)

            # if run is not None:
            #     run.log()

    # if run is not None:
    # log results matrix

    if save_path is not None:
        _save_heatmap(results, save_path)