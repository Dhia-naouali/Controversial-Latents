import torch
import pandas as pd
import torch.nn.functional as F

from .extractors import build_all_extractors
from .losses import linear_cka


def run_cross_eval(image_sets, run=None):
    extractors = build_all_extractors()
    modes = list(image_sets.keys())

    extractor_matrices = {}

    for extractor_name, extractor in extractors.items():
        extractor.eval()

        feats_per_mode = {}

        with torch.no_grad():
            for mode in modes:
                images = image_sets[mode].cuda()

                feats = extractor(images)

                if extractor_name == "classifier":
                    feats = F.softmax(feats.float(), dim=1)

                feats = F.normalize(feats, dim=1)
                feats_per_mode[mode] = feats

        matrix = torch.zeros(len(modes), len(modes))

        for i, m1 in enumerate(modes):
            for j, m2 in enumerate(modes):
                matrix[i, j] = linear_cka(
                    feats_per_mode[m1],
                    feats_per_mode[m2]
                )

        extractor_matrices[extractor_name] = matrix.cpu()

    avg_matrix = torch.stack(list(extractor_matrices.values())).mean(0)
    df = pd.DataFrame(
        avg_matrix.numpy(),
        index=modes,
        columns=modes
    ).round(4)

    return df
    