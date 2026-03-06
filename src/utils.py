import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt


MEANs = (0.485, 0.456, 0.406) # imagenet
STDs  = (0.229, 0.224, 0.225) # imagenet


CLIP_MEANs = (0.48145466, 0.4578275,  0.40821073)
CLIP_STDs  = (0.26862954, 0.26130258, 0.27577711)

imagenet_prompts = [
    "tench",
    "goldfish",
    "great white shark",
    "tiger shark",
    "hammerhead shark",
    "stingray",
    "cock",
    "hen",
    "ostrich",
    "brambling",
    "goldfinch",
    "house finch",
    "junco",
    "indigo bunting",
    "robin",
    "bulbul",
    "jay",
    "magpie",
    "chickadee",
    "water ouzel",
    "kite",
    "bald eagle",
    "vulture",
    "great grey owl",
    "European fire salamander",
    "common newt",
    "eft",
    "spotted salamander",
    "axolotl",
    "bullfrog",
    "tree frog",
    "tailed frog",
    "loggerhead sea turtle",
    "leatherback turtle",
    "mud turtle",
    "terrapin",
    "box turtle",
    "banded gecko",
    "green iguana",
    "Carolina anole",
    "desert grassland whiptail lizard",
    "agama",
    "frilled lizard",
    "alligator lizard",
    "Gila monster",
    "European green lizard",
    "chameleon",
    "Komodo dragon",
    "African chameleon",
    "Jackson's chameleon",
]

def freeze(module):
    for p in module.parameters():
        p.requires_grad_(False)


def save_images(images, img_path, title=None):
    n = len(images)
    n_cols = 4
    n_rows = (n+3) // 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_rows*8, n_cols*8))
    axes = axes.flatten()

    for i, axis in enumerate(axes):
        img = images[i]
        img = (img - img.min()) / (img.max() - img.min())
        axis.imshow((img * 255).astype(np.uint8))
        axis.axis("off")

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()
    Path(img_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(img_path, dpi=300)
    plt.close(fig)    


def save_heatmap(df, out_path):
    plt.figure(figsize=(6,5))

    sns.heatmap(
        df,
        annot=True,
        fmt=".3f",
        cmap="magma",
        square=True,
    )

    plt.title("inter-Mode CKA")
    plt.xlabel("Mode")
    plt.ylabel("Mode")

    plt.tight_layout()
    plt.savefig(Path(out_path) / "inter-models-cka.png", dpi=300)