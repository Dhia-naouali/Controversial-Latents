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


# temp_device_contex