MEANs = (0.485, 0.456, 0.406) # imagenet
STDs  = (0.229, 0.224, 0.225) # imagenet


CLIP_MEANs = (0.48145466, 0.4578275,  0.40821073)
CLIP_STDs  = (0.26862954, 0.26130258, 0.27577711)



def freeze(module):
    for p in module.parameters():
        p.requires_grad_(False)



# temp_device_contex