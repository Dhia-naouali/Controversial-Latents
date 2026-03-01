import torch
from torch import nn
import torch.nn.functional as F

import timm
from transformers import AutoModel, CLIPModel

from .utils import (
    MEANs as IN_MEANs, 
    STDs as IN_STDs, 
    CLIP_MEANs, 
    CLIP_STDs,
    freeze
)


def in_to_clip_norm(x):
    return x


class BaseExtractor(nn.Module):
    feat_dim: int

    def forward(self, x):
        feats = self._encode(x):
        return F.normalize(feats, dim=1)
    
    def _encode(self, x):
        raise NotImplemented()
    

class Dino(BaseExtractor):
    feat_dim = 768

    def __init__(self, model_id="facebook/dinov2-base"):
        self.model = AutoModel.from_pretrained(model_id).eval()
        freeze(self.model)

    def _encode(self, x):
        return self.model(x).last_hidden_state[:, 0]
    

class IJEPA(BaseExtractor):
    feat_dim = 1280
    def __init__(self, model_id="facebook/ijepa_vith14_1k"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_id).eval()
        freeze(self.model)


    def _encode(self, x):
        return self.model(x).last_hidden_state[:, 1:].mean(dim=1)
    

class CLIP(BaseExtractor):
    feat_dim = 768

    def __init__(self, model_id="openai/clip-vit-large-patch14"):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_id).eval()
        for p in self.model.parameters():
            p.requries_grad_(False)
        
    def _encode(self, x):
        x = in_to_clip_norm(x)
        return self.model.vision_model(x).pooler_output
    

class Classifier(BaseExtractor):
    feat_dim = 1000
    def __init__(self, model_name="resnet50"):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=1000).eval()
        freeze(self.model)

        
    def forward(self, x):
        return self.model(x)
    
    def _encode(self, x):
        return self(x)


class Ensemble(BaseExtractor):
    def __init__(
            self, models, per_model_loss=True
    ):
        super().__init__()
        self.models = nn.ModuleList([m[1] for m in models])

        self.names = [m[0] for m in models]
        self.weights = [m[2] for m in models]

        self.per_model_loss = per_model_loss
        self.feat_dim = sum(m.feat_dim for m in self.models)

    def forward(self, x):
        results = {
            name: model(x) for name, model in zip(self.names, self.models)
        }

        if self.per_model_loss:
            return results
        
        preds = torch.cat(
            [results[name] * w for name, w in zip(self.names, self.weights)], 
            dim=1
        )

        return F.normalize(preds, dim=1)
    


def build_extractor(config):
    name = config.name

    if name == "dino":
        model = Dino(model_id=config.get("model", "facebook/dinov2-base"))
    elif name == "ijepa":
        model = IJEPA(model_id=config.get("model", "facebook/ijepa-vith14_1k"))
    elif name == "clip":
        model = CLIP(model_id=config.get("model", "openai/clip-vit-large-patch14"))
    elif name == "classifier":
        model = Classifier(model_name=config.get("model", "resnet50"))
    elif name == "ensemble":
        models = []
        for m in config.models:
            child_config = m
            child_name = child_config.name
            w = child_config.get("weight", 1.0)
            if child_name == "dino":
                child = Dino(child_config.get("model", "facebook/dinov2-base"))
            elif child_name == "ijepa":
                child = IJEPA(child_config.get("model", "facebook/ijepa_vith14_1k"))
            elif name == "classifier":
                child = Classifier(model_name=child_config.get("model", "resnet50"))
            else:
                raise ValueError(f"unknown ensemble member: {child_name}")
            models.append((child_name, child, w))
        model = Ensemble(models, per_extractor_loss=True)
    else:
        raise ValueError(f"unknown ensemble member: {child_name}")

    return model.cuda()



def build_all_extractors():
    return {
        mn: m.eval().cuda() for mn, m in 
        {
            "dino": Dino(),
            "ijepa": IJEPA(),
            "clip": CLIP(),
            "classifier": Classifier()
        }.items()
    }