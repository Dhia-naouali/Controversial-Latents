import torch
from torch import nn
import torch.nn.functional as F
import timm

from .utils import freeze

class Backbone(nn.Module):
    def __init__(
        self,
        model_name,
        layer_name,
    ):
        super().__init__()
        self.model_name = model_name
        self.layer_name = layer_name

        self._backbone = timm.create_model(model_name, pretrained=True, num_classes=0).eval()
        freeze(self._backbone)

        self._feats = None
        self._hook_handle = self._register_hook()

        sample_input = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            self._backbone(sample_input)
        self.feat_dim = self._feats.shape[1]

    def _register_hook(self):
        module = self._backbone

        # recursively resolve module / layer
        for part in self.layer_name.split("."):
            module = getattr(module, part)

        def hook(module_, ins, outs):
            if outs.ndim == 4:
                self._feats = outs.mean(dim=[2, 3])
            elif outs.ndim == 3:
                self._feats = outs[:, 0]
            else:
                self._feats = outs
    
        return module.register_forward_hook(hook)
    
    def forward(self, x):
        self._backbone(x)
        return F.normalize(self._feats, dim=1)
    
    def __del__(self):
        if hasattr(self, "_hook_handle"):
            self._hook_handle.remove()
            super().__del__()


class Head(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, proj_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim, bias=False),
            nn.LayerNorm(proj_dim, affine=False)
        )

    def forward(self, x):
        return F.normalize(self.mlp(x), dim=1)
    

class SimCLRModel(nn.Module):
    def __init__(
        self, 
        model_name="resnet50",
        proj_dim=128,
        hidden_dim=512,
        layer_name="layer4",
    ):
        super().__init__()
        self.backbone = Backbone(model_name, layer_name)
        self.head = Head(self.backbone.feat_dim, hidden_dim, proj_dim)

    def forward(self, x):
        return self.head(
            self.backbone(x)
        )
    
    @torch.no_grad()
    def encode(self, x):
        return self.backbone(x)