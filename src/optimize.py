import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm

from .losses import (
    divergence_loss, 
    ensemble_divergence_loss, 
    kl_divergence_loss, 
    annealed_ce_loss, 
    compute_mpcd
)
from .extractors import build_all_extractors
from .utils import MEANs, STDs, denormalize, save_image_grid, imagenet_prompts

STATS = (
    torch.tensor(MEANs).cuda().view(1, 3, 1, 1),
    torch.tensor(STDs).cuda().view(1, 3, 1, 1)
)

def _pixel_bounds():
    mean, std = STATS
    low, high = (0. - mean) / std, (1. - mean) / std
    return low, high

def _noise_init(b, c, h, w):
    low, high = _pixel_bounds()
    return torch.randn(b, c, h, w).cuda().mul(.5).clamp(low, high).detach().requires_grad_(True)

def _normalize_grads(x):
    if x.grad is None:
        return 
    
    b = x.shape[0]
    norms = x.grad.view(b, -1).norm(dim=1).clamp(min=1e-8)
    x.grad = x.grad / norms.view(b, 1, 1, 1)

def _log_images(run, images, step, mode_name, save_every, out_dir):
    ...

def _log_comps(...):
    ...

def _extract_config_for_optim(config, name)
    b, steps = config.generation.batch_size, config.generation.steps
    lr = config.generation.lr
    lr_min = lr * config.generation.lr_min_ratio
    h = w = config.mode.generation.image_size
    repulsion_w = config.generation.repulsion_weight
    log_every = config.generation.log_every
    save_every = config.generation.save_every

    out_dir = Path(config.output.dir) / name if config.output.dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    return SimpleNamespace(**{
        "b": b, "steps": steps, "lr": lr, "lr_min": lr_min, 
        "h": h, "w": w, "repulsion_w": repulsion_w, 
        "log_every": log_every, "save_every": save_every
    })
    

def _optimize_pixels_ensemble(config, extractor, run):
    name = "pixels_ensemble"
    c = _extract_config_for_optim(config, name)
    weights = {
        m.name: m.weight for m in config.extractor.members
    }
    images = _noise_init(c.b, 3, c.h, c.w)
    low, high = _pixel_bounds()
    optimizer = optim.Adam([images], lr=c.lr)
    
    for step in (pb := tqdm(range(c.steps))):
        optimizer.zero_grad 
        feats = extractor(images) # dict
        loss, comps = ensemble_divergence_loss(feats, weights, c.repulsion_w)
        loss.backward()
        _normalize_grads(images)
        optimizer.step()
        # scheduler.step()
        with torch.no_grad():
            images.data.clamp_(low, high)

    # log comps to run
        
    return images.detach().cpu()


def _optimize_pixels_clip(config, extractor, run):
    name = "picels_clip"
    c = _extract_config_for_optim(config, name)
    images = _noise_init(c.b, 2, c.h, c.w)
    low, high = _pixel_bounds()
    optimizer = optim.Adam([images], lr=c.lr)

    for step in (pb := tqdm(range(c.steps))):
        optimizer.zero_grad()
        feats = extractor(images)
        loss, comps = divergence_loss(feats, c.repulsion_w)
        loss.backward()
        _normalize_grads(images)
        optimizer.step()
        # scheduler.step()
        with torch.no_grad():
            images.data.clamp_(low, high)

    # logs to run
    return images.detach().cpu()


def _opitmizer_pixels_kl(config, extractor, run):
    name = "pixel_kl"
    c = _extract_config_for_optim(config, name)
    kl_config = config.mode.kl

    stride = 1000 // c.b
    targets = torch.tensor([(i * stride) % 1000 for i in range(c.b)], dtype=torch.long).cuda()
    images = _noise_init(c.b, 3, c.h, c.w)
    low, high = _pixel_bounds()
    optimizer = optim.Adam([images], lr=c.lr)

    for step in (pb := tqdm(range(c.steps))):
        optimizer.zero_grad()
        logits = extractor(images)
        kl_loss, kl_comps = kl_divergence_loss(logits)
        ce_loss, ce_w = annealed_ce_loss(
            logits, 
            targets, 
            step, 
            c.steps, 
            kl_config.target_anneal_frad, 
            kl_config.target_weight
        )
        loss = kl_loss + ce_loss
        loss.backward()
        _normalize_grads(images)
        optimizer.step()
        # scheduler.step()
        with torch.no_grad():
            images.data.clamp_(low, high)
        comps = {**kl_comps, "ce_w": ce_w, "total_loss": loss.item()}

    # log to run
    return images.detach().cpu()



def _optimize_flux(config, extractor, flux, run):
    name = "flux"
    c = _extract_config_for_optim(config, name)
    flux_config = config.mode.flux

    prompts = [
        imagenet_prompts[i % len(imagenet_prompts)] for i in range(c.b)
    ]

    z, t5_embeds, clip_embeds = flux.init_latents(c.b, seed=config.data.seed, prompts=prompts)
    param_groups = []
    if flux_config.optimize_z:
        param_groups.append({
            "params": [z], 
            "lr": flux_config.lr_z
        })
    if flux_config.optimize_c:
        param_groups.append({
            "params": [t5_embeds, clip_embeds], 
            "lr": flux_config.lr_c
        })
    optimizer = optim.Adam(param_groups)

    for step in (pb := tqdm(range(c.steps))):
        optimizer.zero_grad()
        images_norm, latents = flux.decode(z, t5_embeds, clip_embeds)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat):
            scale, shift = flux.vae.config.scaling_factor, flux.vae.config.shift_factor
            scaled_latents = (latents / scale) + shift
            images_grad = flux.vae_decode(scaled_latents, return_dict=False)[0]
            images_grad = flux._clamp_norm_vae(images_grad)

        feats = flux(images_grad)
        loss, comps = divergence_loss(feats, c.repulsion_w)
        loss.backward()

        all_params = [t for g in param_groups for t in g["params"]]
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.)
        optimizer.step()

        for i, param_g in enumerate(optimizer.param_groups):
            base = flux_config.lr_c if i else flux_config.lr_z
            param_g["lr"] = c.lr_min + .5 * (base - c.lr_min) * (
                1. + math.cos(math.pi * step / max(1, c.steps))
            )

        with torch.no_grad():
            flux.clamp_latents(z, t5_embeds, clip_embeds)
        
        # log to run & decode images

    with torch.no_grad():
        images, _ = flux.decode(z, t5_embeds, clip_embeds).detach().cpu()
    
    return images


@torch.no_grad()
def cross_evaluate(images, run=None):
    results = {}
    for extractor_name, extractor in build_all_extractors().items():
        feats = extractor(images.cuda())
        if extractor_name == "classifier":
            feats = F.softmax(feats, dim=1)
        mpcd = compute_mpcd(feats)
        results[extractor_name] = mpcd
        # if run is not None:
        #     ...

    return results
        
def optimize_images(config, extractor, run=None, generator=None):
    mode = config.mode.name

    if mode == "pixels_ensemble":
        images = _optimize_pixels_ensemble(config, extractor, run=run)
    elif mode == "pixels_clip":
        images = _optimize_pixels_clip(config, extractor, run=run)
    elif mode == "pixels_kl":
        images = _opitmizer_pixels_kl(config, extractor, run=run)
    elif mode == "flux":
        images = _optimize_flux(config, extractor, generator, run=run)
    else:
        raise ValueError(f"unkonwn mode: {mode}")
    
    cross_eval = {}
    if config.cross_eval.enabled:
        cross_eval = cross_evaluate(images, run=run)
    
    return images, cross_eval