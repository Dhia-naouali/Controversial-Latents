import torch
import torch.nn.functional as F


def divergence_loss(feats, repulsion_weight=5e-2):
    feats = F.normalize(feats, dim=1)
    b = feats.shape[0]
    sim = feats @ feats.T
    mask = ~torch.eye(b, dtype=torch.bool).cuda()
    pw_dist = 1. - sim[mask]
    mpcd = pw_dist.mean()
    repulsion = torch.exp(-pw_dist).mean()
    loss = -mpcd + repulsion * repulsion_weight
    return loss, {
        "loss": loss.item(), 
        "mpcd": mpcd.item(), 
        "repulsion": repulsion.item()
    }

def ensemble_divergence_loss(feats_dict, weights=None, repulsion_weight=5e-2):
    if weights is None:
        w = 1. / len(feats_dict)
        weights = {k: w for k in feats_dict}

    total_loss = torch.tensor(0.).cuda()
    components = {}
    for name, feats in feats_dict.items():
        loss, comps_ = divergence_loss(feats, repulsion_weight)
        total_loss += loss * weights.get(name, 1. / len(feats_dict))

        for k, v in comps_.items():
            components[f"{name}_{k}"] = v

    components["loss"] = total_loss.item()
    return total_loss, components

def kl_divergence_loss(logits):
    probs = F.softmax(logits, dim=1)
    log_probs = torch.log(probs + 1e-8)
    b = probs.shape[0]
    p = probs.unsqueeze(0).expand(b, b, -1)
    lp = log_probs.unsqueeze(0).expand(b, b, -1)

    q = probs.unsqueeze(1).expand(b, b, -1)
    lq = log_probs.unsqueeze(1).expand(b, b, -1)

    kl = .5 * ((p * (lp-lq)) + (q * (lq-lp))).sum(dim=-1)
    mask = ~torch.eye(b, dtype=torch.bool).cuda()
    mean_kl = kl[mask].mean()
    
    return -mean_kl, {"mean_kl": mean_kl.item()}

def annealed_ce_loss(logits, targets, step, total_steps, anneal_frac=.3, initial_w=1.):
    progress = step / max(1, anneal_frac*total_steps)
    weight = initial_w * max(0., 1. - progress)
    if weight < 1e-6:
        return torch.tensor(0.).cuda(), 0.
    
    return weight * F.cross_entropy(logits, targets), weight


def nt_xent_loss(z1, z2, temp=7e-2):
    N = z1.shape[0]
    z = F.normalize(
        torch.cat([z1, z2], dim=0),
        dim=1
    )
    sim = (z @ z.T) / temp
    sim.fill_diagonal_(float("-inf"))
    
    labels = torch.cat([
        torch.arange(0, N).cuda() + N,
        torch.arange(0, N).cuda()
    ])

    return F.cross_entropy(sim, labels)

@torch.no_grad()
def compute_mpcd(feats):
    feats = F.normalize(feats, dim=1)
    sim = feats @ feats.T
    b = feats.shape[0]
    mask = ~torch.eye(b, dtype=torch.bool).cuda()
    return (1. - sim[mask]).mean().item()