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



def linear_cka(x1, x2):
    x1 = x1 - x1.mean(dim=0, keepdim=True)
    x2 = x2 - x2.mean(dim=0, keepdim=True)

    K = x1 @ x1.T
    L = x2 @ x2.T

    hsic = (K * L).sum()

    norm_x1 = torch.norm(K)
    norm_x2 = torch.norm(L)

    return hsic / (norm_x1 * norm_x2 + 1e-8)


def ensemble_divergence_loss(feats_dict, intra_weight=0.5, weights=None, weight=None, repulsion_weight=None):
    names = list(feats_dict.keys())
    total_cka = 0.
    pair_count = 0

    feats_dict = {
        k: F.normalize(v, dim=1)
        for k, v in feats_dict.items()
    }

    # inter-models divergence
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            cka_val = linear_cka(
                feats_dict[names[i]],
                feats_dict[names[j]]
            )
            total_cka += cka_val
            pair_count += 1

    mean_cka = total_cka / max(1, pair_count)

    # intra-model divergence
    intra_loss = 0.
    intra_weight = 0.1
    if intra_weight > 0:
        for feats in feats_dict.values():
            sim = feats @ feats.T
            mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
            intra_loss += (1. - sim[mask]).mean()
        intra_loss /= len(feats_dict)

    loss = mean_cka - intra_weight * intra_loss

    return loss, {
        "mean_cka": mean_cka.item(),
        "intra_loss": intra_loss.item() if intra_weight > 0 else 0.,
        "total_loss": loss.item()
    }


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