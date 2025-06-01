# Helper function for training and testing.

import time
import torch
import torch.nn.functional as F
from itertools import combinations

from src.utils.adv import fgsm, pgd
from src.utils.helpers import list_of_distances, make_one_hot


def downsampled(tensor, ref_tensor):
    return F.interpolate(
        tensor,
        size=ref_tensor.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )


def _jeffrey_divergence(
    u: torch.Tensor, v: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    u = u + eps
    v = v + eps

    # KL(u‖v)
    kl_uv = torch.sum(u * (torch.log(u) - torch.log(v)))
    # KL(v‖u)
    kl_vu = torch.sum(v * (torch.log(v) - torch.log(u)))

    return 0.5 * (kl_uv + kl_vu)


def _jeffrey_similarity(vectors: torch.Tensor) -> torch.Tensor:
    k = vectors.size(0)
    if k < 2:
        return torch.zeros((), device=vectors.device)

    sims = []
    for i, j in combinations(range(k), r=2):
        d_j = _jeffrey_divergence(vectors[i], vectors[j])
        sims.append(torch.exp(-d_j))

    return torch.stack(sims).mean()


def prototype_diversity_loss(
    feature_map: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    proto_classes: torch.Tensor,
    lambda_j: float = 0.25,
    temperature: float = 1.0,
) -> torch.Tensor:
    if lambda_j == 0.0:
        return torch.zeros((), device=feature_map.device)

    bsz, d, h, w = feature_map.shape
    n_classes = int(proto_classes.max().item() + 1)

    # Move channels last for easier indexing.
    feat_flat = feature_map.permute(0, 2, 3, 1).reshape(bsz, h * w, d)  # (B, N, D)
    lab_flat = labels.view(bsz, -1)  # (B, N)

    loss_acc = 0.0
    for b in range(bsz):
        for c in range(n_classes):
            # mask for class *c* in sample *b*
            idx = (lab_flat[b] == c).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue  # class absent in this sample

            prot_idx = (proto_classes == c).nonzero(as_tuple=True)[0]
            if prot_idx.numel() < 2:
                continue  # Need at least two prototypes to compare.

            # Feature vectors for class‑c pixels
            z_c = feat_flat[b, idx]  # (N_c, D)

            # Build probability vectors v(Z, p) for every prototype p∈P_c
            v_list = []
            for p_id in prot_idx:
                p = prototypes[p_id]  # (D,)
                # Euclidean distances to all selected pixels
                dist = torch.norm(z_c - p, dim=1) / temperature  # (N_c,)
                v = F.softmax(dist, dim=0)  # (N_c,)
                v_list.append(v)

            v_stack = torch.stack(v_list)  # (K, N_c)
            sj = _jeffrey_similarity(v_stack)
            loss_acc = loss_acc + sj

    loss = loss_acc / n_classes
    return loss * lambda_j


# --- Activation-overlap loss for prototype diversity ---------------------------
def activation_overlap_loss(
    similarity_maps: torch.Tensor,
    proto_classes: torch.Tensor,
    lambda_div: float = 0.25,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Prototype–activation overlap loss.
    Encourages prototypes of the same class to fire on disjoint regions.

    Args:
        similarity_maps: tensor of shape (B, P, H, W) with similarity
                         scores for every prototype.
        proto_classes:   tensor of shape (P,) that maps each prototype
                         to its semantic class index.
        lambda_div:      scalar weight applied to the loss.
        eps:             numerical stability term.
    """
    if lambda_div == 0.0:
        return torch.zeros((), device=similarity_maps.device)

    # Keep only positive evidence.
    acts = similarity_maps.clamp(min=0)          # (B, P, H, W)
    B, P, H, W = acts.shape
    S = H * W
    acts = acts.view(B, P, S)                    # (B, P, S)

    # L1‑normalise each activation map so that it sums to 1.
    acts = acts / (acts.sum(dim=2, keepdim=True) + eps)

    n_classes = int(proto_classes.max().item() + 1)
    loss_acc = 0.0
    for c in range(n_classes):
        prot_idx = (proto_classes == c).nonzero(as_tuple=True)[0]
        k = prot_idx.numel()
        if k < 2:
            continue

        a_c = acts[:, prot_idx]                  # (B, k, S)
        # Pairwise inner products <Â_i, Â_j>
        sim = torch.einsum("bks,bls->bkl", a_c, a_c)  # (B, k, k)
        # Upper‑triangular without diagonal
        sim_pairs = sim[:, torch.triu_indices(k, k, offset=1)[0],
                           torch.triu_indices(k, k, offset=1)[1]]
        loss_acc += sim_pairs.mean()

    return (loss_acc / n_classes) * lambda_div


def _train_or_test(
    model,
    dataloader,
    optimizer=None,
    use_l1_mask=True,
    coefs=None,
    log=print,
    adversarial=False,
):
    """
    Train or test.
    Args:
        model: the multi-gpu model.
        dataloader: train or test dataloader.
        optimizer: if None, will be test evaluation.
        adversarial: if True, perform fast FGSM adversarial training or PGD evaluation.
    Returns:
        accuracy of the given model.
    """
    if adversarial:
        raise NotImplementedError(
            "Adversarial training is not implemented yet."
        )

    is_train = optimizer is not None
    start = time.time()
    dice_sum = 0
    n_examples = 0
    n_batches = 0
    total_cross_entropy = 0
    total_diversity_loss = 0

    for i, (image, seg_mask) in enumerate(dataloader):
        input = image.cuda()
        target = seg_mask.cuda()
        target = target.squeeze(1).long()

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            output, score_maps = model(input)

            # compute loss on latent score maps and downsampled target mask
            downsampled_target = downsampled(
                target.unsqueeze(0).float(), ref_tensor=score_maps
            ).squeeze(0).long()
            cross_entropy = torch.nn.functional.cross_entropy(
                score_maps, downsampled_target,
            )

            # cross_entropy = torch.nn.functional.cross_entropy(output, target)

            # --- activation‑overlap diversity loss --------------------
            overlap_loss = activation_overlap_loss(
                similarity_maps=model.module.similarity_maps,
                proto_classes=model.module.prototype_class_identity.argmax(dim=1),
            )
            model.module.similarity_maps = None
            model.module.feature_map = None
            # L1 regulariser on incorrect connections
            if use_l1_mask:
                # prototype_class_identity: (num_prototypes, num_classes)
                # last_layer.weight:       (num_classes, num_prototypes, 1, 1)
                l1_mask = (
                    1
                    - torch.t(model.module.prototype_class_identity)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                    .cuda()
                )
                l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
            else:
                l1 = model.module.last_layer.weight.norm(p=1)

            # evaluation statistics
            # Unified accuracy: pixel-level for segmentation, sample-level for classification
            preds = output.argmax(dim=1)
            dice_sum += (
                2
                * torch.sum(preds * target)
                / (torch.sum(preds) + torch.sum(target) + 1e-6)
            ).item()
            n_examples += target.size(0)

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_diversity_loss += overlap_loss.item()

        # compute gradient and do SGD step
        if is_train:
            if coefs is not None:
                loss = (
                    coefs["crs_ent"] * cross_entropy
                    + coefs["lam"] * overlap_loss
                    + coefs["l1"] * l1
                )
            else:
                loss = cross_entropy + 0.25 * overlap_loss + 1e-4 * l1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del input
        del target
        del output
        del preds
        del l1
        del cross_entropy
        del overlap_loss
        del score_maps

    end = time.time()

    log("\ttime: \t{0}".format(end - start))
    log("\tcross ent: \t{0}".format(total_cross_entropy / n_batches))
    log("\tdiversity: \t{0}".format(total_diversity_loss / n_batches))

    log("\taccu: \t\t{0}%".format(dice_sum / n_batches * 100))
    log("\tl1: \t\t{0}".format(model.module.last_layer.weight.norm(p=1).item()))
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log("\tp dist pair: \t{0}".format(p_avg_pair_dist.item()))

    return dice_sum / n_batches


def train(
    model,
    dataloader,
    optimizer,
    coefs=None,
    log=print,
    adversarial=False,
):
    assert optimizer is not None

    log("\ttrain")
    model.train()
    return _train_or_test(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        coefs=coefs,
        log=log,
        adversarial=adversarial,
    )


def test(model, dataloader, log=print, adversarial=False):
    log("\ttest")
    model.eval()
    return _train_or_test(
        model=model,
        dataloader=dataloader,
        optimizer=None,
        log=log,
        adversarial=adversarial,
    )


def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log("\tlast layer")


def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log("\twarm")


def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log("\tjoint")
