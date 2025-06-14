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
        mode="nearest",
    )


def _jeffrey_divergence(
    u: torch.Tensor, v: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    u = u + eps
    v = v + eps

    kl_uv = torch.sum(u * (torch.log(u) - torch.log(v)))
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

    feat_flat = feature_map.permute(0, 2, 3, 1).reshape(bsz, h * w, d)
    lab_flat = labels.view(bsz, -1)

    loss_acc = 0.0
    for b in range(bsz):
        for c in range(n_classes):
            idx = (lab_flat[b] == c).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue

            prot_idx = (proto_classes == c).nonzero(as_tuple=True)[0]
            if prot_idx.numel() < 2:
                continue

            z_c = feat_flat[b, idx]

            v_list = []
            for p_id in prot_idx:
                p = prototypes[p_id]
                dist = torch.norm(z_c - p, dim=1) / temperature
                v = F.softmax(dist, dim=0)
                v_list.append(v)

            v_stack = torch.stack(v_list)
            sj = _jeffrey_similarity(v_stack)
            loss_acc = loss_acc + sj

    loss = loss_acc / n_classes
    return loss * lambda_j


def activation_overlap_loss(
    similarity_maps: torch.Tensor,
    proto_classes: torch.Tensor,
    lambda_div: float = 0.25,
    eps: float = 1e-6,
) -> torch.Tensor:
    if lambda_div == 0.0:
        return torch.zeros((), device=similarity_maps.device)

    # Keep only positive evidence.
    acts = similarity_maps.clamp(min=0)
    B, P, H, W = acts.shape
    S = H * W
    acts = acts.view(B, P, S)
    acts = acts / (acts.sum(dim=2, keepdim=True) + eps)

    n_classes = int(proto_classes.max().item() + 1)
    loss_acc = 0.0
    for c in range(n_classes):
        prot_idx = (proto_classes == c).nonzero(as_tuple=True)[0]
        k = prot_idx.numel()
        if k < 2:
            continue

        a_c = acts[:, prot_idx]
        sim = torch.einsum("bks,bls->bkl", a_c, a_c)
        sim_pairs = sim[:, torch.triu_indices(k, k, offset=1)[0],
                           torch.triu_indices(k, k, offset=1)[1]]
        loss_acc += sim_pairs.mean()

    return (loss_acc / n_classes) * lambda_div


# Soft multi-class Dice loss
def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.contiguous().view(pred.size(0), pred.size(1), -1)
    target = target.contiguous().view(target.size(0), target.size(1), -1)

    intersection = 2.0 * (pred * target).sum(-1)
    denominator = pred.sum(-1) + target.sum(-1)
    dice = (intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def _train_or_test(
    model,
    dataloader,
    optimizer=None,
    use_l1_mask=True,
    coefs=None,
    log=print,
    adversarial=False,
    dice_weight=0.5,
    label_smoothing=0.1,
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
    n_examples = 0
    n_batches = 0
    total_cross_entropy = 0
    total_diversity_loss = 0
    total_dice_loss = 0
    accu_sum = 0

    for i, (image, seg_mask, _) in enumerate(dataloader):
        batch_start_time = time.time()
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
                target.unsqueeze(1).float(), ref_tensor=score_maps
            ).squeeze(1).long()
            cross_entropy = torch.nn.functional.cross_entropy(
                score_maps, downsampled_target, label_smoothing=label_smoothing
            )

            probs = F.softmax(score_maps, dim=1)
            target_one_hot = F.one_hot(
                downsampled_target, num_classes=probs.size(1)
            ).permute(0, 3, 1, 2).float().cuda()
            dice_loss_val = dice_loss(probs, target_one_hot)
            # target_one_hot = F.one_hot(
            #     target, num_classes=output.size(1)
            # ).permute(0, 3, 1, 2).float().cuda()
            # probs = F.softmax(output, dim=1)
            # dice_loss_val = dice_loss(probs, target_one_hot)

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
            # pixel accuracy
            preds = output.argmax(dim=1)
            accu_sum += torch.sum(preds == target).item() / target.numel()
            n_examples += target.size(0)

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_diversity_loss += overlap_loss.item()
            total_dice_loss += dice_loss_val.item()

        # compute gradient and do SGD step
        if is_train:
            if coefs is not None:
                loss = (
                    # CE probably better than dice when training on downsampled space
                    1.0 * cross_entropy
                    + dice_weight * dice_loss_val
                    + coefs["lam"] * overlap_loss
                    + coefs["l1"] * l1
                )
            else:
                # loss = cross_entropy + 0.25 * overlap_loss + 1e-4 * l1
                raise Exception("coefs must be provided for training")
                loss = cross_entropy + dice_loss_val + 0.25 * overlap_loss + 1e-4 * l1

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
        del dice_loss_val
        batch_end_time = time.time()
        # log(f"took {batch_end_time - batch_start_time:.2f} seconds ")

    end = time.time()

    log("\ttime: \t{0}".format(end - start))
    log("\tcross ent: \t{0}".format(total_cross_entropy / n_batches))
    log("\tdiversity: \t{0}".format(total_diversity_loss / n_batches))
    log("\tdice loss: \t{0}".format(total_dice_loss / n_batches))

    log("\taccu: \t\t{0}%".format(accu_sum / n_batches * 100))
    log("\tl1: \t\t{0}".format(model.module.last_layer.weight.norm(p=1).item()))
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log("\tp dist pair: \t{0}".format(p_avg_pair_dist.item()))

    return accu_sum / n_batches


def train(
    model,
    dataloader,
    optimizer,
    coefs=None,
    log=print,
    adversarial=False,
    dice_weight=0.5,
    label_smoothing=0.1,
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
        dice_weight=dice_weight,
        label_smoothing=label_smoothing,
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
    # allow training of last feature layer
    for p in model.module.features.layer4.parameters():
        p.requires_grad = True
    for p in model.module.aspp.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.decoder.parameters():
        p.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log("\twarm")


def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.aspp.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.decoder.parameters():
        p.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log("\tjoint")
