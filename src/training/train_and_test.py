# Helper function for training and testing.

import time
import torch

from src.utils.adv import fgsm, pgd
from src.utils.helpers import list_of_distances, make_one_hot


def _train_or_test(
    model,
    dataloader,
    optimizer=None,
    class_specific=True,
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
    is_train = optimizer is not None
    start = time.time()
    dice_sum = 0
    n_examples = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0

    for i, (image, seg_mask) in enumerate(dataloader):
        input = image.cuda()
        target = seg_mask.cuda()
        target = target.squeeze(1).long()

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            if adversarial:
                input = (
                    fgsm(model, input, target)
                    if is_train
                    else pgd(model, input, target)
                )

            # nn.Module has implemented __call__() function
            # so no need to call .forward
            output, distances = model(input)

            # compute loss
            cross_entropy = torch.nn.functional.cross_entropy(output, target)
            separation_cost = torch.tensor(0.0, device=input.device)

            if class_specific:
                B, P, Hf, Wf = distances.shape
                mask_down = torch.nn.functional.interpolate(
                    target.unsqueeze(1).float(),
                    size=(Hf, Wf),
                    mode="nearest",
                ).long().squeeze(1)
                proto_id = model.module.prototype_class_identity.cuda()

                # --- Build per‑pixel mask of “correct‑class” prototypes -------------
                matches = proto_id[:, mask_down]             # (P, B, Hf, Wf)
                proto_match = matches.permute(1, 0, 2, 3)    # (B, P, Hf, Wf)
                proto_not_match = 1 - proto_match

                # --- Cluster cost: smallest distance to ANY correct‑class prototype -
                INF = 1e6
                dist_corr = distances + (proto_not_match * INF)  # mask out wrong protos
                min_corr, _ = torch.min(dist_corr, dim=1)        # (B, Hf, Wf)
                cluster_cost = torch.mean(min_corr)

                # --- Separation cost: (negative) smallest distance to ANY wrong‑class prototype
                dist_wrong = distances + (proto_match * INF)     # mask out correct protos
                min_wrong, _ = torch.min(dist_wrong, dim=1)      # (B, Hf, Wf)
                separation_cost = -torch.mean(min_wrong)

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

            else:
                min_distance, _ = torch.min(distances, dim=1)
                cluster_cost = torch.mean(min_distance)
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
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()

        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (
                        coefs["crs_ent"] * cross_entropy
                        + coefs["clst"] * cluster_cost
                        + coefs["sep"] * separation_cost
                        + coefs["l1"] * l1
                    )
                else:
                    loss = (
                        cross_entropy
                        + 0.8 * cluster_cost
                        - 0.08 * separation_cost
                        + 1e-4 * l1
                    )
            else:
                if coefs is not None:
                    loss = (
                        coefs["crs_ent"] * cross_entropy
                        + coefs["clst"] * cluster_cost
                        + coefs["l1"] * l1
                    )
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del input
        del target
        del output
        del preds
        del l1
        del cross_entropy
        del distances

    end = time.time()

    log("\ttime: \t{0}".format(end - start))
    log("\tcross ent: \t{0}".format(total_cross_entropy / n_batches))
    log("\tcluster: \t{0}".format(total_cluster_cost / n_batches))
    if class_specific:
        log("\tseparation:\t{0}".format(total_separation_cost / n_batches))

    if adversarial:
        log("\trob: \t\t{0}%".format(dice_sum / n_batches * 100))
    else:
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
    class_specific=False,
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
        class_specific=class_specific,
        coefs=coefs,
        log=log,
        adversarial=adversarial,
    )


def test(model, dataloader, class_specific=False, log=print, adversarial=False):
    log("\ttest")
    model.eval()
    return _train_or_test(
        model=model,
        dataloader=dataloader,
        optimizer=None,
        class_specific=class_specific,
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
