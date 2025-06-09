# Train ProtoPNets on normal CUB-200-2011 dataset.
# Used for the Head-On-Stomach Experiment.

import os
import shutil
import yaml

import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import optuna

import argparse
import re

import importlib.util

from src.utils.helpers import makedir
from src.models import seg_model
from src.training import push
from src.training import prune
from src.training import train_and_test as tnt

from src.utils import save
from src.utils.log import create_logger
import logging
from src.data.raman_dataset import create_raman_mask_dataloaders_from_ids


# Optuna objective function for hyperparameter search
def objective(trial):
    # Sample hyperparameters
    base_arch = "resnet18"
    # num_proto = trial.suggest_int("num_prototypes", 20, 120, step=20)
    num_proto = 120
    # features_lr = trial.suggest_float("features_lr", 1e-6, 6e-6, log=True)
    # features_lr = 1e-6
    # add_on_layers_lr = trial.suggest_float("add_on_layers_lr", 1e-4, 3e-4, log=True)
    # prototype_vectors_lr = trial.suggest_float("prototype_vectors_lr", 1e-4, 5e-4, log=True)
    # _joint_lr_step_size = trial.suggest_int("joint_lr_step_size", 5, 10, step=1)
    # lam_coeff = trial.suggest_float("lam_coeff", 0.2, 1.0, step=0.2)
    lam_coeff = 0.2
    # l1_coeff = trial.suggest_float("l1_coeff", 3e-4, 1e-3, log=True)
    l1_coeff = 1e-4
    # _intermediate_channels = trial.suggest_int("intermediate_channels", 128, 512, step=128)
    _intermediate_channels = 128
    # _dice_weight = trial.suggest_float("dice_weight", 0.0, 0.8, step=0.1)
    _dice_weight = 0.7
    # _label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2, step=0.05)
    _label_smoothing = 0.0

    # Override hyperparams
    global base_architecture, num_prototypes, joint_optimizer_lrs, coefs, experiment_run, intermediate_channels, dice_weight, label_smoothing
    base_architecture = base_arch
    num_prototypes = num_proto
    # joint_optimizer_lrs["features"] = features_lr
    coefs["lam"] = lam_coeff
    coefs["l1"] = l1_coeff
    experiment_run = f"optuna_trial_{trial.number}"
    intermediate_channels = _intermediate_channels
    dice_weight = _dice_weight
    label_smoothing = _label_smoothing

    # Run training and return validation accuracy
    val_acc = run_training()
    return val_acc


def run_training():
    base_architecture_type = re.match("^[a-z]*", base_architecture).group(0)

    if colab:
        model_dir = (
            "/content/PPNet/saved_models/"
            + base_architecture
            + "/"
            + experiment_run
            + "/"
        )
    else:
        model_dir = (
            "/home/"
            + username
            + "/code/this-does-not-look-like-that/out_model/saved_models/"
            + base_architecture
            + "/"
            + experiment_run
            + "/"
        )
    makedir(model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), "settings.py"), dst=model_dir)
    shutil.copy(
        src=os.path.join(
            os.getcwd(), "src/models/", base_architecture_type + "_features.py"
        ),
        dst=model_dir,
    )
    shutil.copy(
        src=os.path.join(os.getcwd(), "src/models/", "seg_model.py"), dst=model_dir
    )
    shutil.copy(
        src=os.path.join(os.getcwd(), "src/training/", "train_and_test.py"),
        dst=model_dir,
    )

    logging.basicConfig(level=logging.INFO)
    log, logclose = create_logger(log_filename=os.path.join(model_dir, "train.log"))
    img_dir = os.path.join(model_dir, "img")
    makedir(img_dir)
    # weight_matrix_filename = "outputL_weights"
    # prototype_img_filename_prefix = "prototype-img"
    # prototype_self_act_filename_prefix = "prototype-self-act"
    # proto_bound_boxes_filename_prefix = "bb"

    # config for training
    config_file = "./configs/unet_segment_newdata.yaml"
    with open(config_file, "r") as stream:
        conf = yaml.safe_load(stream)

    # all datasets

    data_transforms = transforms.Compose(
        [
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
        ]
    )

    # train set
    train_dl = create_raman_mask_dataloaders_from_ids(
        conf["data"]["train_ids"],
        conf,
        transforms=data_transforms,
        shuffle=True,
        is_train=True,
        be_fast=True,
    )
    # push set, use the same config as train set
    train_push_dl = create_raman_mask_dataloaders_from_ids(
        conf["data"]["train_ids"],
        conf,
        transforms=None,
        shuffle=False,
        is_train=True,
        be_fast=False,
    )
    # test set
    val_conf = conf.copy()
    val_conf["data"]["extra_filtering"] = []
    val_conf["training"]["stride"] = val_conf["training"]["patch_size"]
    val_dl = create_raman_mask_dataloaders_from_ids(
        val_conf["data"]["val_ids"],
        val_conf,
        shuffle=False,
        is_train=False,
        be_fast=False,
    )

    log(
        f"Running experiment with arch: {base_architecture}, num_proto: {num_prototypes}, "
        f"lam_coeff: {coefs['lam']}"
        f", intermediate_channels: {intermediate_channels}, experiment_run: {experiment_run}"
        f", dice_weight: {dice_weight}, label_smoothing: {label_smoothing}"
    )
    # we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
    log("training set size: {0}".format(len(train_dl.dataset)))
    log("push set size: {0}".format(len(train_push_dl.dataset)))
    log("test set size: {0}".format(len(val_dl.dataset)))
    log("train batch size: {0}".format(train_dl.batch_size))

    # construct the model
    prototype_shape = (num_prototypes, num_channels, 3, 3)
    ppnet = seg_model.construct_segmentation_PPNet(
        base_architecture=base_architecture,
        in_channels=24,
        pretrained=True,
        img_size=img_size,
        prototype_shape=prototype_shape,
        num_classes=num_classes,
        prototype_activation_function=prototype_activation_function,
        intermediate_channels=intermediate_channels,
    )
    # if prototype_activation_function == 'linear':
    #    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)

    # define optimizer
    joint_optimizer_specs = [
        {
            "params": ppnet.features.parameters(),
            "lr": joint_optimizer_lrs["features"],
            "weight_decay": 1e-3,
        },  # bias are now also being regularized
        {
            "params": ppnet.add_on_layers.parameters(),
            "lr": joint_optimizer_lrs["add_on_layers"],
            "weight_decay": 1e-3,
        },
        {
            "params": ppnet.prototype_vectors,
            "lr": joint_optimizer_lrs["prototype_vectors"],
        },
        {
            "params": ppnet.aspp.parameters(),
            "lr": warm_optimizer_lrs["add_on_layers"],
            "weight_decay": 1e-3,
        },
    ]
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        joint_optimizer, step_size=joint_lr_step_size, gamma=0.5
    )

    warm_optimizer_specs = [
        {
            # allow last feature layer training
            "params": ppnet.features.layer4.parameters(),
            "lr": warm_optimizer_lrs["features"],
            "weight_decay": 1e-3,
        },
        {
            "params": ppnet.add_on_layers.parameters(),
            "lr": warm_optimizer_lrs["add_on_layers"],
            "weight_decay": 1e-3,
        },
        {
            "params": ppnet.prototype_vectors,
            "lr": warm_optimizer_lrs["prototype_vectors"],
        },
        {
            "params": ppnet.aspp.parameters(),
            "lr": warm_optimizer_lrs["add_on_layers"],
            "weight_decay": 1e-3,
        },
        {
            "params": ppnet.decoder.parameters(),
            "lr": warm_optimizer_lrs["add_on_layers"],
            "weight_decay": 1e-3,
        },
    ]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

    last_layer_optimizer_specs = [
        {"params": ppnet.last_layer.parameters(), "lr": last_layer_optimizer_lr}
    ]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    # train the model
    log("start training")

    target_accu = 0.65
    accu = 0.0
    for epoch in range(num_train_epochs):
        log("epoch: \t{0}".format(epoch))

        if epoch < num_warm_epochs:
            tnt.warm_only(model=ppnet_multi, log=log)
            _ = tnt.train(
                model=ppnet_multi,
                dataloader=train_dl,
                optimizer=warm_optimizer,
                coefs=coefs,
                log=log,
                dice_weight=dice_weight,
                label_smoothing=label_smoothing,
            )
        else:
            tnt.joint(model=ppnet_multi, log=log)
            _ = tnt.train(
                model=ppnet_multi,
                dataloader=train_dl,
                optimizer=joint_optimizer,
                coefs=coefs,
                log=log,
                dice_weight=dice_weight,
                label_smoothing=label_smoothing,
            )
            joint_lr_scheduler.step()

        accu = tnt.test(
            model=ppnet_multi,
            dataloader=val_dl,
            log=log,
        )
        save.save_model_w_condition(
            model=ppnet,
            model_dir=model_dir,
            model_name=str(epoch) + "nopush",
            accu=accu,
            target_accu=target_accu,
            log=log,
        )

        if epoch >= push_start and epoch in push_epochs:
            # TODO: need to get good performance without pushing first
            push.push_prototypes(
                train_push_dl,  # pytorch dataloader (must be unnormalized in [0,1])
                prototype_network_parallel=ppnet_multi,  # pytorch network with prototype_vectors
                log=log,
            )
            accu = tnt.test(
                model=ppnet_multi,
                dataloader=val_dl,
                log=log,
            )
            save.save_model_w_condition(
                model=ppnet,
                model_dir=model_dir,
                model_name=str(epoch) + "push",
                accu=accu,
                target_accu=target_accu,
                log=log,
            )

            if prototype_activation_function != "linear":
                tnt.last_only(model=ppnet_multi, log=log)
                # We don't need to train the last layer for *that* long
                for i in range(1):
                    log("iteration: \t{0}".format(i))
                    _ = tnt.train(
                        model=ppnet_multi,
                        dataloader=train_dl,
                        optimizer=last_layer_optimizer,
                        coefs=coefs,
                        log=log,
                        dice_weight=dice_weight,
                        label_smoothing=label_smoothing,
                    )
                    accu = tnt.test(
                        model=ppnet_multi,
                        dataloader=val_dl,
                        log=log,
                    )
                    save.save_model_w_condition(
                        model=ppnet,
                        model_dir=model_dir,
                        model_name=str(epoch) + "_" + str(i) + "push",
                        accu=accu,
                        target_accu=target_accu,
                        log=log,
                    )

    logclose()
    return accu


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-gpuid", nargs=1, type=str, default="0"
    )  # python3 main.py -gpuid=0,1,2,3
    parser.add_argument(
        "--hyperparams",
        type=str,
        default="settings.py",
        help="Path to the hyperparameters script (Python file)",
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid[0]
    print(os.environ["CUDA_VISIBLE_DEVICES"])

    # Load hyperparameters from external script
    spec = importlib.util.spec_from_file_location("hyperparams", args.hyperparams)
    hyperparams = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hyperparams)

    # Override settings values with hyperparams
    base_architecture = hyperparams.base_architecture
    img_size = hyperparams.img_size
    num_prototypes = hyperparams.num_prototypes
    num_channels = hyperparams.num_channels
    num_classes = hyperparams.num_classes
    prototype_activation_function = hyperparams.prototype_activation_function
    intermediate_channels = hyperparams.intermediate_channels
    experiment_run = hyperparams.experiment_run
    colab = hyperparams.colab
    username = hyperparams.username
    joint_optimizer_lrs = hyperparams.joint_optimizer_lrs
    joint_lr_step_size = hyperparams.joint_lr_step_size
    warm_optimizer_lrs = hyperparams.warm_optimizer_lrs
    last_layer_optimizer_lr = hyperparams.last_layer_optimizer_lr
    coefs = hyperparams.coefs
    num_train_epochs = hyperparams.num_train_epochs
    num_warm_epochs = hyperparams.num_warm_epochs
    push_start = hyperparams.push_start
    push_epochs = hyperparams.push_epochs

    # Run Optuna hyperparameter search
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    study.trials_dataframe().to_csv(
        os.path.join(os.getcwd(), "optuna_trials.csv"), index=False
    )

    print("Best trial:")
    print(f"  Value: {study.best_value}")
    print("  Params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
