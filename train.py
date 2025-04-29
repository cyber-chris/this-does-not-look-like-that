# Train ProtoPNets on normal CUB-200-2011 dataset.
# Used for the Head-On-Stomach Experiment.

import os
import shutil
import yaml

import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import re

from src.utils.helpers import makedir
from src.models import seg_model
from src.training import push
from src.training import prune
from src.training import train_and_test as tnt

from src.utils import save
from src.utils.log import create_logger
import logging
from src.data.raman_dataset import create_raman_mask_dataloaders_from_ids


parser = argparse.ArgumentParser()
parser.add_argument(
    "-gpuid", nargs=1, type=str, default="0"
)  # python3 main.py -gpuid=0,1,2,3
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid[0]
print(os.environ["CUDA_VISIBLE_DEVICES"])


# book keeping namings and code
from settings import (
    base_architecture,
    img_size,
    prototype_shape,
    num_classes,
    prototype_activation_function,
    add_on_layers_type,
    experiment_run,
)
from settings import colab, username

base_architecture_type = re.match("^[a-z]*", base_architecture).group(0)

if colab:
    model_dir = (
        "/content/PPNet/saved_models/" + base_architecture + "/" + experiment_run + "/"
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
shutil.copy(src=os.path.join(os.getcwd(), "src/models/", "model.py"), dst=model_dir)
shutil.copy(
    src=os.path.join(os.getcwd(), "src/training/", "train_and_test.py"), dst=model_dir
)

logging.basicConfig(level=logging.INFO)
log, logclose = create_logger(log_filename=os.path.join(model_dir, "train.log"))
img_dir = os.path.join(model_dir, "img")
makedir(img_dir)
weight_matrix_filename = "outputL_weights"
prototype_img_filename_prefix = "prototype-img"
prototype_self_act_filename_prefix = "prototype-self-act"
proto_bound_boxes_filename_prefix = "bb"


# load the data
from settings import (
    train_dir,
    test_dir,
    train_push_dir,
    train_batch_size,
    test_batch_size,
    train_push_batch_size,
)


# config for training
config_file = "./configs/unet_segment_newdata.yaml"
with open(config_file, "r") as stream:
    conf = yaml.safe_load(stream)

# all datasets

# train set
train_dl = create_raman_mask_dataloaders_from_ids(
    conf["data"]["train_ids"],
    conf,
    transforms=None,
    shuffle=True,
    is_train=True,
)
# push set, use the same config as train set
train_push_dl = create_raman_mask_dataloaders_from_ids(
    conf["data"]["train_ids"],
    conf,
    transforms=None,
    shuffle=False,
    is_train=True,
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
)

# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
log("training set size: {0}".format(len(train_dl.dataset)))
log("push set size: {0}".format(len(train_push_dl.dataset)))
log("test set size: {0}".format(len(val_dl.dataset)))
log("batch size: {0}".format(train_batch_size))


# construct the model
ppnet = seg_model.construct_segmentation_PPNet(
    base_architecture=base_architecture,
    in_channels=24,
    pretrained=True,
    img_size=img_size,
    prototype_shape=prototype_shape,
    num_classes=num_classes,
    prototype_activation_function=prototype_activation_function,
    add_on_layers_type=add_on_layers_type,
)
# if prototype_activation_function == 'linear':
#    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
class_specific = True


# define optimizer
from settings import joint_optimizer_lrs, joint_lr_step_size

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
    {"params": ppnet.prototype_vectors, "lr": joint_optimizer_lrs["prototype_vectors"]},
]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(
    joint_optimizer, step_size=joint_lr_step_size, gamma=0.1
)

from settings import warm_optimizer_lrs

warm_optimizer_specs = [
    {
        "params": ppnet.add_on_layers.parameters(),
        "lr": warm_optimizer_lrs["add_on_layers"],
        "weight_decay": 1e-3,
    },
    {"params": ppnet.prototype_vectors, "lr": warm_optimizer_lrs["prototype_vectors"]},
]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

from settings import last_layer_optimizer_lr

last_layer_optimizer_specs = [
    {"params": ppnet.last_layer.parameters(), "lr": last_layer_optimizer_lr}
]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)


# weighting of different training losses
from settings import coefs

# number of training epochs, number of warm epochs, push start epoch, push epochs
from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs

# train the model
log("start training")
import copy

for epoch in range(num_train_epochs):
    log("epoch: \t{0}".format(epoch))

    if epoch < num_warm_epochs:
        tnt.warm_only(model=ppnet_multi, log=log)
        _ = tnt.train(
            model=ppnet_multi,
            dataloader=train_dl,
            optimizer=warm_optimizer,
            class_specific=class_specific,
            coefs=coefs,
            log=log,
        )
    else:
        tnt.joint(model=ppnet_multi, log=log)
        _ = tnt.train(
            model=ppnet_multi,
            dataloader=train_dl,
            optimizer=joint_optimizer,
            class_specific=class_specific,
            coefs=coefs,
            log=log,
        )
        joint_lr_scheduler.step()

    accu = tnt.test(
        model=ppnet_multi,
        dataloader=val_dl,
        class_specific=class_specific,
        log=log,
    )
    save.save_model_w_condition(
        model=ppnet,
        model_dir=model_dir,
        model_name=str(epoch) + "nopush",
        accu=accu,
        target_accu=0.70,
        log=log,
    )

    if epoch >= push_start and epoch in push_epochs:
        push.push_prototypes(
            train_push_dl,  # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi,  # pytorch network with prototype_vectors
            class_specific=class_specific,
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir,  # if not None, prototypes will be saved here
            epoch_number=epoch,  # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log,
        )
        accu = tnt.test(
            model=ppnet_multi,
            dataloader=val_dl,
            class_specific=class_specific,
            log=log,
        )
        save.save_model_w_condition(
            model=ppnet,
            model_dir=model_dir,
            model_name=str(epoch) + "push",
            accu=accu,
            target_accu=0.70,
            log=log,
        )

        if prototype_activation_function != "linear":
            tnt.last_only(model=ppnet_multi, log=log)
            # We don't need to train the last layer for *that* long
            for i in range(5):
                log("iteration: \t{0}".format(i))
                _ = tnt.train(
                    model=ppnet_multi,
                    dataloader=train_dl,
                    optimizer=last_layer_optimizer,
                    class_specific=class_specific,
                    coefs=coefs,
                    log=log,
                )
                accu = tnt.test(
                    model=ppnet_multi,
                    dataloader=val_dl,
                    class_specific=class_specific,
                    log=log,
                )
                save.save_model_w_condition(
                    model=ppnet,
                    model_dir=model_dir,
                    model_name=str(epoch) + "_" + str(i) + "push",
                    accu=accu,
                    target_accu=0.70,
                    log=log,
                )


logclose()
