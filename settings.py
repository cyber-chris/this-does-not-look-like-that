# Configuration for training ProtoPNets.

import os
import getpass


username = getpass.getuser()

base_architecture = "resnet18"
img_size = 512
num_channels = 64

num_prototypes = 20
# prototype_shape = (20, num_channels, 1, 1)
num_classes = 2
prototype_activation_function = "log"
intermediate_channels = 512

experiment_run = "011"

JPEG_QUALITY = 20


if "COLAB_GPU" in os.environ:
    data_path = "/content/PPNet/datasets/cub200_cropped/"
    pretrained_model_dir = "/content/PPNet/pretrained_models/"
    colab = True
else:
    # data_path = "/scratch/PPNet/datasets/cub200_cropped/"
    # pretrained_model_dir = "/cluster/scratch/{}/PPNet/pretrained_models/".format(
    #     username
    # )
    data_path = "/home/ct678/code/this-does-not-look-like-that/out_model/datasets/cub200_cropped/"
    pretrained_model_dir = "/home/ct678/code/this-does-not-look-like-that/out_model/"
    colab = False
    colab = False

train_dir = data_path + "train_cropped_augmented/"
test_dir = data_path + "test_cropped/"
train_push_dir = data_path + "train_cropped/"

joint_optimizer_lrs = {
    "features": 1e-5,
    "add_on_layers": 1e-4,
    "prototype_vectors": 1e-4,
}
joint_lr_step_size = 5

warm_optimizer_lrs = {
    "features": 1e-5,
    "add_on_layers": 5e-4, 
    "prototype_vectors": 5e-4
}

last_layer_optimizer_lr = 1e-5

coefs = {
    "crs_ent": 1,
    "lam": 0.25,
    "l1": 1e-4,
}

num_train_epochs = 15
num_warm_epochs = 15

push_start = 14
push_epochs = [14]


# configuration for adversarial training
epsilon = 8.0
alpha = 10.0
pgd_alpha = 2.0
pgd_attack_iters = 10
