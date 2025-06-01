# ProtoPNet model definition.

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math

import sys

sys.path.insert(0, "src/models/")

from resnet_features import (
    resnet18_features,
    resnet34_features,
    resnet50_features,
    resnet101_features,
    resnet152_features,
)
from densenet_features import (
    densenet121_features,
    densenet161_features,
    densenet169_features,
    densenet201_features,
)
from vgg_features import (
    vgg11_features,
    vgg11_bn_features,
    vgg13_features,
    vgg13_bn_features,
)
from vgg_features import (
    vgg16_features,
    vgg16_bn_features,
    vgg19_features,
    vgg19_bn_features,
)

from src.utils.receptive_field import compute_proto_layer_rf_info_v2


base_architecture_to_features = {
    "resnet18": resnet18_features,
    "resnet34": resnet34_features,
    "resnet50": resnet50_features,
    "resnet101": resnet101_features,
    "resnet152": resnet152_features,
    "densenet121": densenet121_features,
    "densenet161": densenet161_features,
    "densenet169": densenet169_features,
    "densenet201": densenet201_features,
    "vgg11": vgg11_features,
    "vgg11_bn": vgg11_bn_features,
    "vgg13": vgg13_features,
    "vgg13_bn": vgg13_bn_features,
    "vgg16": vgg16_features,
    "vgg16_bn": vgg16_bn_features,
    "vgg19": vgg19_features,
    "vgg19_bn": vgg19_bn_features,
}

# ------------------------------------------------------------------
# Atrous Spatial Pyramid Pooling (ASPP) block
# Inspired by DeepLab‑v3: captures multi‑scale context with dilated convs
# ------------------------------------------------------------------
class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, atrous_rates=(6, 12, 18, 24)):
        super().__init__()
        modules = []
        # 1×1 conv branch
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        )
        # Dilated conv branches
        for rate in atrous_rates:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=rate,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                )
            )
        # Image‑level pooling branch
        modules.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        )
        self.branches = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * len(modules), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[2:]
        feats = []
        for idx, branch in enumerate(self.branches):
            y = branch(x)
            # Upsample the global‐pool branch
            if idx == len(self.branches) - 1:
                y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
            feats.append(y)
        x = torch.cat(feats, dim=1)
        return self.project(x)


class SegmentationPPNet(nn.Module):
    def __init__(
        self,
        features,
        img_size,
        prototype_shape,
        proto_layer_rf_info,
        num_classes,
        init_weights,
        prototype_activation_function,
        add_on_layers_type,
    ):
        """
        Construct a ProtoPNet.
        """
        super(SegmentationPPNet, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.epsilon = 1e-4

        # prototype_activation_function could be 'log', 'linear',
        # or a generic function that converts distance to similarity score
        self.prototype_activation_function = prototype_activation_function

        """
        Here we are initializing the class identities of the prototypes
        Without domain specific knowledge we allocate the same number of
        prototypes for each class
        """
        assert self.num_prototypes % self.num_classes == 0
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(
            self.num_prototypes, self.num_classes
        )

        num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

        self.proto_layer_rf_info = proto_layer_rf_info

        # this has to be named features to allow the precise loading
        self.features = features

        # we store the last feature map generated
        self.feature_map = None
        self.similarity_maps = None

        features_name = str(self.features).upper()
        if features_name.startswith("VGG") or features_name.startswith("RES"):
            first_add_on_layer_in_channels = [
                i for i in features.modules() if isinstance(i, nn.Conv2d)
            ][-1].out_channels
        elif features_name.startswith("DENSE"):
            first_add_on_layer_in_channels = [
                i for i in features.modules() if isinstance(i, nn.BatchNorm2d)
            ][-1].num_features
        else:
            raise Exception("other base base_architecture NOT implemented")

        # Multi‑scale context encoder
        # self.aspp = ASPP(
        #     first_add_on_layer_in_channels,
        #     first_add_on_layer_in_channels,
        #     atrous_rates=(1, 3, 6),
        # )
        self.aspp = nn.Identity()

        intermediate_channels = first_add_on_layer_in_channels
        self.add_on_layers = nn.Sequential(
            nn.Conv2d(first_add_on_layer_in_channels, intermediate_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(),
            # Optional: another 3x3 convolution
            nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(),
            # Final 1x1 convolution to project to prototype depth
            nn.Conv2d(intermediate_channels, self.prototype_shape[1], kernel_size=1, bias=False),
            nn.Sigmoid() # If Sigmoid is desired before prototype distances
        )

        # Scale‑aware prototype initialization (Xavier uniform)
        torch.manual_seed(42)  # keep determinism
        fan_in = self.prototype_shape[1] * self.prototype_shape[2] * self.prototype_shape[3]
        bound = 1.0 / math.sqrt(fan_in)
        init_proto = torch.empty(self.prototype_shape).uniform_(-bound, bound)
        self.prototype_vectors = nn.Parameter(init_proto, requires_grad=True)

        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)

        # segmentation task
        self.last_layer = nn.Conv2d(
            in_channels=self.num_prototypes,
            out_channels=self.num_classes,
            kernel_size=1,
            bias=False,
        )

        if init_weights:
            self._initialize_weights()

    def conv_features(self, x):
        """
        the feature input to prototype layer
        """
        x = self.features(x)
        x = self.aspp(x)
        x = self.add_on_layers(x)
        return x

    @staticmethod
    def _weighted_l2_convolution(input, filter, weights):
        """
        input of shape N * c * h * w
        filter of shape P * c * h1 * w1
        weight of shape P * c * h1 * w1
        """
        input2 = input**2
        input_patch_weighted_norm2 = F.conv2d(input=input2, weight=weights)

        filter2 = filter**2
        weighted_filter2 = filter2 * weights
        filter_weighted_norm2 = torch.sum(weighted_filter2, dim=(1, 2, 3))
        filter_weighted_norm2_reshape = filter_weighted_norm2.view(-1, 1, 1)

        weighted_filter = filter * weights
        weighted_inner_product = F.conv2d(input=input, weight=weighted_filter)

        # use broadcast
        intermediate_result = (
            -2 * weighted_inner_product + filter_weighted_norm2_reshape
        )
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(input_patch_weighted_norm2 + intermediate_result)

        return distances

    def _l2_convolution(self, x):
        """
        apply self.prototype_vectors as l2-convolution filters on input x
        """
        x2 = x**2
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)

        p2 = self.prototype_vectors**2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=self.prototype_vectors)
        intermediate_result = -2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances

    def prototype_distances(self, x):
        """
        x is the raw input
        """
        conv_features = self.conv_features(x)
        self.feature_map = conv_features
        distances = self._l2_convolution(conv_features)
        return distances

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == "log":
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == "linear":
            return -distances
        else:
            return self.prototype_activation_function(distances)

    def forward(self, x):
        # Modified forward pass for segmentation
        distances = self.prototype_distances(x)
        similarity_maps = self.distance_2_similarity(distances)
        self.similarity_maps = similarity_maps

        score_maps = self.last_layer(similarity_maps)
        logits_full_upsampled = F.interpolate(
            score_maps,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        return logits_full_upsampled, score_maps

    def push_forward(self, x):
        """this method is needed for the pushing operation"""
        conv_output = self.conv_features(x)
        distances = self._l2_convolution(conv_output)
        return conv_output, distances

    def prune_prototypes(self, prototypes_to_prune):
        """
        prototypes_to_prune: a list of indices each in
        [0, current number of prototypes - 1] that indicates the prototypes to
        be removed
        """
        prototypes_to_keep = list(
            set(range(self.num_prototypes)) - set(prototypes_to_prune)
        )

        self.prototype_vectors = nn.Parameter(
            self.prototype_vectors.data[prototypes_to_keep, ...], requires_grad=True
        )

        self.prototype_shape = list(self.prototype_vectors.size())
        self.num_prototypes = self.prototype_shape[0]

        # changing self.last_layer in place
        # changing in_features and out_features make sure the numbers are consistent
        self.last_layer.in_channels = self.num_prototypes
        self.last_layer.out_channels = self.num_classes
        self.last_layer.weight.data = self.last_layer.weight.data[:, prototypes_to_keep]

        # self.ones is nn.Parameter
        self.ones = nn.Parameter(
            self.ones.data[prototypes_to_keep, ...], requires_grad=False
        )
        # self.prototype_class_identity is torch tensor
        # so it does not need .data access for value update
        self.prototype_class_identity = self.prototype_class_identity[
            prototypes_to_keep, :
        ]

    def __repr__(self):
        # PPNet(self, features, img_size, prototype_shape,
        # proto_layer_rf_info, num_classes, init_weights=True):
        rep = (
            "PPNet(\n"
            "\tfeatures: {},\n"
            "\timg_size: {},\n"
            "\tprototype_shape: {},\n"
            "\tproto_layer_rf_info: {},\n"
            "\tnum_classes: {},\n"
            "\tepsilon: {}\n"
            ")"
        )

        return rep.format(
            self.features,
            self.img_size,
            self.prototype_shape,
            self.proto_layer_rf_info,
            self.num_classes,
            self.epsilon,
        )

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        """
        the incorrect strength will be actual strength if -0.5 then input -0.5
        """
        # final layer weight shape is (num_classes, num_prototypes, 1, 1)
        positive_one_weights_locations = (
            torch.t(self.prototype_class_identity).unsqueeze(-1).unsqueeze(-1)
        )
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations
        )

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)


def construct_segmentation_PPNet(
    base_architecture,
    in_channels,
    pretrained=True,
    img_size=256,
    prototype_shape=(2000, 512, 1, 1),
    num_classes=200,
    prototype_activation_function="log",
    add_on_layers_type="bottleneck",
):
    """Construct a ProtoPNet.
    Args:
        base_architecture (str): name of the convolutional backbone.
        in_channels (int): number of input channels.
        pretrained (bool): whether to use pretrained weights for the convolutional backbone (default: true).
        img_size (int): image size (default: 224).
        prototype_shape (tuple): prototype shape (default: (2000, 512, 1, 1)).
        num_classes (int): number of classes in the dataset (default: 200).
        prototype_activation_function: prototype activation function (default: log).
        add_on_layers_type: type of add on layers (default: bottleneck).
    Returns:
        torch.nn.Module: ProtoPNet.
    """

    # Use base architecture but replace the first conv layer
    features = base_architecture_to_features[base_architecture](pretrained=pretrained)
    old = features.conv1
    features.conv1 = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=(old.bias is not None),
    )
    nn.init.kaiming_normal_(features.conv1.weight, mode="fan_out", nonlinearity="relu")

    layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
    proto_layer_rf_info = compute_proto_layer_rf_info_v2(
        img_size=img_size,
        layer_filter_sizes=layer_filter_sizes,
        layer_strides=layer_strides,
        layer_paddings=layer_paddings,
        prototype_kernel_size=prototype_shape[2],
    )
    return SegmentationPPNet(
        features=features,
        img_size=img_size,
        prototype_shape=prototype_shape,
        proto_layer_rf_info=proto_layer_rf_info,
        num_classes=num_classes,
        init_weights=True,
        prototype_activation_function=prototype_activation_function,
        add_on_layers_type=add_on_layers_type,
    )
