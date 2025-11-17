import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np

# Assuming these can be imported from OFA's toolkit or your existing files
from ofa.utils.layers import (
    IdentityLayer,
    ResidualBlock, # This was used in OFAResNets32x32_10_26 for the static subnet, might need a generic version
    ConvLayer,
    LinearLayer,
    MyModule,
    set_layer_from_config # For building static subnet from config
)
from ofa.utils import get_same_padding, make_divisible, MyNetwork, val2list, get_net_device, build_activation, MyGlobalAvgPool2d
from ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import (
    DynamicConvLayer,
    DynamicLinearLayer
)

from ofa.imagenet_classification.elastic_nn.modules.dynamic_op import (
    DynamicSeparableConv2d,
    DynamicConv2d,
    DynamicBatchNorm2d,
    DynamicSE,
    DynamicGroupNorm,
    DynamicLinear
)
from ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import copy_bn

class NewResidualBlock(MyModule):
    """
    Residual block derived from DynamicResidualBlock.
    This block is a part of subnetwork extracted from supernetwork
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        expand_ratio=0.25,
        mid_channels=None,
        act_func="relu",
        groups=1,
        downsample_mode="avgpool_conv",
    ):
        super(NewResidualBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels
        self.act_func = act_func
        self.groups = groups

        self.downsample_mode = downsample_mode

        if self.mid_channels is None:
            feature_dim = round(self.out_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        feature_dim = make_divisible(feature_dim, MyNetwork.CHANNEL_DIVISIBLE)
        self.mid_channels = feature_dim

        pad = get_same_padding(self.kernel_size)
        self.conv1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(
                            self.in_channels,
                            feature_dim,
                            kernel_size,
                            stride,
                            groups=groups,
                            padding=pad,
                            bias=False,
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(feature_dim)),
                    ("act", build_activation(self.act_func, inplace=True)),
                ]
            )
        )
        self.conv2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(
                            feature_dim,
                            self.out_channels,
                            kernel_size,
                            padding=pad,
                            bias=False,
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(self.out_channels)),
                    ("act", build_activation(self.act_func, inplace=True)),
                ]
            )
        )

        if stride == 1 and in_channels == out_channels:
            self.downsample = IdentityLayer(in_channels, out_channels)
        elif self.downsample_mode == "avgpool_conv":
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "avg_pool",
                            nn.AvgPool2d(
                                kernel_size=stride,
                                stride=stride,
                                padding=0,
                                ceil_mode=True,
                            ),
                        ),
                        (
                            "conv",
                            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                        ),
                        ("bn", nn.BatchNorm2d(out_channels)),
                    ]
                )
            )
        else:
            raise NotImplementedError

        self.final_act = build_activation(self.act_func, inplace=True)

    def forward(self, x):
        residual = self.downsample(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = x + residual
        x = self.final_act(x)
        return x

    @property
    def module_str(self):
        return "(%s, %s)" % (
            "%dx%d_ResidualBlock_%d->%d->%d_S%d_G%d"
            % (
                self.kernel_size,
                self.kernel_size,
                self.in_channels,
                self.mid_channels,
                self.out_channels,
                self.stride,
                self.groups,
            ),
            "Identity"
            if isinstance(self.downsample, IdentityLayer)
            else self.downsample_mode,
        )

    @property
    def config(self):
        return {
            "name": NewResidualBlock.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "expand_ratio": self.expand_ratio,
            "mid_channels": self.mid_channels,
            "act_func": self.act_func,
            "groups": self.groups,
            "downsample_mode": self.downsample_mode,
        }

    @staticmethod
    def build_from_config(config):
        return NewResidualBlock(**config)


class DynamicResidualBlock(MyModule):
    """
    Dynamic Residual Block part of supernet ranging from Resnets
    from 10-26 layers.
    """

    def __init__(
        self,
        in_channel_list,
        out_channel_list,
        expand_ratio_list=0.25,
        kernel_size=3,
        stride=1,
        act_func="relu",
        downsample_mode="avgpool_conv",
        bn_gamma_zero_init=False,
    ):
        super(DynamicResidualBlock, self).__init__()

        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list
        self.expand_ratio_list = val2list(expand_ratio_list)
        self.bn_gamma_zero_init = bn_gamma_zero_init

        self.kernel_size = kernel_size
        self.stride = stride
        self.act_func = act_func
        self.downsample_mode = downsample_mode

        # build modules
        max_middle_channel = make_divisible(
            round(max(self.out_channel_list) * max(self.expand_ratio_list)),
            MyNetwork.CHANNEL_DIVISIBLE,
        )

        self.conv1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        DynamicConv2d(
                            max(self.in_channel_list),
                            max_middle_channel,
                            kernel_size,
                            stride,
                        ),
                    ),
                    ("bn", DynamicBatchNorm2d(max_middle_channel)),
                    ("act", build_activation(self.act_func, inplace=True)),
                ]
            )
        )
        self.conv2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        DynamicConv2d(
                            max_middle_channel, max(self.out_channel_list), kernel_size,
                        ),
                    ),
                    ("bn", DynamicBatchNorm2d(max(self.out_channel_list))),
                    ("act", build_activation(self.act_func, inplace=True)),
                ]
            )
        )

        # if self.stride == 1 and self.in_channel_list == self.out_channel_list:
        #     self.downsample = IdentityLayer(
        #         max(self.in_channel_list), max(self.out_channel_list)
        #     )
        # elif self.downsample_mode == "avgpool_conv":
        #     self.downsample = nn.Sequential(
        #         OrderedDict(
        #             [
        #                 (
        #                     "avg_pool",
        #                     nn.AvgPool2d(
        #                         kernel_size=stride,
        #                         stride=stride,
        #                         padding=0,
        #                         ceil_mode=True,
        #                     ),
        #                 ),
        #                 (
        #                     "conv",
        #                     DynamicConv2d(
        #                         max(self.in_channel_list), max(self.out_channel_list),
        #                     ),
        #                 ),
        #                 ("bn", DynamicBatchNorm2d(max(self.out_channel_list))),
        #             ]
        #         )
        #     )


        if self.downsample_mode == "avgpool_conv":
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "avg_pool",
                            nn.AvgPool2d(
                                kernel_size=stride,
                                stride=stride,
                                padding=0,
                                ceil_mode=True,
                            ),
                        ),
                        (
                            "conv",
                            DynamicConv2d(
                                max(self.in_channel_list), max(self.out_channel_list), kernel_size=1
                            ),
                        ),
                        ("bn", DynamicBatchNorm2d(max(self.out_channel_list))),
                    ]
                )
            )
        else:
            raise NotImplementedError


        self.final_act = build_activation(self.act_func, inplace=True)
        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_out_channel = max(self.out_channel_list)

        if self.bn_gamma_zero_init:
            nn.init.constant_(self.conv2.bn.bn.weight, 0)

    def forward(self, x):
        feature_dim = self.active_middle_channels
        self.conv1.conv.active_out_channel = feature_dim
        self.conv2.conv.active_out_channel = self.active_out_channel
        if not isinstance(self.downsample, IdentityLayer):
            self.downsample.conv.active_out_channel = self.active_out_channel

        residual = self.downsample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        x = self.final_act(x)
        return x

    @property
    def module_str(self):
        return "(%s, %s)" % (
            "%dx%d_Residual_in->%d->%d_S%d"
            % (
                self.kernel_size,
                self.kernel_size,
                self.active_middle_channels,
                self.active_out_channel,
                self.stride,
            ),
            "Identity"
            if isinstance(self.downsample, IdentityLayer)
            else self.downsample_mode,
        )

    @property
    def config(self):
        return {
            "name": DynamicResidualBlock.__name__,
            "in_channel_list": self.in_channel_list,
            "out_channel_list": self.out_channel_list,
            "expand_ratio_list": self.expand_ratio_list,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "act_func": self.act_func,
            "downsample_mode": self.downsample_mode,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicResidualBlock(**config)

    @property
    def in_channels(self):
        return max(self.in_channel_list)

    @property
    def out_channels(self):
        return max(self.out_channel_list)

    @property
    def active_middle_channels(self):
        feature_dim = round(self.active_out_channel * self.active_expand_ratio)
        feature_dim = make_divisible(feature_dim, MyNetwork.CHANNEL_DIVISIBLE)
        return feature_dim

    def get_active_subnet(self, in_channel, preserve_weight=True):
        # build the new layer
        sub_layer = NewResidualBlock.build_from_config(
            self.get_active_subnet_config(in_channel)
        )
        sub_layer = sub_layer.to(get_net_device(self))
        if not preserve_weight:
            return sub_layer

        # copy weight from current layer
        sub_layer.conv1.conv.weight.data.copy_(
            self.conv1.conv.get_active_filter(
                self.active_middle_channels, in_channel
            ).data
        )
        copy_bn(sub_layer.conv1.bn, self.conv1.bn.bn)

        sub_layer.conv2.conv.weight.data.copy_(
            self.conv2.conv.get_active_filter(
                self.active_out_channel, self.active_middle_channels
            ).data
        )
        copy_bn(sub_layer.conv2.bn, self.conv2.bn.bn)

        # if not isinstance(self.downsample, IdentityLayer):
        #     sub_layer.downsample.conv.weight.data.copy_(
        #         self.downsample.conv.get_active_filter(
        #             self.active_out_channel, in_channel
        #         ).data
        #     )
        #     copy_bn(sub_layer.downsample.bn, self.downsample.bn.bn)

        if not isinstance(self.downsample, IdentityLayer) and not isinstance(sub_layer.downsample, IdentityLayer):
            sub_layer.downsample.conv.weight.data.copy_(
                self.downsample.conv.get_active_filter(
                    self.active_out_channel, in_channel
                ).data
            )
            copy_bn(sub_layer.downsample.bn, self.downsample.bn.bn)

        return sub_layer

    def get_active_subnet_config(self, in_channel):
        return {
            #             'name': NewResidualBlock.__name__,
            "in_channels": in_channel,
            "out_channels": self.active_out_channel,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "expand_ratio": self.active_expand_ratio,
            "mid_channels": self.active_middle_channels,
            "act_func": self.act_func,
            "groups": 1,
            "downsample_mode": self.downsample_mode,
        }

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        raise NotImplementedError

class GenericStaticResNetSubnet(MyNetwork): # Analogous to ResNets32x32_10_26
    def __init__(self, input_stem, blocks, classifier):
        super(GenericStaticResNetSubnet, self).__init__()
        self.input_stem = nn.ModuleList(input_stem)
        self.blocks = nn.ModuleList(blocks)
        self.global_avg_pool = MyGlobalAvgPool2d(keep_dim=False)
        self.classifier = classifier

    def forward(self, x):
        for layer in self.input_stem:
            x = layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x

    @property
    def config(self):
        # TODO: Implement this to reflect the specific static configuration
        return {
            'name': GenericStaticResNetSubnet.__name__,
            'bn': self.get_bn_param(),
            'input_stem': [layer.config for layer in self.input_stem],
            'blocks': [block.config for block in self.blocks],
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        # TODO: Implement this thoroughly
        input_stem = [set_layer_from_config(cfg) for cfg in config.get('input_stem', [])]
        blocks = [NewResidualBlock.build_from_config(cfg) for cfg in config.get('blocks', [])] # Assuming NewResidualBlock is for static
        classifier = set_layer_from_config(config['classifier'])
        
        net = GenericStaticResNetSubnet(input_stem, blocks, classifier)
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        return net


class GenericOFAResNet(MyNetwork): # Analogous to OFAResNets32x32_10_26
    def __init__(self,
                 num_stages: int,
                 initial_input_hw: int, # Not directly used in layer construction, but for info
                 initial_input_channels: int,
                 stem_stride: int,
                 original_stem_out_channels: int,
                 original_stage_base_channels: list, # list of ints
                 stage_downsample_factors: list, # list of ints (strides)
                 max_extra_blocks_per_stage: int,
                 channel_divisible_by: int,
                 width_multiplier_choices: list, # list of floats
                 expansion_ratio_choices: list, # list of floats
                 n_classes=1000,
                 bn_param=(0.1, 1e-5), # (momentum, eps)
                 dropout_rate=0,
                 bn_gamma_zero_init=False,
                 act_func='relu' # Default activation
                ):
        super(GenericOFAResNet, self).__init__()

        # Store architecture definition parameters
        self.num_stages = num_stages
        self.initial_input_hw = initial_input_hw
        self.initial_input_channels = initial_input_channels
        self.stem_stride = stem_stride
        self.original_stem_out_channels = original_stem_out_channels
        self.original_stage_base_channels = val2list(original_stage_base_channels, self.num_stages)
        self.stage_downsample_factors = val2list(stage_downsample_factors, self.num_stages)
        self.max_extra_blocks_per_stage = max_extra_blocks_per_stage
        self.channel_divisible_by = channel_divisible_by
        self.width_multiplier_choices = sorted(list(set(val2list(width_multiplier_choices))))
        self.expansion_ratio_choices = sorted(list(set(val2list(expansion_ratio_choices))))
        self.n_classes = n_classes
        self.bn_gamma_zero_init = bn_gamma_zero_init
        self.act_func = act_func

        # Calculate max possible channel counts for dynamic layers
        self.stem_out_channel_list = [
            make_divisible(self.original_stem_out_channels * w, self.channel_divisible_by)
            for w in self.width_multiplier_choices
        ]
        if not self.stem_out_channel_list: self.stem_out_channel_list = [self.original_stem_out_channels]


        # Input Stem
        self.input_stem = nn.ModuleList([
            DynamicConvLayer(
                in_channel_list=val2list(self.initial_input_channels),
                out_channel_list=self.stem_out_channel_list,
                kernel_size=3, # Assuming 3x3 stem, make configurable if needed
                stride=self.stem_stride,
                act_func=self.act_func,
                use_bn=True
            )
        ])

        # Blocks
        self.blocks = nn.ModuleList()
        current_max_in_channels_list = self.stem_out_channel_list # Input to the first stage

        for i in range(self.num_stages):
            stage_max_out_channels_list = [
                make_divisible(self.original_stage_base_channels[i] * w, self.channel_divisible_by)
                for w in self.width_multiplier_choices
            ]
            if not stage_max_out_channels_list: stage_max_out_channels_list = [self.original_stage_base_channels[i]]


            for block_idx in range(self.max_extra_blocks_per_stage + 1):
                stride = self.stage_downsample_factors[i] if block_idx == 0 else 1
                
                # For the first block of a stage, in_channel_list is from the previous stage/stem.
                # For subsequent blocks within the same stage, in_channel_list is the current stage's out_channel_list.
                block_in_channels_list = current_max_in_channels_list if block_idx == 0 else stage_max_out_channels_list

                residual_block = DynamicResidualBlock( # Using the block from ofa_resnets_32x32_10_26
                    in_channel_list=block_in_channels_list,
                    out_channel_list=stage_max_out_channels_list,
                    expand_ratio_list=self.expansion_ratio_choices,
                    kernel_size=3, # Assuming 3x3, make configurable if needed
                    stride=stride,
                    act_func=self.act_func,
                    bn_gamma_zero_init=self.bn_gamma_zero_init,
                    # downsample_mode can be 'avgpool_conv' or 'conv'
                )
                self.blocks.append(residual_block)
            current_max_in_channels_list = stage_max_out_channels_list # Output of this stage is input to next

        # Global Average Pool
        self.global_avg_pool = MyGlobalAvgPool2d(keep_dim=False)

        # Classifier
        self.classifier = DynamicLinearLayer(
            in_features_list=current_max_in_channels_list, # From the last stage
            out_features=self.n_classes,
            dropout_rate=dropout_rate
        )

        # Runtime depth: stores how many blocks to *skip* from the end of each stage's definition
        self.runtime_depth = [self.max_extra_blocks_per_stage] * self.num_stages
        self.set_bn_param(*bn_param)
        self.set_max_net() # Initialize to max network

    @property
    def grouped_block_index(self):
        # Calculates lists of block indices for each stage
        # Assumes self.blocks contains all blocks sequentially, stage by stage
        # Each stage has (self.max_extra_blocks_per_stage + 1) blocks defined in self.blocks
        num_blocks_defined_per_stage = self.max_extra_blocks_per_stage + 1
        grouped_indexes = []
        current_idx = 0
        for _ in range(self.num_stages):
            grouped_indexes.append(list(range(current_idx, current_idx + num_blocks_defined_per_stage)))
            current_idx += num_blocks_defined_per_stage
        return grouped_indexes

    def set_active_subnet(self, d: list, e_indices: list, w_indices: list, **kwargs):
        """
        d: list of extra block counts for each stage. len(d) == num_stages. d[i] in [0, max_extra_blocks_per_stage]
        e_indices: FLAT list of indices into self.expansion_ratio_choices.
                   Expected length: num_stages * (max_extra_blocks_per_stage + 1)
        w_indices: list of indices into self.width_multiplier_choices.
                   len(w_indices) == num_stages + 1 (1 for stem, num_stages for stages)
        """
        if len(d) != self.num_stages:
            raise ValueError(f"Depth vector 'd' length {len(d)} != num_stages {self.num_stages}")
        if len(w_indices) != self.num_stages + 1:
            raise ValueError(f"Width_indices vector 'w_indices' length {len(w_indices)} != num_stages+1 {self.num_stages + 1}")
        
        expected_e_len = self.num_stages * (self.max_extra_blocks_per_stage + 1)
        if len(e_indices) != expected_e_len:
            raise ValueError(f"Expansion_indices vector 'e_indices' length {len(e_indices)} != expected {expected_e_len}")

        # Set active stem output channel
        stem_w_idx = w_indices[0]
        self.input_stem[0].active_out_channel = self.stem_out_channel_list[stem_w_idx]

        e_idx_offset = 0
        for stage_id in range(self.num_stages):
            # Determine how many blocks are active in this stage
            num_extra_blocks_active = d[stage_id]
            if not (0 <= num_extra_blocks_active <= self.max_extra_blocks_per_stage):
                raise ValueError(f"Depth choice d[{stage_id}]={num_extra_blocks_active} out of range [0, {self.max_extra_blocks_per_stage}]")
            
            self.runtime_depth[stage_id] = self.max_extra_blocks_per_stage - num_extra_blocks_active
            
            stage_w_idx = w_indices[stage_id + 1]
            stage_active_out_channel = make_divisible(
                self.original_stage_base_channels[stage_id] * self.width_multiplier_choices[stage_w_idx],
                self.channel_divisible_by
            )
            if stage_active_out_channel < self.channel_divisible_by:
                stage_active_out_channel = self.channel_divisible_by


            stage_block_indices = self.grouped_block_index[stage_id]
            
            num_blocks_defined_this_stage = self.max_extra_blocks_per_stage + 1

            for i in range(num_blocks_defined_this_stage): # Iterate all defined blocks for the stage
                block_supernet_idx = stage_block_indices[i]
                current_block = self.blocks[block_supernet_idx]
                current_block.active_out_channel = stage_active_out_channel
                
                # Only set active_expand_ratio if the block is actually active in the current subnet
                if i < (num_extra_blocks_active + 1): # (num_extra_blocks_active + 1) is num active blocks
                    expansion_idx_for_block = e_indices[e_idx_offset + i]
                    if not (0 <= expansion_idx_for_block < len(self.expansion_ratio_choices)):
                         raise ValueError(f"e_indices[{e_idx_offset + i}]={expansion_idx_for_block} out of range for expansion_ratio_choices (len {len(self.expansion_ratio_choices)})")
                    current_block.active_expand_ratio = self.expansion_ratio_choices[expansion_idx_for_block]
            
            e_idx_offset += num_blocks_defined_this_stage


        # Set active features for the classifier
        # The input to classifier depends on the output of the last active stage
        final_stage_w_idx = w_indices[self.num_stages] # w_indices[-1]
        final_stage_active_out_channel = make_divisible(
            self.original_stage_base_channels[-1] * self.width_multiplier_choices[final_stage_w_idx],
            self.channel_divisible_by
        )
        if final_stage_active_out_channel < self.channel_divisible_by:
            final_stage_active_out_channel = self.channel_divisible_by

        self.classifier.active_in_features = final_stage_active_out_channel

    def set_max_net(self):
        d_max = [self.max_extra_blocks_per_stage] * self.num_stages
        e_indices_max = [len(self.expansion_ratio_choices) - 1] * (self.num_stages * (self.max_extra_blocks_per_stage + 1))
        w_indices_max = [len(self.width_multiplier_choices) - 1] * (self.num_stages + 1)
        self.set_active_subnet(d=d_max, e_indices=e_indices_max, w_indices=w_indices_max)

    def forward(self, x):
        # Stem
        for layer in self.input_stem:
            x = layer(x)

        # Blocks
        # Need to use self.grouped_block_index and self.runtime_depth to select active blocks
        all_stage_indices = self.grouped_block_index
        for stage_id in range(self.num_stages):
            num_active_blocks_this_stage = (self.max_extra_blocks_per_stage + 1) - self.runtime_depth[stage_id]
            active_block_indices_this_stage = all_stage_indices[stage_id][:num_active_blocks_this_stage]
            
            for block_idx_in_supernet in active_block_indices_this_stage:
                x = self.blocks[block_idx_in_supernet](x)
        
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x

    def get_active_subnet(self, preserve_weight=True):
        # This is complex. It needs to build a new *static* network
        # based on the current active settings and copy weights.
        
        # 1. Get active stem
        active_stem_layers = [self.input_stem[0].get_active_subnet(self.initial_input_channels, preserve_weight)]
        current_active_channels = self.input_stem[0].active_out_channel

        active_blocks = []
        all_supernet_stage_indices = self.grouped_block_index

        for stage_id in range(self.num_stages):
            num_active_blocks_this_stage = (self.max_extra_blocks_per_stage + 1) - self.runtime_depth[stage_id]
            supernet_indices_for_active_blocks = all_supernet_stage_indices[stage_id][:num_active_blocks_this_stage]

            for block_supernet_idx in supernet_indices_for_active_blocks:
                supernet_block = self.blocks[block_supernet_idx]
                # The get_active_subnet of DynamicResidualBlock needs the *actual* active input channel count
                static_block = supernet_block.get_active_subnet(current_active_channels, preserve_weight)
                active_blocks.append(static_block)
                current_active_channels = supernet_block.active_out_channel # This should be the out_channel for this stage

        active_classifier = self.classifier.get_active_subnet(current_active_channels, preserve_weight)
        
        # Create an instance of the static subnet class
        # We defined GenericStaticResNetSubnet earlier for this.
        subnet = GenericStaticResNetSubnet(active_stem_layers, active_blocks, active_classifier)
        subnet.set_bn_param(**self.get_bn_param())
        return subnet

    @property
    def config(self):
        # TODO: Implement this to represent the supernetwork's full potential configuration
        return {
            'name': GenericOFAResNet.__name__,
            'bn': self.get_bn_param(),
            # ... other parameters defining the supernet structure ...
            'num_stages': self.num_stages,
            'original_stage_base_channels': self.original_stage_base_channels,
            # ... etc.
            'input_stem': [layer.config for layer in self.input_stem],
            'blocks': [block.config for block in self.blocks],
            'classifier': self.classifier.config,
        }

    # Add other necessary methods like sample_active_subnet, get_active_net_config if needed,
    # adapting them from OFAResNets32x32_10_26.
    def sample_active_subnet(self):
        """ Samples a random configuration for d, e_indices, w_indices """
        
        # Sample d: list of extra block counts for each stage
        d_setting = [
            np.random.randint(0, self.max_extra_blocks_per_stage + 1) 
            for _ in range(self.num_stages)
        ]
        
        # Sample e_indices: FLAT list of indices into self.expansion_ratio_choices
        num_total_block_slots = self.num_stages * (self.max_extra_blocks_per_stage + 1)
        e_indices_setting = [
            np.random.randint(0, len(self.expansion_ratio_choices))
            for _ in range(num_total_block_slots)
        ]
        
        # Sample w_indices: list of indices into self.width_multiplier_choices
        w_indices_setting = [
            np.random.randint(0, len(self.width_multiplier_choices))
            for _ in range(self.num_stages + 1) # stem + stages
        ]
        
        arch_config = {
            "d": d_setting,
            "e_indices": e_indices_setting,
            "w_indices": w_indices_setting,
        }
        self.set_active_subnet(**arch_config)
        return arch_config # Return the configuration dict

    def get_active_net_config(self):
        # This should return the config of the currently *active* subnet,
        # in a format that GenericStaticResNetSubnet.build_from_config can understand.

        # 1. Get active stem config
        # DynamicConvLayer.get_active_subnet_config(in_channel)
        active_stem_configs = [self.input_stem[0].get_active_subnet_config(self.initial_input_channels)]
        current_active_channels = self.input_stem[0].active_out_channel

        active_block_configs = []
        all_supernet_stage_indices = self.grouped_block_index

        for stage_id in range(self.num_stages):
            num_active_blocks_this_stage = (self.max_extra_blocks_per_stage + 1) - self.runtime_depth[stage_id]
            supernet_indices_for_active_blocks = all_supernet_stage_indices[stage_id][:num_active_blocks_this_stage]

            for block_supernet_idx in supernet_indices_for_active_blocks:
                supernet_block = self.blocks[block_supernet_idx]
                # DynamicResidualBlock.get_active_subnet_config(in_channel)
                block_config = supernet_block.get_active_subnet_config(current_active_channels)
                active_block_configs.append(block_config)
                current_active_channels = supernet_block.active_out_channel

        # DynamicLinearLayer.get_active_subnet_config(in_features)
        active_classifier_config = self.classifier.get_active_subnet_config(current_active_channels)
        
        return {
            'name': GenericStaticResNetSubnet.__name__, # Target static class
            'bn': self.get_bn_param(),
            'input_stem': active_stem_configs,
            'blocks': active_block_configs,
            'classifier': active_classifier_config,
        }
    
    def set_min_net(self):
        d_min = [0] * self.num_stages # Smallest depth means 0 extra blocks
        e_indices_min = [0] * (self.num_stages * (self.max_extra_blocks_per_stage + 1))
        w_indices_min = [0] * (self.num_stages + 1)
        self.set_active_subnet(d=d_min, e_indices=e_indices_min, w_indices=w_indices_min)