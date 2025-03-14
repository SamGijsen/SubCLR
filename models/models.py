import torch
import torch.nn as nn
import torch.nn.functional as F
from base.base_model import BaseModel

from torch import Tensor

class EEG_ResNet(BaseModel):
    def __init__(self, 
                 in_channels=1, 
                 conv1_params=[(4, 32, 1), (8, 32, 1), (16, 32, 1)], # size, dim, stride
                 n_blocks=4, 
                 res_params=[(4, 32, 1), (8, 32, 1), (16, 32, 1)],
                 res_pool_size=[4, 4, 4, 4],
                 dropout_p=False,
                 res_dropout_p=False,
                 proj_size=None): 
        super().__init__()
        self.dropout_p = dropout_p
        self.init_dim = sum([k[1] for k in conv1_params])
        self.res_dim = sum([k[1] for k in res_params])
        self.proj_size = proj_size
        assert self.init_dim == self.res_dim # currently only allow for differences in kernel size or stride

        # initial conv block
        self.conv1_layers = nn.ModuleList([
        nn.Sequential(
            nn.ReflectionPad1d((kernel_size // 2, (kernel_size - 1) // 2)),
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        )            
        for kernel_size, out_channels, stride in conv1_params])

        self.bn1 = nn.BatchNorm1d(self.init_dim)
        self.elu1 = nn.ELU(inplace=True)
        
        # residual blocks
        blocks = [EEG_ResNet_ResidualBlock(self.res_dim, res_params, res_pool_size[i], res_dropout_p=res_dropout_p)
                  for i in range(n_blocks)]
        self.res_blocks = nn.Sequential(*blocks)

        # post res blocks
        self.average_pool = nn.AdaptiveAvgPool1d(1)
        if self.dropout_p:
            self.dropout = nn.Dropout(self.dropout_p)
        self.flat = nn.Flatten(start_dim=1)

        if self.proj_size:
            if len(self.proj_size) == 2:
                self.proj = nn.Sequential(
                    nn.Linear(self.res_dim, self.proj_size[0]),
                    nn.BatchNorm1d(self.proj_size[0]),
                    nn.ELU(inplace=True),
                    nn.Linear(self.proj_size[0], self.proj_size[1], bias=False))
            elif len(self.proj_size) == 1:
                self.proj = nn.Sequential(
                    nn.Linear(self.res_dim, self.proj_size[0], bias=False)
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_outputs = [conv_layer(x) for conv_layer in self.conv1_layers]
        x = torch.cat(conv_outputs, dim=1)
        x = self.bn1(x)
        x = self.elu1(x)

        x = self.res_blocks(x)

        x = self.average_pool(x)
        if self.dropout_p:
            x = self.dropout(x)
        x = self.flat(x)

        if self.proj_size:
            proj_x = self.proj(x)
            return x, proj_x
        else:
            return x

class EEG_ResNet_ResidualBlock(BaseModel):
    def __init__(self, in_channels, 
                 conv_params=[(4, 32, 1), (8, 32, 1), (16, 32, 1)],
                 pool_size=4,
                 res_dropout_p=False):
        super(EEG_ResNet_ResidualBlock, self).__init__()
        self.pool_size = pool_size
        self.res_dropout_p = res_dropout_p

        # first conv with kernel_size and stride
        self.conv1_layers = nn.ModuleList([
        nn.Sequential(
            nn.ReflectionPad1d((kernel_size // 2, (kernel_size - 1) // 2)),
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        )            
        for kernel_size, out_channels, stride in conv_params])

        self.bn1 = nn.BatchNorm1d(sum([k[1] for k in conv_params]))
        self.elu1 = nn.ELU(inplace=True)

        # second conv with kernel_size and stride=1
        self.conv2_layers = nn.ModuleList([
        nn.Sequential(
            nn.ReflectionPad1d((kernel_size // 2, (kernel_size - 1) // 2)),
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1)
        )            
        for kernel_size, out_channels, stride in conv_params])

        self.bn2 = nn.BatchNorm1d(sum([k[1] for k in conv_params]))
        self.maxpool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)

        # skip conv with kernel_size=1 and stride
        self.stream_conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
            for _, out_channels, stride in conv_params
        ])
        self.stream_bn = nn.BatchNorm1d(sum([k[1] for k in conv_params]))

        if self.res_dropout_p:
            self.dropout = nn.Dropout(self.res_dropout_p)


    def forward(self, x: Tensor) -> Tensor:
        
        # residual block with two convolutions
        conv1_outputs = [conv_layer(x) for conv_layer in self.conv1_layers]
        out = torch.cat(conv1_outputs, dim=1)
        out = self.bn1(out)
        out = self.elu1(out)

        conv2_outputs = [conv_layer(out) for conv_layer in self.conv2_layers]
        out = torch.cat(conv2_outputs, dim=1)
        out = self.bn2(out)

        # residual connection 
        res_outputs = [conv_layer(x) for conv_layer in self.stream_conv_layers]
        res = torch.cat(res_outputs, dim=1)
        res = self.stream_bn(res)

        out = out + res 

        if self.pool_size>0:
            out = self.maxpool(out)

        if self.res_dropout_p:
            out = self.dropout(out)
            
        return out

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)
    
class Epoch_Classifier_Head(BaseModel):
    def __init__(self, in_dim, dim, out_dim, dropout_p=0.0):
        super(Epoch_Classifier_Head, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(dim, out_dim, bias=False)
        )
    def forward(self, x):
        return self.layers(x)

class ResNet1_Projector(BaseModel):
    def __init__(self, in_dim=64, dim=128, out_dim=64, n_layers=3, bn=False):
        super().__init__()
        # we need an initial and final layer.
        assert n_layers > 1

        # first layer
        layers = [nn.Linear(in_dim, dim)]
        if bn:
            layers.append(nn.BatchNorm1d(dim))
        layers.append(nn.ReLU(inplace=True))

        # append additional middle layers depending on n_layers
        for _ in range(n_layers - 2):  # Subtract 2 to account for the first and last layers
            layers.append(nn.Linear(dim, dim))
            if bn:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))

        # last layer
        layers.append(nn.Linear(dim, out_dim, bias=False))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class ResNet1_SpatialClassifier(BaseModel):
    def __init__(self, in_channels=21, in_dim=64, dim=128, out_dim=1, bn=False, dropout_p=False):
        super().__init__()
        if not isinstance(dim, list):
            dim = [dim, dim]
        self.bn = bn

        self.spatial_conv = nn.Conv1d(in_channels=in_channels, out_channels=dim[0], kernel_size=1)
        self.flatten = (nn.Flatten(start_dim=1))
        self.relu1 = nn.ReLU(inplace=True)
        self.dense1 = nn.Linear(dim[0]*in_dim, dim[1])
        self.relu2 = nn.ReLU(inplace=True)
        self.dense_final = nn.Linear(dim[1], out_dim, bias=False)

        if self.bn:
            self.bn1 = nn.BatchNorm1d(dim[0]*in_dim)
            self.bn2 = nn.BatchNorm1d(dim[1])
        self.dropout_p = dropout_p
        if self.dropout_p:
            self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.spatial_conv(x)
        x = self.flatten(x)
        if self.bn:
            x = self.bn1(x)
        x = self.relu1(x)
        if self.dropout_p:
            x = self.dropout(x)
        x = self.dense1(x)
        if self.bn:
            x = self.bn2(x)
        x = self.relu2(x)
        if self.dropout_p:
            x = self.dropout(x)
        x = self.dense_final(x)

        return x

class ShallowNet(BaseModel):
    def __init__(
            self, 
            in_channels: int,
            n_time_samples: int,
            n_classes: int,
            n_filt_time: int = 40,
            n_filt_spat: int = 40,
            len_filt_time: int = 25,
            mean_pool_len: int = 75,
            mean_pool_stride: int = 15,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.add_module("conv_time", nn.Conv2d(1, n_filt_time, (len_filt_time, 1), stride=1))
        self.add_module("conv_spat", nn.Conv2d(n_filt_time, n_filt_spat, (1, in_channels), stride=1, bias=False))
        self.add_module("batch_norm", nn.BatchNorm2d(n_filt_spat))
        self.add_module("mean_pool", nn.AvgPool2d((mean_pool_len, 1), stride=(mean_pool_stride, 1)))
        self.add_module("dropout", nn.Dropout(0.5))
        # self.add_module("l1", nn.Linear(
        #     int(((n_time_samples - len_filt_time) - mean_pool_len) / mean_pool_stride + 1)*n_filt_spat, 
        #     1
        #     ))
        self.final_conv_len = int(((n_time_samples - len_filt_time) - mean_pool_len) / mean_pool_stride + 1)
        self.add_module("conv_classifier", nn.Conv2d(n_filt_spat, n_classes, (self.final_conv_len, 1), bias=True))

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.batch_norm(x)
        x = torch.square(x)
        x = self.mean_pool(x)
        x = torch.log(x)
        x = self.dropout(x)
        x = self.conv_classifier(x)

        # x = x.view(x.shape[0], -1)
        # x = self.l1(x)
        x = x.squeeze(-1).squeeze(-1)

        return x
    
class ShallowNet_1D(BaseModel):
    def __init__(
            self, 
            n_time_samples: int,
            n_filt_time: int = 40,
            len_filt_time: int = 25,
            mean_pool_len: int = 75,
            mean_pool_stride: int = 15,
            dropout_p: float = 0.5
    ):
        super().__init__()
        self.dropout_p = dropout_p
        
        self.add_module("conv_time", nn.Conv1d(1, n_filt_time, kernel_size=len_filt_time))
        self.add_module("batch_norm", nn.BatchNorm1d(n_filt_time))
        self.add_module("mean_pool", nn.AvgPool1d(mean_pool_len, stride=mean_pool_stride))
        if dropout_p > 0:
            self.add_module("dropout", nn.Dropout(dropout_p))
        
        output_after_conv = n_time_samples - len_filt_time + 1
        output_after_pool = ((output_after_conv - mean_pool_len) // mean_pool_stride) + 1
        self.l1 = nn.Linear(output_after_pool * n_filt_time, 96)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_time(x)
        x = self.batch_norm(x)
        x = torch.square(x)
        x = self.mean_pool(x)
        x = torch.log(x)
        x = x.flatten(start_dim=1)
        if self.dropout_p:
            x = self.dropout(x)
        x = self.l1(x)

        return x
    
class ShallowNet_Encoder(BaseModel):
    def __init__(
            self, 
            in_channels: int,
            n_time_samples: int,
            n_filt_time: int = 40,
            n_filt_spat: int = 40,
            len_filt_time: int = 25,
            mean_pool_len: int = 75,
            mean_pool_stride: int = 15,
    ):
        super().__init__()
        
        self.add_module("conv_time", nn.Conv2d(1, n_filt_time, (1, len_filt_time), stride=1))
        self.add_module("batch_norm", nn.BatchNorm2d(n_filt_time))
        self.add_module("conv_spat", nn.Conv2d(n_filt_time, n_filt_spat, (in_channels, 1), stride=1))
        self.add_module("mean_pool", nn.AvgPool2d((1, mean_pool_len), stride=(1, mean_pool_stride)))
        self.add_module("dropout", nn.Dropout(0.5))
        self.add_module("l1", nn.Linear(
            int(((n_time_samples - len_filt_time) - mean_pool_len) / mean_pool_stride + 1)*n_filt_spat, 
            96
            ))

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_time(x)
        x = self.batch_norm(x)
        x = self.conv_spat(x)
        x = torch.square(x)
        x = self.mean_pool(x)
        x = torch.log(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        x = self.l1(x)

        return x

