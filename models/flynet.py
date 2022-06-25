import torch

from models.seg_oprs import *
from models.stdcnet import STDCNet813

class ESA(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x)
        # BCHW -> BHCW
        y = y.permute(0, 2, 1, 3).contiguous()
        y = self.conv(y)

        # Change the dimensions back to BCHW
        y = y.permute(0, 2, 1, 3).contiguous()
        y = torch.sigmoid_(y)
        return x * y.expand_as(x)
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class Gate(torch.nn.Module):
    def __init__(
            self,
            concat_channels: int,
    ) -> None:
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv0 = ConvBnRelu(32,concat_channels,1,1,0)
        self.concat = ConvBnRelu(concat_channels, concat_channels,3,1,1)
        self.esa = ESA()
        self.init_weight()

    def forward(self, x_t, x):
        x = self.maxpool(x)
        x = self.conv0(x)
        x = x + x_t
        xt = self.concat(x)
        xt = self.esa(xt)
        return xt

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class DBGAMoudle(nn.Module):
    def __init__(self,in_chan,out_chan,norm_layer=nn.BatchNorm2d):
        super(DBGAMoudle, self).__init__()
        self.convx = ConvBnRelu(in_chan, out_chan, 3, 1, 1,
                       has_bn=True,
                       has_relu=True, has_bias=False, norm_layer=norm_layer)
        self.init_weight()
        self.spatial = ConvBnRelu(256, 128, 3, 1, 1)
        self.refine = ConvBnRelu(128, 128, 3, 1, 1)

    def forward(self,x1,x2,x3):
        x1 = self.spatial(x1)
        x2 = self.refine(x2)
        feat_out = self.convx(x1 + x2)
        feat_out = F.interpolate(feat_out,scale_factor=2, mode='bilinear', align_corners=True) + x3
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class FlyNet(nn.Module):
    def __init__(self, classes, is_training=True,
                 norm_layer=nn.BatchNorm2d):
        super(FlyNet, self).__init__()
        self.context_path = STDCNet813(pretrain_model='./checkpoints/STDCNet813M_73.91.tar')#

        config = [64, 256, 512, 1024]
        self.business_layer = []
        self.is_training = is_training

        conv_channel = 128
        self.global_context = ConvBnRelu(config[-1], 128, 3, 1, 1)

        self.arm = MultiSpectralAttentionLayer(config[-2], 7,7,  reduction=16, freq_sel_method = 'top16')
        self.ffm = DBGAMoudle(conv_channel, config[0])

        heads = [FlyNetHead(conv_channel, classes, 8,
                             True, norm_layer),
                 FlyNetHead(conv_channel, classes, 4,
                             True, norm_layer),
                 FlyNetHead(64, classes, 4,
                             False, norm_layer)]

        self.heads = nn.ModuleList(heads)

        self.head_0 = FlyNetHead(conv_channel, classes, 8,
                             True, norm_layer)
        self.head_1 =  FlyNetHead(64, classes, 4,
                             True, norm_layer)
        self.head_2 = FlyNetHead(64, classes, 4,
                             False, norm_layer)
        self.headedge = FlyNetHead(64, 1, 4,
                             False, norm_layer)
        self.last_fusion = Gate(64)
        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            if not hasattr(child,'get_params'):
                continue
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (DBGAMoudle, FlyNetHead)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params

    def up(self,x,up_size=None):
        if not up_size is None:
            return F.interpolate(x,
                      size=up_size,
                      mode='bilinear', align_corners=True)
        return F.interpolate(x,
                      scale_factor=2,
                      mode='bilinear', align_corners=True)

    def forward(self, data):
        pred_out = []

        context_blocks = self.context_path(data)
        context_blocks.reverse()

        global_context_global = self.global_context(context_blocks[0])
        global_context = self.up(global_context_global,context_blocks[1].size()[2:])

        fm = self.arm(context_blocks[1]) + global_context
        fm = self.up(fm ,context_blocks[2].size()[2:])
        pred_out.append(fm)

        concate_fm = self.ffm(context_blocks[2], fm,context_blocks[-2])
        pred_out.append(concate_fm)

        last_fm = self.last_fusion(concate_fm,context_blocks[-1])
        pred_out.append(last_fm)

        out1 = self.head_2(pred_out[-1])
        if self.is_training:
            out2 = self.head_0(pred_out[0])
            out3 = self.head_1(pred_out[1])
            out4 = self.headedge(pred_out[-1])
            return out1,out2,out3,out4
        return out1

class FlyNetHead(nn.Module):
    def __init__(self, in_planes, out_planes, scale,
                 is_aux=False, norm_layer=nn.BatchNorm2d):
        super(FlyNetHead, self).__init__()
        if is_aux:
            self.conv_3x3 = ConvBnRelu(in_planes, 128, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)
        else:
            self.conv_3x3 = ConvBnRelu(in_planes, 64, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)
        if is_aux:
            self.conv_1x1 = nn.Conv2d(128, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        else:
            self.conv_1x1 = nn.Conv2d(64, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        self.scale = scale
        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module,nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

    def forward(self, x):
        fm = self.conv_3x3(x)
        output = self.conv_1x1(fm)
        if self.scale > 1:
            output = F.interpolate(output, scale_factor=self.scale,
                                   mode='bilinear',
                                   align_corners=True)
        return output

from torchsummary import  summary
if __name__ == "__main__":
    model = FlyNet(19,is_training=True)
    model.cuda()
    model.eval()
    summary(model,(3,360,480))
