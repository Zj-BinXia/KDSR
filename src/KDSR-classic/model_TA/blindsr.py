import torch
from torch import nn
import model.common as common
import torch.nn.functional as F


def make_model(args):
    return BlindSR(args)

class IDR_DDC(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, reduction):
        super(IDR_DDC, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        self.kernel = nn.Sequential(
            nn.Linear(channels_in, channels_in, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(channels_in, channels_in * self.kernel_size * self.kernel_size, bias=False)
        )

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        b, c, h, w = x[0].size()

        # branch 1
        kernel = self.kernel(x[1]).view(-1, 1, self.kernel_size, self.kernel_size)
        out = F.conv2d(x[0].view(1, -1, h, w), kernel, groups=b*c, padding=(self.kernel_size-1)//2)
        out = out.view(b, -1, h, w)


        return out



class IDR_DCRB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction):
        super(IDR_DCRB, self).__init__()

        self.da_conv1 = IDR_DDC(n_feat, n_feat, kernel_size, reduction)
        self.conv1 = conv(n_feat, n_feat, kernel_size)
        self.relu =  nn.LeakyReLU(0.1, True)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''

        out = self.relu(self.da_conv1(x))
        out = self.conv1(out)
        out = out  + x[0]

        return out


class DAG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_blocks):
        super(DAG, self).__init__()
        self.n_blocks = n_blocks
        modules_body = [
            IDR_DCRB(conv, n_feat, kernel_size, reduction) \
            for _ in range(n_blocks)
        ]
        # modules_body.append(conv(n_feat, n_feat, kernel_size))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        res = x[0]
        for i in range(self.n_blocks):
            res = self.body[i]([res, x[1]])

        return res


class KDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(KDSR, self).__init__()
        n_blocks = args.n_blocks 
        n_feats = args.n_feats
        kernel_size = 3
        reduction = 8
        scale = int(args.scale[0])

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        # head module
        modules_head = [conv(3, n_feats, kernel_size)]
        self.head = nn.Sequential(*modules_head)

        

        # body
        modules_body = [
            DAG(common.default_conv, n_feats, kernel_size, reduction, n_blocks)
        ]
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

        # tail
        modules_tail = [common.Upsampler(conv, scale, n_feats, act=False),
                        conv(n_feats, 3, kernel_size)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, k_v):

        # sub mean
        x = self.sub_mean(x)

        # head
        x = self.head(x)

        # body
        res = x
        res = self.body[0]([res, k_v])
        res = self.body[-1](res)
        res = res + x

        # tail
        x = self.tail(res)

        # add mean
        x = self.add_mean(x)

        return x



class KD_IDE(nn.Module):
    def __init__(self,args):
        super(KD_IDE, self).__init__()
        n_feats = args.n_feats
        n_resblocks = args.n_resblocks
        scale = int(args.scale[0])
        E1=[nn.Conv2d(3+scale*scale*3, n_feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True)]
        E2=[
            common.ResBlock(
                common.default_conv, n_feats, kernel_size=3
            ) for _ in range(n_resblocks)
        ]
        E3=[
            nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        ]
        E=E1+E2+E3
        self.E = nn.Sequential(
            *E
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True)
        )

        # compress
        self.compress = nn.Sequential(
            nn.Linear(n_feats * 4, n_feats),
            nn.LeakyReLU(0.1, True)
        )

    def forward(self, x):
        fea = self.E(x).squeeze(-1).squeeze(-1)
        T_fea = []
        fea1 = self.mlp(fea)
        fea = self.compress(fea1)
        T_fea.append(fea1)
        return fea,T_fea


class BlindSR(nn.Module):
    def __init__(self, args):
        super(BlindSR, self).__init__()

        # Generator
        self.G = KDSR(args)

        self.E = KD_IDE(args)


    def forward(self, x, deg_repre):
        if self.training:

            # degradation-aware represenetion learning
            deg_repre, T_fea = self.E(deg_repre)

            # degradation-aware SR
            sr = self.G(x, deg_repre)

            return sr, T_fea
        else:
            # degradation-aware represenetion learning
            deg_repre, _ = self.E(deg_repre)

            # degradation-aware SR
            sr = self.G(x, deg_repre)

            return sr
