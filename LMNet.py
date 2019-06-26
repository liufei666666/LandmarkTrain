import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.autograd import Variable
#from flops_benchmark import add_flops_counting_methods
#from torchsummary import summary



class SeparableConv2d(nn.Module):
    def __init__(self,in_planes,out_planes):
        super(SeparableConv2d,self).__init__()
        self.sblock= nn.Sequential(
            nn.Conv2d(in_planes,in_planes,3,1,1,groups=in_planes,bias=False),
            nn.Conv2d(in_planes,out_planes,1,1,0,bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )

    def forward(self,x):
        x = self.sblock(x)
        return x

class SeparableConv2dS2(nn.Module):
    def __init__(self,in_planes,out_planes):
        super(SeparableConv2dS2,self).__init__()
        self.sblock= nn.Sequential(
            nn.Conv2d(in_planes,in_planes,3,2,1,groups=in_planes,bias=False),
            nn.Conv2d(in_planes,out_planes,1,1,0,bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )

    def forward(self,x):
        x = self.sblock(x)
        return x

class SeparableConv2dS3(nn.Module):
    def __init__(self,in_planes,out_planes):
        super(SeparableConv2dS3,self).__init__()
        self.sblock= nn.Sequential(
            nn.Conv2d(in_planes,in_planes,2,1,1,groups=in_planes,bias=False),
            nn.Conv2d(in_planes,out_planes,1,1,0,bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )

    def forward(self,x):
        x = self.sblock(x)
        return x

class SeparableConv2dS4(nn.Module):
    def __init__(self,in_planes,out_planes):
        super(SeparableConv2dS4,self).__init__()
        self.sblock= nn.Sequential(
            nn.Conv2d(in_planes,in_planes,4,1,0,groups=in_planes,bias=False),
            nn.Conv2d(in_planes,out_planes,1,1,0,bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )

    def forward(self,x):
        x = self.sblock(x)
        return x


class SeparableConv2dS5(nn.Module):
    def __init__(self,in_planes,out_planes):
        super(SeparableConv2dS5,self).__init__()
        self.sblock= nn.Sequential(
            nn.Conv2d(in_planes,in_planes,3,1,0,groups=in_planes,bias=False),
            nn.Conv2d(in_planes,out_planes,1,1,0,bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )

    def forward(self,x):
        x = self.sblock(x)
        return x



class SeparableConv2dS6(nn.Module):
    def __init__(self,in_planes,out_planes):
        super(SeparableConv2dS6,self).__init__()
        self.sblock= nn.Sequential(
            nn.Conv2d(in_planes,in_planes,2,1,0,groups=in_planes,bias=False),
            nn.Conv2d(in_planes,out_planes,1,1,0,bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )

    def forward(self,x):
        x = self.sblock(x)
        return x



class SeparableConv2d5S2(nn.Module):
    def __init__(self,in_planes,out_planes):
        super(SeparableConv2d5S2,self).__init__()
        self.sblock= nn.Sequential(
            nn.Conv2d(in_planes,in_planes,5,2,2,groups=in_planes,bias=False),
            nn.Conv2d(in_planes,out_planes,1,1,0,bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )

    def forward(self,x):
        x = self.sblock(x)
        return x


class SeparableConv2d7S2(nn.Module):
    def __init__(self,in_planes,out_planes):
        super(SeparableConv2d7S2,self).__init__()
        self.sblock= nn.Sequential(
            nn.Conv2d(in_planes,in_planes,7,2,3,groups=in_planes,bias=False),
            nn.Conv2d(in_planes,out_planes,1,1,0,bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )

    def forward(self,x):
        x = self.sblock(x)
        return x


class SeparableConv2d5(nn.Module):
    def __init__(self,in_planes,out_planes):
        super(SeparableConv2d5,self).__init__()
        self.sblock= nn.Sequential(
            nn.Conv2d(in_planes,in_planes,5,1,2,groups=in_planes,bias=False),
            nn.Conv2d(in_planes,out_planes,1,1,0,bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )

    def forward(self,x):
        x = self.sblock(x)
        return x



class SeparableConv2d7(nn.Module):
    def __init__(self,in_planes,out_planes):
        super(SeparableConv2d5,self).__init__()
        self.sblock= nn.Sequential(
            nn.Conv2d(in_planes,in_planes,7,1,3,groups=in_planes,bias=False),
            nn.Conv2d(in_planes,out_planes,1,1,0,bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )

    def forward(self,x):
        x = self.sblock(x)
        return x

class LMNet14(nn.Module):
    def __init__(self):
        super(LMNet14, self).__init__()
        self.pre_layers = nn.Sequential(
                    #nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2),
                    nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
                    #nn.BatchNorm2d(24),
                    nn.ReLU(),
                )
        self.Sblock1_S2 = SeparableConv2dS2(8,16)
        self.Sblock2_S2 = SeparableConv2dS2(16,32)
        self.Sblock3 = SeparableConv2d(32,32)
        self.Sblock3_S2 = SeparableConv2d5S2(32,64)
        self.Sblock4 = SeparableConv2d(64,64)
        self.Sblock4_1 = SeparableConv2d5(64,64)
        self.Sblock4_S2 = SeparableConv2d5S2(64,128)
        self.Sblock5 = SeparableConv2d5(128,128)
        self.Sblock5_1 = SeparableConv2d5(128,128)
        self.Sblock5_S2 = SeparableConv2d7S2(128,256)
        self.classifier = nn.Sequential(
            nn.Linear(256*4*4, 256),
            nn.ReLU(),
            )
        self.lmleye = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )
        self.lmreye = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )
        self.lmlbrow = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )
        self.lmrbrow = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )
        self.lmnose = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )
        self.lmmouth = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 36),
        )

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.Sblock1_S2(out)
        out = self.Sblock2_S2(out)
        out = self.Sblock3(out)
        out = self.Sblock3_S2(out)
        out = self.Sblock4(out)
        out = self.Sblock4_1(out)
        out = self.Sblock4_S2(out)
        out = self.Sblock5(out)
        out = self.Sblock5_1(out)
        out = self.Sblock5_S2(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        lmleye = self.lmleye(out)
        lmreye = self.lmreye(out)
        lmlbrow = self.lmlbrow(out)
        lmrbrow = self.lmrbrow(out)
        lmnose = self.lmnose(out)
        lmmouth = self.lmmouth(out)
        #return out
        return lmleye,lmreye,lmlbrow,lmrbrow,lmnose,lmmouth



class LMNet14b(nn.Module):
    def __init__(self):
        super(LMNet14b, self).__init__()
        self.pre_layers = nn.Sequential(
                    #nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2),
                    nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
                    #nn.BatchNorm2d(24),
                    nn.ReLU(),
                )
        self.Sblock1_S2 = SeparableConv2dS2(8,16)
        self.Sblock2_S2 = SeparableConv2dS2(16,32)
        self.Sblock3 = SeparableConv2d(32,32)
        self.Sblock3_S2 = SeparableConv2d5S2(32,64)
        self.Sblock4 = SeparableConv2d5(64,64)
        self.Sblock4_1 = SeparableConv2d5(64,64)
        self.Sblock4_S2 = SeparableConv2d5S2(64,128)
        self.Sblock5 = SeparableConv2d5(128,128)
        self.Sblock5_1 = SeparableConv2d5(128,128)
        self.Sblock5_S2 = SeparableConv2d7S2(128,256)
        self.classifier = nn.Sequential(
            nn.Linear(256*4*4, 256),
            nn.ReLU(),
            )
        self.lmleye = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )
        self.lmreye = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )
        self.lmlbrow = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )
        self.lmrbrow = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )
        self.lmnose = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )
        self.lmmouth = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 36),
        )

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.Sblock1_S2(out)
        out = self.Sblock2_S2(out)
        out = self.Sblock3(out)
        out = self.Sblock3_S2(out)
        out = self.Sblock4(out)
        out = self.Sblock4_1(out)
        out = self.Sblock4_S2(out)
        out = self.Sblock5(out)
        out = self.Sblock5_1(out)
        out = self.Sblock5_S2(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        lmleye = self.lmleye(out)
        lmreye = self.lmreye(out)
        lmlbrow = self.lmlbrow(out)
        lmrbrow = self.lmrbrow(out)
        lmnose = self.lmnose(out)
        lmmouth = self.lmmouth(out)
        #return out
        return lmleye,lmreye,lmlbrow,lmrbrow,lmnose,lmmouth



class LMNet14c(nn.Module):
    def __init__(self):
        super(LMNet14b, self).__init__()
        self.pre_layers = nn.Sequential(
                    #nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2),
                    nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
                    #nn.BatchNorm2d(24),
                    nn.ReLU(),
                )
        self.Sblock1_S2 = SeparableConv2dS2(8,16)
        self.Sblock2_S2 = SeparableConv2dS2(16,32)
        self.Sblock3 = SeparableConv2d(32,32)
        self.Sblock3_S2 = SeparableConv2d5S2(32,64)
        self.Sblock4 = SeparableConv2d5(64,64)
        self.Sblock4_1 = SeparableConv2d5(64,64)
        self.Sblock4_S2 = SeparableConv2d5S2(64,128)
        self.Sblock5 = SeparableConv2d5(128,128)
        self.Sblock5_1 = SeparableConv2d5(128,128)
        self.Sblock5_S2 = SeparableConv2d7S2(128,256)
        self.classifier = nn.Sequential(
            nn.Linear(256*4*4, 256),
            nn.ReLU(),
            )
        self.lmleye = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )
        self.lmreye = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )
        self.lmlbrow = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )
        self.lmrbrow = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )
        self.lmnose = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )
        self.lmmouth = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 36),
        )

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.Sblock1_S2(out)
        out = self.Sblock2_S2(out)
        out = self.Sblock3(out)
        out = self.Sblock3_S2(out)
        out = self.Sblock4(out)
        out = self.Sblock4_1(out)
        out = self.Sblock4_S2(out)
        out = self.Sblock5(out)
        out = self.Sblock5_1(out)
        out = self.Sblock5_S2(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        lmleye = self.lmleye(out)
        lmreye = self.lmreye(out)
        lmlbrow = self.lmlbrow(out)
        lmrbrow = self.lmrbrow(out)
        lmnose = self.lmnose(out)
        lmmouth = self.lmmouth(out)
        #return out
        return lmleye,lmreye,lmlbrow,lmrbrow,lmnose,lmmouth


class LMNet14d(nn.Module):
    def __init__(self):
        super(LMNet14d, self).__init__()
        self.pre_layers = nn.Sequential(
                    #nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2),
                    nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
                    #nn.BatchNorm2d(24),
                    nn.ReLU(),
                )
        self.Sblock1_S2 = SeparableConv2dS2(8,16)
        self.Sblock2_S2 = SeparableConv2dS2(16,32)
        self.Sblock3 = SeparableConv2d(32,32)
        self.Sblock3_S2 = SeparableConv2d5S2(32,64)
        self.Sblock4 = SeparableConv2d5(64,64)
        self.Sblock4_1 = SeparableConv2d5(64,64)
        self.Sblock4_S2 = SeparableConv2d5S2(64,128)
        self.Sblock5 = SeparableConv2d5(128,128)
        self.Sblock5_1 = SeparableConv2d5(128,128)
        self.Sblock5_S2 = SeparableConv2d7S2(128,256)
        self.Sblock6 = SeparableConv2d(256,256)
        self.classifier1 = nn.Sequential(
            nn.Linear(256*4*4, 256),
            nn.ReLU(),
        )
        self.lmleye = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )
        self.lmreye = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )
        self.lmlbrow = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )
        self.lmrbrow = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )
        self.lmnose = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )
        self.lmmouth = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 36),
        )

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.Sblock1_S2(out)
        out = self.Sblock2_S2(out)
        out = self.Sblock3(out)
        out = self.Sblock3_S2(out)
        out = self.Sblock4(out)
        out = self.Sblock4_1(out)
        out = self.Sblock4_S2(out)
        out = self.Sblock5(out)
        out = self.Sblock5_1(out)
        out = self.Sblock5_S2(out)
        out = self.Sblock6(out)
        out = out.view(out.size(0), -1)
        out = self.classifier1(out)
        lmleye = self.lmleye(out)
        lmreye = self.lmreye(out)
        lmlbrow = self.lmlbrow(out)
        lmrbrow = self.lmrbrow(out)
        lmnose = self.lmnose(out)
        lmmouth = self.lmmouth(out)
        #return out
        return lmleye,lmreye,lmlbrow,lmrbrow,lmnose,lmmouth


class LMNet14e(nn.Module):
    def __init__(self):
        super(LMNet14e, self).__init__()
        self.pre_layers = nn.Sequential(
                    #nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2),
                    nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
                    #nn.BatchNorm2d(24),
                    nn.ReLU(),
                )
        self.Sblock1_S2 = SeparableConv2dS2(8,16)
        self.Sblock2_S2 = SeparableConv2dS2(16,32)
        self.Sblock3 = SeparableConv2d(32,32)
        self.Sblock3_S2 = SeparableConv2d5S2(32,64)
        self.Sblock4 = SeparableConv2d5(64,64)
        self.Sblock4_1 = SeparableConv2d5(64,64)
        self.Sblock4_S2 = SeparableConv2d5S2(64,128)
        self.Sblock5 = SeparableConv2d5(128,128)
        self.Sblock5_1 = SeparableConv2d5(128,128)
        self.Sblock5_S2 = SeparableConv2d7S2(128,256)
        self.Sblock6 = SeparableConv2d(256,256)
        self.classifier1 = nn.Sequential(
            nn.Linear(256*4*4, 256),
            nn.ReLU(),
        )
        self.lmleye = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )
        self.lmreye = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )
        self.lmlbrow = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )
        self.lmrbrow = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )
        self.lmnose = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )
        self.lmmouth = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 36),
        )
        self.lmpupil = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )
    def forward(self, x):
        out = self.pre_layers(x)
        out = self.Sblock1_S2(out)
        out = self.Sblock2_S2(out)
        out = self.Sblock3(out)
        out = self.Sblock3_S2(out)
        out = self.Sblock4(out)
        out = self.Sblock4_1(out)
        out = self.Sblock4_S2(out)
        out = self.Sblock5(out)
        out = self.Sblock5_1(out)
        out = self.Sblock5_S2(out)
        out = self.Sblock6(out)
        out = out.view(out.size(0), -1)
        out = self.classifier1(out)
        lmleye = self.lmleye(out)
        lmreye = self.lmreye(out)
        lmlbrow = self.lmlbrow(out)
        lmrbrow = self.lmrbrow(out)
        lmnose = self.lmnose(out)
        lmmouth = self.lmmouth(out)
        lmpupil = self.lmpupil(out)
        #return out
        return lmleye,lmreye,lmlbrow,lmrbrow,lmnose,lmmouth,lmpupil


class LMNet15(nn.Module):
    def __init__(self):
        super(LMNet15, self).__init__()
        self.pre_layers = nn.Sequential(
                    #nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2),
                    nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
                    #nn.BatchNorm2d(24),
                    nn.ReLU(),
                )
        self.Sblock1_S2 = SeparableConv2dS2(8,16)
        self.Sblock2_S2 = SeparableConv2dS2(16,32)
        self.Sblock3 = SeparableConv2d(32,32)
        self.Sblock3_S2 = SeparableConv2d5S2(32,64)
        self.Sblock4 = SeparableConv2d5(64,64)
        self.Sblock4_1 = SeparableConv2d5(64,64)
        self.Sblock4_S2 = SeparableConv2d5S2(64,128)
        self.Sblock5 = SeparableConv2d5(128,128)
        self.Sblock5_1 = SeparableConv2d5(128,128)
        self.Sblock5_S2 = SeparableConv2d7S2(128,256)
        self.classifier = nn.Sequential(
            nn.Linear(256*4*4, 256),
            nn.ReLU(),
        )
        self.cls = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
        self.lmleye = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )
        self.lmreye = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )
        self.lmlbrow = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )
        self.lmrbrow = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )
        self.lmnose = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )
        self.lmmouth = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 36),
        )

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.Sblock1_S2(out)
        out = self.Sblock2_S2(out)
        out = self.Sblock3(out)
        out = self.Sblock3_S2(out)
        out = self.Sblock4(out)
        out = self.Sblock4_1(out)
        out = self.Sblock4_S2(out)
        out = self.Sblock5(out)
        out = self.Sblock5_1(out)
        out = self.Sblock5_S2(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        elecls= self.cls(out)
        lmleye = self.lmleye(out)
        lmreye = self.lmreye(out)
        lmlbrow = self.lmlbrow(out)
        lmrbrow = self.lmrbrow(out)
        lmnose = self.lmnose(out)
        lmmouth = self.lmmouth(out)
        #return out
        return elecls,lmleye,lmreye,lmlbrow,lmrbrow,lmnose,lmmouth




class LMNet15b(nn.Module):
    def __init__(self):
        super(LMNet15b, self).__init__()
        self.pre_layers = nn.Sequential(
                    #nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2),
                    nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
                    #nn.BatchNorm2d(24),
                    nn.ReLU(),
                )
        self.Sblock1_S2 = SeparableConv2dS2(8,16)
        self.Sblock2_S2 = SeparableConv2dS2(16,32)
        self.Sblock3 = SeparableConv2d(32,32)
        self.Sblock3_S2 = SeparableConv2d5S2(32,64)
        self.Sblock4 = SeparableConv2d5(64,64)
        self.Sblock4_1 = SeparableConv2d5(64,64)
        self.Sblock4_S2 = SeparableConv2d5S2(64,128)
        self.Sblock5 = SeparableConv2d5(128,128)
        self.Sblock5_1 = SeparableConv2d5(128,128)
        self.Sblock5_S2 = SeparableConv2d7S2(128,256)
        self.Sblock6 = SeparableConv2d(256,256)
        self.classifier = nn.Sequential(
            nn.Linear(256*4*4, 256),
            nn.ReLU(),
        )
        self.cls = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
        self.lmleye = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )
        self.lmreye = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )
        self.lmlbrow = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )
        self.lmrbrow = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )
        self.lmnose = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )
        self.lmmouth = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 36),
        )

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.Sblock1_S2(out)
        out = self.Sblock2_S2(out)
        out = self.Sblock3(out)
        out = self.Sblock3_S2(out)
        out = self.Sblock4(out)
        out = self.Sblock4_1(out)
        out = self.Sblock4_S2(out)
        out = self.Sblock5(out)
        out = self.Sblock5_1(out)
        out = self.Sblock5_S2(out)
        out = self.Sblock6(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        elecls= self.cls(out)
        lmleye = self.lmleye(out)
        lmreye = self.lmreye(out)
        lmlbrow = self.lmlbrow(out)
        lmrbrow = self.lmrbrow(out)
        lmnose = self.lmnose(out)
        lmmouth = self.lmmouth(out)
        #return out
        return elecls,lmleye,lmreye,lmlbrow,lmrbrow,lmnose,lmmouth



class LMNet15c(nn.Module):
    def __init__(self):
        super(LMNet15c, self).__init__()
        self.pre_layers = nn.Sequential(
                    #nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2),
                    nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
                    #nn.BatchNorm2d(24),
                    nn.ReLU(),
                )
        self.Sblock1_S2 = SeparableConv2dS2(8,16)
        self.Sblock2_S2 = SeparableConv2dS2(16,32)
        self.Sblock3 = SeparableConv2d(32,32)
        self.Sblock3_S2 = SeparableConv2d5S2(32,64)
        self.Sblock4 = SeparableConv2d5(64,64)
        self.Sblock4_1 = SeparableConv2d5(64,64)
        self.Sblock4_S2 = SeparableConv2d5S2(64,128)
        self.Sblock5 = SeparableConv2d5(128,128)
        self.Sblock5_1 = SeparableConv2d5(128,128)
        self.Sblock5_S2 = SeparableConv2d7S2(128,256)
        self.Sblock6 = SeparableConv2d(256,256)
        self.classifier2 = nn.Sequential(
            nn.Linear(256*2*4, 256),
            nn.ReLU(),
        )
        self.classifier1 = nn.Sequential(
            nn.Linear(256*4*4, 256),
            nn.ReLU(),
        )
        self.cls = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
        self.lmleye = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )
        self.lmreye = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )
        self.lmlbrow = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )
        self.lmrbrow = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )
        self.lmnose = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )
        self.lmmouth = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 36),
        )

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.Sblock1_S2(out)
        out = self.Sblock2_S2(out)
        out = self.Sblock3(out)
        out = self.Sblock3_S2(out)
        out = self.Sblock4(out)
        out = self.Sblock4_1(out)
        out = self.Sblock4_S2(out)
        out = self.Sblock5(out)
        out = self.Sblock5_1(out)
        out = self.Sblock5_S2(out)
        out1 = self.Sblock6(out)
        out2 = out1[:,:,0:2,:].contiguous()
        out1 = out1.view(out1.size(0), -1)
        out2 = out2.view(out2.size(0), -1)
        out1 = self.classifier1(out1)
        out2 = self.classifier2(out2)
        elecls= self.cls(out2)
        lmleye = self.lmleye(out1)
        lmreye = self.lmreye(out1)
        lmlbrow = self.lmlbrow(out1)
        lmrbrow = self.lmrbrow(out1)
        lmnose = self.lmnose(out1)
        lmmouth = self.lmmouth(out1)
        #return out1
        return elecls,lmleye,lmreye,lmlbrow,lmrbrow,lmnose,lmmouth

class SeparableConv2d3D(nn.Module):
    def __init__(self,in_planes,out_planes):
        super(SeparableConv2d3D,self).__init__()
        self.sblock= nn.Sequential(
            nn.Conv2d(in_planes,in_planes,3,1,2,groups=in_planes,dilation=2,bias=False),
            nn.Conv2d(in_planes,out_planes,1,1,0,bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )

    def forward(self,x):
        x = self.sblock(x)
        return x


class SeparableConv2d3D2(nn.Module):
    def __init__(self,in_planes,out_planes):
        super(SeparableConv2d3D2,self).__init__()
        self.sblock= nn.Sequential(
            nn.Conv2d(in_planes,in_planes,3,2,2,groups=in_planes,dilation=2,bias=False),
            nn.Conv2d(in_planes,out_planes,1,1,0,bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )

    def forward(self,x):
        x = self.sblock(x)
        return x



class SeparableConv2d5D(nn.Module):
    def __init__(self,in_planes,out_planes):
        super(SeparableConv2d5D,self).__init__()
        self.sblock= nn.Sequential(
            nn.Conv2d(in_planes,in_planes,5,1,2,groups=in_planes,dilation=2,bias=False),
            nn.Conv2d(in_planes,out_planes,1,1,0,bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )
    def forward(self,x):
        x = self.sblock(x)
        return x



class SeparableConv2d5D2(nn.Module):
    def __init__(self,in_planes,out_planes):
        super(SeparableConv2d5D2,self).__init__()
        self.sblock= nn.Sequential(
            nn.Conv2d(in_planes,in_planes,5,2,4,groups=in_planes,dilation=2,bias=False),
            nn.Conv2d(in_planes,out_planes,1,1,0,bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )
    def forward(self,x):
        x = self.sblock(x)
        return x

class LMNet15d(nn.Module):
    def __init__(self):
        super(LMNet15d, self).__init__()
        self.pre_layers = nn.Sequential(
                    #nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2),
                    nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
                    #nn.BatchNorm2d(24),
                    nn.ReLU(),
                )
        self.Sblock1_S2 = SeparableConv2dS2(8,16)
        self.Sblock2_S2 = SeparableConv2dS2(16,32)
        self.Sblock3 = SeparableConv2d(32,32)
        self.Sblock3_S2 = SeparableConv2dS2(32,64)
        self.Sblock4 = SeparableConv2d3D(64,64)
        self.Sblock4_1 = SeparableConv2d3D(64,64)
        self.Sblock4_S2 = SeparableConv2d5S2(64,128)
        self.Sblock5 = SeparableConv2d3D(128,128)
        self.Sblock5_1 = SeparableConv2d3D(128,128)
        self.Sblock5_S2 = SeparableConv2d7S2(128,256)
        self.Sblock6 = SeparableConv2d(256,256)
        self.classifier2 = nn.Sequential(
            nn.Linear(256*2*4, 256),
            nn.ReLU(),
        )
        self.classifier1 = nn.Sequential(
            nn.Linear(256*4*4, 256),
            nn.ReLU(),
        )
        self.cls = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
        self.lmleye = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )
        self.lmreye = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )
        self.lmlbrow = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )
        self.lmrbrow = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )
        self.lmnose = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )
        self.lmmouth = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 36),
        )

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.Sblock1_S2(out)
        out = self.Sblock2_S2(out)
        out = self.Sblock3(out)
        out = self.Sblock3_S2(out)
        out = self.Sblock4(out)
        out = self.Sblock4_1(out)
        out = self.Sblock4_S2(out)
        out = self.Sblock5(out)
        out = self.Sblock5_1(out)
        out = self.Sblock5_S2(out)
        out1 = self.Sblock6(out)
        out2 = out1[:,:,0:2,:].contiguous()
        out1 = out1.view(out1.size(0), -1)
        out2 = out2.view(out2.size(0), -1)
        out1 = self.classifier1(out1)
        out2 = self.classifier2(out2)
        elecls= self.cls(out2)
        lmleye = self.lmleye(out1)
        lmreye = self.lmreye(out1)
        lmlbrow = self.lmlbrow(out1)
        lmrbrow = self.lmrbrow(out1)
        lmnose = self.lmnose(out1)
        lmmouth = self.lmmouth(out1)
        #return out1
        return elecls,lmleye,lmreye,lmlbrow,lmrbrow,lmnose,lmmouth



def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % groups == 0)
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class DownsampleUnit(nn.Module):
    def __init__(self, inplanes, c_tag=0.5, activation=nn.ReLU, groups=2):
        super(DownsampleUnit, self).__init__()

        self.conv1r = nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False)
        self.bn1r = nn.BatchNorm2d(inplanes)
        self.conv2r = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1, bias=False, groups=inplanes)
        self.bn2r = nn.BatchNorm2d(inplanes)
        self.conv3r = nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False)
        self.bn3r = nn.BatchNorm2d(inplanes)

        self.conv1l = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1, bias=False, groups=inplanes)
        self.bn1l = nn.BatchNorm2d(inplanes)
        self.conv2l = nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False)
        self.bn2l = nn.BatchNorm2d(inplanes)
        self.activation = activation(inplace=True)

        self.groups = groups
        self.inplanes = inplanes

    def forward(self, x):
        out_r = self.conv1r(x)
        out_r = self.bn1r(out_r)
        out_r = self.activation(out_r)

        out_r = self.conv2r(out_r)
        out_r = self.bn2r(out_r)

        out_r = self.conv3r(out_r)
        out_r = self.bn3r(out_r)
        out_r = self.activation(out_r)

        out_l = self.conv1l(x)
        out_l = self.bn1l(out_l)

        out_l = self.conv2l(out_l)
        out_l = self.bn2l(out_l)
        out_l = self.activation(out_l)
        
        #return channel_shuffle(torch.cat((out_r, out_l), 1), self.groups)
        return torch.cat((out_r, out_l),1)


class BasicUnit(nn.Module):
    def __init__(self, inplanes, outplanes, c_tag=0.5, activation=nn.ReLU, SE=False, residual=False, groups=2):
        super(BasicUnit, self).__init__()
        self.left_part = int(round(c_tag * inplanes))
        self.right_part_in = inplanes - self.left_part
        self.right_part_out = outplanes - self.left_part
        self.conv1 = nn.Conv2d(self.right_part_in, self.right_part_out, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.right_part_out)
        self.conv2 = nn.Conv2d(self.right_part_out, self.right_part_out, kernel_size=3, padding=1, bias=False,
                               groups=self.right_part_out)
        self.bn2 = nn.BatchNorm2d(self.right_part_out)
        self.conv3 = nn.Conv2d(self.right_part_out, self.right_part_out, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.right_part_out)
        self.activation = activation(inplace=True)

        self.inplanes = inplanes
        self.outplanes = outplanes
        self.residual = residual
        self.groups = groups
        self.SE = SE
        if self.SE:
            self.SELayer = SELayer(self.right_part_out, 2)  # TODO

    def forward(self, x):
        left = x[:, :self.left_part, :, :]
        right = x[:, self.left_part:, :, :]
        out = self.conv1(right)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.activation(out)

        if self.SE:
            out = self.SELayer(out)
        if self.residual and self.inplanes == self.outplanes:
            out += right
        #return channel_shuffle(torch.cat((left, out), 1), self.groups)
        return torch.cat((left, out),1)



class LMNet16(nn.Module):
    def __init__(self):
        super(LMNet16, self).__init__()
        self.pre_layers = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(16),
                    nn.MaxPool2d(kernel_size=3, stride=2)
                )
        self.Downsample1  =DownsampleUnit(16)
        self.BasicUnit1_1 =BasicUnit(32,32)
        self.BasicUnit1_2 =BasicUnit(32,32)
        self.Downsample2  =DownsampleUnit(32)
        self.BasicUnit2_1 =BasicUnit(64,64)
        self.BasicUnit2_2 =BasicUnit(64,64)
        self.Downsample3  =DownsampleUnit(64)
        self.BasicUnit3_1 =BasicUnit(128,128)
        self.BasicUnit3_2 =BasicUnit(128,128)
        self.Downsample4  =DownsampleUnit(128)
        self.BasicUnit4_1 =BasicUnit(256,256)
        self.BasicUnit4_2 =BasicUnit(256,256)
        #self.avgpool      =nn.AdaptiveAvgPool2d(1)  
        #self.avgpool      =nn.AvgPool2d(4,4)  
        self.classifier2 = nn.Sequential(
            nn.Linear(256*2*4, 256),
            nn.ReLU(),
        )
        self.classifier1 = nn.Sequential(
            nn.Linear(256*4*4, 256),
            nn.ReLU(),
        )
        self.cls = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
        self.lmleye = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )
        self.lmreye = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )
        self.lmlbrow = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )
        self.lmrbrow = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )
        self.lmnose = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )
        self.lmmouth = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 36),
        )
        
    def forward(self, x):
        out = self.pre_layers(x)
        out = self.Downsample1(out)
        out = self.BasicUnit1_1(out)
        out = self.BasicUnit1_2(out)
        out = self.Downsample2(out)
        out = self.BasicUnit2_1(out)
        out = self.BasicUnit2_2(out)
        out = self.Downsample3(out)
        out = self.BasicUnit3_1(out)
        out = self.BasicUnit3_2(out)
        out = self.Downsample4(out)
        out = self.BasicUnit4_1(out)
        out1 = self.BasicUnit4_2(out)
        #out = self.avgpool(out)
        #out = out.view(out.size(0,-1))
        out2 = out1[:,:,0:2,:].contiguous()
        out1 = out1.view(out1.size(0), -1)
        out2 = out2.view(out2.size(0), -1)
        out1 = self.classifier1(out1)
        out2 = self.classifier2(out2)
        elecls= self.cls(out2)
        lmleye = self.lmleye(out1)
        lmreye = self.lmreye(out1)
        lmlbrow = self.lmlbrow(out1)
        lmrbrow = self.lmrbrow(out1)
        lmnose = self.lmnose(out1)
        lmmouth = self.lmmouth(out1)
        #return out
        return elecls,lmleye,lmreye,lmlbrow,lmrbrow,lmnose,lmmouth
        


class LMNet16L(nn.Module):
    def __init__(self):
        super(LMNet16L, self).__init__()
        self.pre_layers = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(16),
                    nn.MaxPool2d(kernel_size=3, stride=2)
                )
        self.Downsample1  =DownsampleUnit(16)
        self.BasicUnit1_1 =BasicUnit(32,32)
        self.BasicUnit1_2 =BasicUnit(32,32)
        self.Downsample2  =DownsampleUnit(32)
        self.BasicUnit2_1 =BasicUnit(64,64)
        self.BasicUnit2_2 =BasicUnit(64,64)
        self.Downsample3  =DownsampleUnit(64)
        self.BasicUnit3_1 =BasicUnit(128,128)
        self.BasicUnit3_2 =BasicUnit(128,128)
        self.Downsample4  =DownsampleUnit(128)
        self.BasicUnit4_1 =BasicUnit(256,256)
        self.BasicUnit4_2 =BasicUnit(256,256)
        self.classifier1 = nn.Sequential(
            nn.Linear(256*4*4, 256),
            nn.ReLU(),
        )
        self.cls = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
        self.lmleye = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )
        self.lmreye = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )
        self.lmlbrow = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )
        self.lmrbrow = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )
        self.lmnose = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )
        self.lmmouth = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 36),
        )
        
    def forward(self, x):
        out = self.pre_layers(x)
        out = self.Downsample1(out)
        out = self.BasicUnit1_1(out)
        out = self.BasicUnit1_2(out)
        out = self.Downsample2(out)
        out = self.BasicUnit2_1(out)
        out = self.BasicUnit2_2(out)
        out = self.Downsample3(out)
        out = self.BasicUnit3_1(out)
        out = self.BasicUnit3_2(out)
        out = self.Downsample4(out)
        out = self.BasicUnit4_1(out)
        out = self.BasicUnit4_2(out)
        #out = self.avgpool(out)
        #out = out.view(out.size(0,-1))
        out = out.view(out.size(0), -1)
        out = self.classifier1(out)
        lmleye = self.lmleye(out)
        lmreye = self.lmreye(out)
        lmlbrow = self.lmlbrow(out)
        lmrbrow = self.lmrbrow(out)
        lmnose = self.lmnose(out)
        lmmouth = self.lmmouth(out)
        #return out
        return lmleye,lmreye,lmlbrow,lmrbrow,lmnose,lmmouth
        


if __name__ == '__main__':
    input= Variable(torch.randn((2,1,256,256)))
    #input= Variable(torch.randn((1,1,96,96)))
    #net= LMNet10()
    #net= LMNet12()
    #net= LMNet12b()
    #net= LMNet13a()
    #net= LMNet14d()
    #net= LMNet15c()
    net= LMNet16()
    #net.load_state_dict(torch.load('/home/ficha/Documents/riku/lm/modelsaved/LMNet8/LMNet8a_30.pth'))
    #net.load_state_dict(torch.load('/home/training-pc/Documents/lm/modelsaved/LMNet13/LMNet13b_141.pth'))

    #print(input)
    print(net)
    raw_input('pause')
    net= add_flops_counting_methods(net)
    net.start_flops_count()
    output= net(input)
    cost=net.compute_average_flops_cost() / 1e6 / 2
    #print(output.size())
    print(output)
    print(cost)

    raw_input('pause')
    net= net.cuda()
    #input=input.cuda()
    #print(net)
    #raw_input('pause')
    #summary(net,(1,96,96))
    summary(net,(1,256,256))
