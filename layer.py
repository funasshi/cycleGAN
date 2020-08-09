import torch
import torch.nn as nn
import torch.nn.functional as F

class Resblock(nn.Module):
    def __init__(self):
        super(Resblock,self).__init__()
        self.conv_res_1=nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.ins_norm_1=nn.InstanceNorm2d(256)
        self.conv_res_2=nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.ins_norm_2=nn.InstanceNorm2d(256)

    def forward(self,x):
        y=self.conv_res_1(x)
        y=F.relu(self.ins_norm_1(y))
        y=self.conv_res_2(y)
        y=self.ins_norm_2(y)
        out=torch.add(x,y)
        return out


class Res9block(nn.Module):
    def __init__(self):
        super(Res9block,self).__init__()
        self.res_1=Resblock()
        self.res_2=Resblock()
        self.res_3=Resblock()
        self.res_4=Resblock()
        self.res_5=Resblock()
        self.res_6=Resblock()
        self.res_7=Resblock()
        self.res_8=Resblock()
        self.res_9=Resblock()
        self.res_all=[self.res_1,self.res_2,self.res_3,self.res_4,self.res_5,self.res_6,self.res_7,self.res_8,self.res_9]

    def forward(self,x):
        for res in self.res_all:
            x=res(x)
        return x

class G_Downsample(nn.Module):
    def __init__(self):
        super(G_Downsample,self).__init__()
        self.conv_down_1=nn.Conv2d(3,64,kernel_size=7,stride=1,padding=3)
        self.ins_norm_1=nn.InstanceNorm2d(64)
        self.conv_down_2=nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1)
        self.ins_norm_2=nn.InstanceNorm2d(128)
        self.conv_down_3=nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1)
        self.ins_norm_3=nn.InstanceNorm2d(256)
    def forward(self,x):
        x=self.conv_down_1(x)
        x=F.relu(self.ins_norm_1(x))
        x=self.conv_down_2(x)
        x=F.relu(self.ins_norm_2(x))
        x=self.conv_down_3(x)
        x=F.relu(self.ins_norm_3(x))
        return x

class G_Upsample(nn.Module):
    def __init__(self):
        super(G_Upsample,self).__init__()
        self.conv_up_1=nn.ConvTranspose2d(256,128,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.ins_norm_1=nn.InstanceNorm2d(128)
        self.conv_up_2=nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.ins_norm_2=nn.InstanceNorm2d(64)
        self.conv_up_3=nn.Conv2d(64,3,kernel_size=7,stride=1,padding=3)

    def forward(self,x):
        x=self.conv_up_1(x)
        x=F.relu(self.ins_norm_1(x))
        x=self.conv_up_2(x)
        x=F.relu(self.ins_norm_2(x))
        x=torch.tanh(self.conv_up_3(x))
        return x



class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.down_sample=G_Downsample()
        self.res_block=Res9block()
        self.up_sample=G_Upsample()

    def forward(self,x):
        x=self.down_sample(x)
        x=self.res_block(x)
        x=self.up_sample(x)
        return x



class Disctiminator(nn.Module):
    def __init__(self):
        super(Disctiminator,self).__init__()
        self.conv_1=nn.Conv2d(3,64,kernel_size=4,stride=2,padding=1)
        self.ins_norm_1=nn.InstanceNorm2d(64)
        self.conv_2=nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1)
        self.ins_norm_2=nn.InstanceNorm2d(128)
        self.conv_3=nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1)
        self.ins_norm_3=nn.InstanceNorm2d(256)
        self.conv_4=nn.Conv2d(256,512,kernel_size=4,stride=2,padding=1)
        self.ins_norm_4=nn.InstanceNorm2d(512)
        self.conv_5=nn.Conv2d(512,1,kernel_size=3,stride=1,padding=1)

    def forward(self,x):
        x=self.conv_1(x)
        x=F.leaky_relu(self.ins_norm_1(x),negative_slope=0.2)
        x=self.conv_2(x)
        x=F.leaky_relu(self.ins_norm_2(x),negative_slope=0.2)
        x=self.conv_3(x)
        x=F.leaky_relu(self.ins_norm_3(x),negative_slope=0.2)
        x=self.conv_4(x)
        x=F.leaky_relu(self.ins_norm_4(x),negative_slope=0.2)
        x=torch.sigmoid(self.conv_5(x))
        return x


    
