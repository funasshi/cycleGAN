import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim
from layer import Generator,Discriminator
import numpy as np
import matplotlib.pyplot as plt
from tools import *


#========================================================
#モデル構築

discriminatorA=Discriminator()
discriminatorB=Discriminator()
generatorAB=Generator()
generatorBA=Generator()
if torch.cuda.is_available():
    discriminatorA.cuda()
    discriminatorB.cuda()
    generatorAB.cuda()
    generatorBA.cuda()



#========================================================
#ハイパーパラメータ
adam_lr_g=0.0002
adam_lr_d=0.0001
adam_beta=(0.5,0.999)
cyc=10
id=0.5*cyc
stddev=0.02**0.5
#========================================================
# #重みの初期化
# def init_weights(m):
#     nn.init.normal_(m.weight, 0.0, stddev)
# discriminatorA.weight
# nn.init.normal_(discriminatorA.weight, 0.0, stddev)
# summary(generatorAB,(3,256,256))
#
# generatorAB.apply(init_weights)
# generatorBA.apply(init_weights)
# discriminatorA.apply(init_weights)
# discriminatorB.apply(init_weights)
#

#========================================================
#ロス定義

#adversarial_loss
criterion_ad_A = nn.MSELoss()
criterion_ad_B = nn.MSELoss()

#cycle_consistancy_loss
criterion_cyc_A = nn.L1Loss()
criterion_cyc_B = nn.L1Loss()

#identity_loss
criterion_id_A = nn.L1Loss()
criterion_id_B = nn.L1Loss()


#========================================================
#最適化関数定義

optimizer_dA = optim.Adam(discriminatorA.parameters(),lr=adam_lr_d,betas=adam_beta)
optimizer_dB = optim.Adam(discriminatorB.parameters(),lr=adam_lr_d,betas=adam_beta)
optimizer_gAB = optim.Adam(generatorAB.parameters(),lr=adam_lr_g,betas=adam_beta)
optimizer_gBA = optim.Adam(generatorBA.parameters(),lr=adam_lr_g,betas=adam_beta)

#========================================================
#pool
poolA=ImagePool(50)
poolB=ImagePool(50)



#========================================================
#discriminatorの訓練

def d_train(trainA,trainB):
    fakeB=generatorAB(trainA)
    fakeA=generatorBA(trainB)
    fakeA=poolA.query(fakeA)
    fakeB=poolB.query(fakeB)
    result_trueA=discriminatorA(trainA)
    result_trueB=discriminatorB(trainB)
    result_fakeA=discriminatorA(fakeA)
    result_fakeB=discriminatorB(fakeB)
    loss_ad_A=(criterion_ad_A(result_trueA,torch.ones_like(result_trueA))+criterion_ad_A(result_fakeA,torch.zeros_like(result_fakeA)))/2
    loss_ad_B=(criterion_ad_B(result_trueB,torch.ones_like(result_trueB))+criterion_ad_B(result_fakeB,torch.zeros_like(result_fakeB)))/2

    loss_d=loss_ad_A+loss_ad_B

    optimizer_dA.zero_grad()
    optimizer_dB.zero_grad()
    optimizer_gAB.zero_grad()
    optimizer_gBA.zero_grad()

    loss_d.backward()
    optimizer_gAB.zero_grad()
    optimizer_gBA.zero_grad()
    optimizer_gAB.step()
    optimizer_gBA.step()
    optimizer_dA.step()
    optimizer_dB.step()
    return loss_d.item()

#========================================================
#generatorの訓練

def g_train(trainA,trainB):
    fakeB=generatorAB(trainA)
    fakeA=generatorBA(trainB)
    reconstA=generatorBA(fakeB)
    reconstB=generatorAB(fakeA)
    identityA=generatorBA(trainA)
    identityB=generatorAB(trainB)


    result_fakeA=discriminatorA(fakeA)
    result_fakeB=discriminatorB(fakeB)

    loss_ad_A=criterion_ad_A(result_fakeA,torch.ones_like(result_fakeA))
    loss_ad_B=criterion_ad_B(result_fakeB,torch.ones_like(result_fakeB))
    ad_loss=loss_ad_A+loss_ad_B
    cycle_lossA=criterion_cyc_A(reconstA,trainA)
    cycle_lossB=criterion_cyc_B(reconstB,trainB)
    cycle_loss=cycle_lossA+cycle_lossB
    identity_lossA=criterion_id_A(identityA,trainA)
    identity_lossB=criterion_id_B(identityB,trainB)
    identity_loss=identity_lossA+identity_lossB

    loss_g=ad_loss+cyc*cycle_loss+id*identity_loss
    optimizer_dA.zero_grad()
    optimizer_dB.zero_grad()
    optimizer_gAB.zero_grad()
    optimizer_gBA.zero_grad()
    loss_g.backward()
    optimizer_dA.zero_grad()
    optimizer_dB.zero_grad()
    optimizer_gAB.step()
    optimizer_gBA.step()
    optimizer_dA.step()
    optimizer_dB.step()
    return loss_g.item()

#========================================================
#datasetloader構築

trainAs,trainBs=load_data("horse2zebra",True)
trainAs=trainAs*2-1
trainBs=trainBs*2-1
dataset = Data(trainAs,trainBs)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=True)

#========================================================
#学習
epochs=int(input("epoch:"))
epoch_x=[]
g_loss_y=[]
d_loss_y=[]

for epoch in range(epochs):
    print("epoch:",epoch)
    loss_d_sum=0
    loss_g_sum=0
    i=0
    for trainA,trainB in trainloader:
        progress(i,len(trainloader))
        if torch.cuda.is_available():
            trainA=trainA.cuda()
            trainB=trainB.cuda()
        loss_d=d_train(trainA,trainB)
        loss_g_1=g_train(trainA,trainB)
        loss_g_2=g_train(trainA,trainB)
        loss_g=(loss_g_1+loss_g_2)/2
        i+=1
        loss_d_sum+=loss_d
        loss_g_sum+=loss_g
    print("")
    loss_d_sum/=i
    loss_g_sum/=i
    g_loss_y.append(loss_g_sum)
    d_loss_y.append(loss_d_sum)
    epoch_x.append(epoch)
    save(epoch,generatorAB,generatorBA)

#========================================================
# ロスグラフ出力

plt.plot(epoch_x,g_loss_y,label='g_loss')
plt.plot(epoch_x,d_loss_y,label='d_loss')
plt.legend()
plt.savefig('output/figure.png')
