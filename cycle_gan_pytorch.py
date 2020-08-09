import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim
from layer import Generator,Disctiminator
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from tools import *




#========================================================
#モデル構築

discriminatorA=Disctiminator()
discriminatorB=Disctiminator()
generatorAB=Generator()
generatorBA=Generator()
if torch.cuda.is_available():
    discriminatorA.cuda()
    discriminatorB.cuda()
    generatorAB.cuda()
    generatorBA.cuda()

#ハイパーパラメータ
adam_lr=0.0002
adam_beta=(0.5,0.999)
cyc=10

#========================================================
#ロス定義

#adversarial_loss
criterion_ad_A = nn.BCELoss()
criterion_ad_B = nn.BCELoss()

#cycle_consistancy_loss
criterion_cyc_A = nn.L1Loss()
criterion_cyc_B = nn.L1Loss()

#identity_loss
criterion_id_A = nn.L1Loss()
criterion_id_B = nn.L1Loss()


#========================================================
#最適化関数定義

optimizer_dA = optim.Adam(discriminatorA.parameters(),lr=adam_lr,betas=adam_beta)
optimizer_dB = optim.Adam(discriminatorB.parameters(),lr=adam_lr,betas=adam_beta)
optimizer_gAB = optim.Adam(generatorAB.parameters(),lr=adam_lr,betas=adam_beta)
optimizer_gBA = optim.Adam(generatorBA.parameters(),lr=adam_lr,betas=adam_beta)

#========================================================
#discriminatorの訓練

def d_train(trainA,trainB):
    fakeB=generatorAB(trainA)
    fakeA=generatorBA(trainB)
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
    return loss_d

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

    loss_ad_A=criterion_ad_A(result_fakeA,torch.zeros_like(result_fakeA))
    loss_ad_B=criterion_ad_B(result_fakeB,torch.zeros_like(result_fakeB))
    ad_loss=loss_ad_A+loss_ad_B
    cycle_lossA=criterion_cyc_A(reconstA,trainA)
    cycle_lossB=criterion_cyc_B(reconstB,trainB)
    cycle_loss=cycle_lossA+cycle_lossB

    loss_g=ad_loss+cyc*cycle_loss
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
    return loss_g

#========================================================
#datasetloader構築

trainA,trainB=load_data("horse2zebra",True)
dataset = Data(trainA*2-1,trainB*2-1)
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
        loss_g=g_train(trainA,trainB)
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
