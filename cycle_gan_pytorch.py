import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim
from layer import Generator,Disctiminator
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import os
from glob import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


#========================================================
#モデル構築

discriminatorA=Disctiminator()
discriminatorB=Disctiminator()
generatorAB=Generator()
generatorBA=Generator()

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
#dataset用意
def load_data(dataset_name,change_to_256=False):
    if not os.path.exists("datasets/"+dataset_name+"/numpy_data"):
        img_list_A = glob('datasets/'+dataset_name+'/trainA/*.' + "jpg")
        img_list_B = glob('datasets/'+dataset_name+'/trainB/*.' + "jpg")
        trainA=[]
        trainB=[]
        for img in img_list_A:
            trainA_img = load_img(img,grayscale=False)
            if change_to_256:
                trainA_img=trainA_img.resize((256,256))
            trainA_img = np.uint8(trainA_img)
            trainA_img = trainA_img /255
            trainA.append(trainA_img)
        for img in img_list_B:
            trainB_img = load_img(img,grayscale=False)
            if change_to_256:
                trainB_img=trainB_img.resize((256,256))
            trainB_img = np.uint8(trainB_img)
            trainB_img = trainB_img /255
            trainB.append(trainB_img)
        trainA=np.array(trainA)
        trainB=np.array(trainB)
        os.mkdir("datasets/"+dataset_name+"/numpy_data")
        name=dataset_name.split("2")
        np.save("datasets/"+dataset_name+"/numpy_data/"+name[0]+"_numpy",trainA)
        np.save("datasets/"+dataset_name+"/numpy_data/"+name[1]+"_numpy",trainB)
    else:
      name=dataset_name.split("2")
      trainA=np.load("datasets/"+dataset_name+"/numpy_data/"+name[0]+"_numpy.npy")
      trainB=np.load("datasets/"+dataset_name+"/numpy_data/"+name[1]+"_numpy.npy")
    sampleA=trainA[100]
    sampleB=trainB[0]
    trainA=torch.Tensor(trainA)
    trainB=torch.Tensor(trainB)
    trainA=trainA.permute(0,3,1,2)
    trainB=trainB.permute(0,3,1,2)
    return trainA,trainB,sampleA,sampleB

class Data(torch.utils.data.Dataset):
    def __init__(self,A,B):
        super(Data,self).__init__()
        self.A=A
        self.B=B
        self.len=min(A.size()[0],B.size()[0])
    def __len__(self):
        return self.len
    def __getitem__(self,index):
        return self.A[index],self.B[index]

trainA,trainB,sampleA,sampleB=load_data("horse2zebra",True)
dataset = Data(trainA*2-1,trainB*-1)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=True)

#========================================================
# # 出力画像表示
def save(epoch):
    # generatorAB.to("cpu")
    # generatorBA.to("cpu")
    plt.imsave("output/trueA/epoch_"+str(epoch)+".png",sampleA)
    plt.imsave("output/fakeB/epoch_"+str(epoch)+".png",numpy2tensor2numpy(sampleA))
    plt.imsave("output/trueB/epoch_"+str(epoch)+".png",sampleB)
    plt.imsave("output/fakeA/epoch_"+str(epoch)+".png",numpy2tensor2numpy(sampleB))
    # generatorAB.cuda()
    # generatorBA.cuda()
def numpy2tensor2numpy(numpy):
    tensor=torch.Tensor(numpy)
    tensor=(tensor*2-1).permute(2,0,1)
    tensor=generatorAB(torch.reshape(tensor,(1,3,256,256)))
    tensor=torch.reshape(tensor,(3,256,256)).permute(1,2,0)
    tensor=(tensor+1)/2
    numpy=tensor.detach().numpy()
    return numpy




#========================================================

#学習

epochs=int(input("epoch:"))
epoch_x=[]
g_loss_y=[]
d_loss_y=[]

for epoch in range(epochs):
    print("epoch:",epoch)
    save(epoch)

    loss_d_sum=0
    loss_g_sum=0
    i=0
    for trainA,trainB in trainloader:
        if i%10==9:
            print("■",end="")
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
#========================================================
# ロスグラフ出力

plt.plot(epoch_x,g_loss_y,label='g_loss')
plt.plot(epoch_x,d_loss_y,label='d_loss')
plt.legend()
plt.savefig('output/figure.png')
