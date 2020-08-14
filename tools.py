import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import os
from glob import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import time


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
    trainA=torch.Tensor(trainA)
    trainB=torch.Tensor(trainB)
    trainA=trainA.permute(0,3,1,2)
    trainB=trainB.permute(0,3,1,2)
    return trainA,trainB

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

def sample_data():#正しい
    sampleA_path="output/sample/sampleA.png"
    sampleA_img = load_img(sampleA_path,grayscale=False)
    sampleA_img=sampleA_img.resize((256,256))
    sampleA_img = np.uint8(sampleA_img)
    sampleA_img = sampleA_img /255
    sampleB_path="output/sample/sampleB.png"
    sampleB_img = load_img(sampleB_path,grayscale=False)
    sampleB_img=sampleB_img.resize((256,256))
    sampleB_img = np.uint8(sampleB_img)
    sampleB_img = sampleB_img /255
    return sampleA_img,sampleB_img

#========================================================
# 出力画像表示
def save(epoch,generatorAB,generatorBA):
    sampleA,sampleB=sample_data()
    if torch.cuda.is_available():
        generatorAB.to("cpu")
        generatorBA.to("cpu")
    plt.imsave("output/trueA/epoch_"+str(epoch)+".png",sampleA)
    sampleA=sampleA*2-1
    sampleA=numpy2tensor(sampleA)
    sampleA=sampleA.reshape((1,3,256,256))
    fakeB=generatorAB(sampleA)
    fakeB=fakeB.reshape((3,256,256))
    fakeB=tensor2numpy(fakeB)
    fakeB=(fakeB+1)/2
    plt.imsave("output/fakeB/epoch_"+str(epoch)+".png",fakeB)
    plt.imsave("output/trueB/epoch_"+str(epoch)+".png",sampleB)
    sampleB=sampleB*2-1
    sampleB=numpy2tensor(sampleB)
    sampleB=sampleB.reshape((1,3,256,256))
    fakeA=generatorBA(sampleB)
    fakeA=fakeA.reshape((3,256,256))
    fakeA=tensor2numpy(fakeA)
    fakeA=(fakeA+1)/2
    plt.imsave("output/fakeA/epoch_"+str(epoch)+".png",fakeA)
    if torch.cuda.is_available():
        generatorAB.cuda()
        generatorBA.cuda()

def numpy2tensor(numpy):
    tensor=torch.Tensor(numpy)
    tensor=tensor.permute(2,0,1)
    return tensor
def tensor2numpy(tensor):
    tensor=tensor.permute(1,2,0)
    numpy=tensor.detach().numpy()
    return numpy

#========================================================
#進捗表示

def progress(p, l):
    sys.stdout.write("\rprocessing : %d %%" %(int(p * 100 / (l - 1))))
    sys.stdout.flush()



class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.images = []

    def query(self, images):
        images.to("cpu")
        if len(self.images)<50:
            self.images.append(images.detach())
        else:
            del self.images[0]
            self.images.append(images.detach())
        return torch.cat(self.images, 0).cuda()
