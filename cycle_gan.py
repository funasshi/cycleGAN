import scipy
import numpy as np
import os
from glob import glob
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import tensorflow as tf
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input,Dense,Reshape,Flatten,Dropout,Concatenate,Add
from tensorflow.keras.layers import BatchNormalization,Activation,ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU,ReLU
from tensorflow.keras.layers import UpSampling2D,Conv2D,Conv2DTranspose,AveragePooling1D
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
import datetime
import matplotlib.pyplot as plt
import sys
from tensorflow_addons.layers import InstanceNormalization
from PIL import Image

class CycleGAN:
    def __init__(self):
        self.img_rows=128
        self.img_cols=128
        self.channels=3
        self.img_shape=(self.img_rows,self.img_cols,self.channels)
        self.patch_rows=self.img_rows//16
        self.patch_cols=self.img_cols//16
        self.df=64
        self.gf=32
        self.lambda_cycle=10.0
        self.lambda_id=0.9*self.lambda_cycle
        optimizer=Adam(0.0002,0.5)

        self.d_A=self.build_discriminator()
        self.d_B=self.build_discriminator()
        self.d_A.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=["accuracy"])
        self.d_B.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=["accuracy"])
        self.g_AB=self.build_generator()
        self.g_BA=self.build_generator()
        img_A=Input(shape=self.img_shape)
        img_B=Input(shape=self.img_shape)
        fake_B=self.g_AB(img_A)
        fake_A=self.g_BA(img_B)
        reconstr_A=self.g_BA(fake_B)
        reconstr_B=self.g_AB(fake_A)
        img_A_id=self.g_BA(img_A)
        img_B_id=self.g_AB(img_B)
        self.d_A.trainable=False
        self.d_B.trainable=False
        valid_A=self.d_A(fake_A)
        valid_B=self.d_B(fake_B)
        self.combined=Model(inputs=[img_A,img_B],outputs=[valid_A,valid_B,reconstr_A,reconstr_B,img_A_id,img_B_id])
        self.combined.compile(loss=["binary_crossentropy","binary_crossentropy","mae","mae","mae","mae"],loss_weights=[1,1,self.lambda_cycle,self.lambda_cycle,self.lambda_id,self.lambda_id],optimizer=optimizer)



    def build_generator(self):
        img=Input(shape=self.img_shape)
        x=Conv2D(filters=64,kernel_size=7,strides=1,padding="same",kernel_initializer=RandomNormal(0,0.02))(img)
        x=InstanceNormalization()(x)
        x=active(x,"ReLU")
        x=Conv2D(filters=128,kernel_size=3,strides=2,padding="same",kernel_initializer=RandomNormal(0,0.02))(x)
        x=InstanceNormalization()(x)
        x=active(x,"ReLU")
        x=Conv2D(filters=256,kernel_size=3,strides=2,padding="same",kernel_initializer=RandomNormal(0,0.02))(x)
        x=InstanceNormalization()(x)
        x=active(x,"ReLU")
        for i in range(6):
            x=resblock(x)
        x=Conv2DTranspose(filters=128,kernel_size=3,strides=2,padding="same",kernel_initializer=RandomNormal(0,0.02))(x)
        x=InstanceNormalization()(x)
        x=active(x,"ReLU")
        x=Conv2DTranspose(filters=64,kernel_size=3,strides=2,padding="same",kernel_initializer=RandomNormal(0,0.02))(x)
        x=InstanceNormalization()(x)
        x=active(x,"ReLU")
        x=Conv2D(filters=3,kernel_size=7,strides=1,padding="same",kernel_initializer=RandomNormal(0,0.02))(x)
        x=InstanceNormalization()(x)
        x=active(x,"ReLU")
        return Model(img,x)



    def build_discriminator(self):

        img=Input(shape=self.img_shape)
        x=Conv2D(filters=64,kernel_size=4,strides=2,padding="same",kernel_initializer=RandomNormal(0,0.02))(img)
        x=active(x,"LeakyReLU")
        x=Conv2D(filters=128,kernel_size=4,strides=2,padding="same",kernel_initializer=RandomNormal(0,0.02))(x)
        x=InstanceNormalization()(x)
        x=active(x,"LeakyReLU")
        x=Conv2D(filters=256,kernel_size=4,strides=2,padding="same",kernel_initializer=RandomNormal(0,0.02))(x)
        x=InstanceNormalization()(x)
        x=active(x,"LeakyReLU")
        x=Conv2D(filters=512,kernel_size=4,strides=2,padding="same",kernel_initializer=RandomNormal(0,0.02))(x)
        x=InstanceNormalization()(x)
        x=active(x,"LeakyReLU")
        x=Conv2D(filters=1,kernel_size=4,strides=1,padding="same",kernel_initializer=RandomNormal(0,0.02))(x)
        # x=Reshape(target_shape=(x.shape[1]*x.shape[2]*x.shape[3],1))(x)
        # x=AveragePooling1D(pool_size=x.shape[1])(x)
        # x=Flatten()(x)

        return Model(img,x)

    def train(self,trainA,trainB,epochs,batch_size=1,sample_interval=50):
        valid=np.ones((batch_size,self.patch_rows,self.patch_cols,1))
        fake=np.zeros((batch_size,self.patch_rows,self.patch_cols,1))

        for epoch in range(epochs):
            print("epoch:"+str(epoch))
            if epoch%10==9:
                imgs_A=trainA[np.random.randint(0,trainA.shape[0],size=1)]
                fake_B=self.g_AB.predict(imgs_A)
                plt.imshow( imgs_A.reshape(128,128,3) )
                plt.show()
                plt.imshow( fake_B.reshape(128,128,3) )
                plt.show()
                imgs_B=trainB[np.random.randint(0,trainA.shape[0],size=1)]
                fake_A=self.g_BA.predict(imgs_B)
                plt.imshow( imgs_B.reshape(128,128,3) )
                plt.show()
                plt.imshow( fake_A.reshape(128,128,3) )
                plt.show()
            for batch_i in range(995//batch_size):
                imgs_A=trainA[np.random.randint(0,trainA.shape[0],size=batch_size)]
                imgs_B=trainB[np.random.randint(0,trainB.shape[0],size=batch_size)]
                fake_B=self.g_AB.predict(imgs_A)
                fake_A=self.g_BA.predict(imgs_B)

                dA_loss_real=self.d_A.train_on_batch(imgs_A,valid)
                dA_loss_fake=self.d_A.train_on_batch(fake_A,fake)
                dA_loss=0.5*np.add(dA_loss_real,dA_loss_fake)
                dB_loss_real=self.d_B.train_on_batch(imgs_B,valid)
                dB_loss_fake=self.d_B.train_on_batch(fake_B,fake)
                dB_loss=0.5*np.add(dB_loss_real,dB_loss_fake)
                d_loss=0.5*np.add(dA_loss,dB_loss)
                g_loss=self.combined.train_on_batch([imgs_A,imgs_B],[valid,valid,imgs_A,imgs_B,imgs_A,imgs_B])

def load_data(dataset_name,from_origin=False):
    if from_origin:
        img_list_A = glob('drive/My Drive/GAN/datasets/'+dataset_name+'/trainA/*.' + "jpg")
        img_list_B = glob('drive/My Drive/GAN/datasets/'+dataset_name+'/trainB/*.' + "jpg")
        trainA=[]
        trainB=[]
        for img in img_list_A:
            trainA_img = load_img(img,grayscale=False)
            trainA_img = Image.fromarray(np.uint8(trainA_img))
            trainA_img = np.asarray(trainA_img.resize((128,128)))
            trainA_img = img_to_array(trainA_img) /255
            trainA.append(trainA_img)
        for img in img_list_B:
            trainB_img = load_img(img,grayscale=False)
            trainB_img = Image.fromarray(np.uint8(trainB_img))
            trainB_img = np.asarray(trainB_img.resize((128,128)))
            trainB_img = img_to_array(trainB_img) /255
            trainB.append(trainB_img)
        trainA=np.array(trainA)
        trainB=np.array(trainB)
        return trainA,trainB
    else:
      name=dataset_name.split("2")
      trainA=np.load("drive/My Drive/GAN/datasets/"+dataset_name+"/"+name[0]+"_numpy.npy")
      trainB=np.load("drive/My Drive/GAN/datasets/"+dataset_name+"/"+name[1]+"_numpy.npy")
      return trainA,trainB

def resblock(y):
    x=Conv2D(filters=256,kernel_size=3,strides=1,padding="same",kernel_initializer=RandomNormal(0,0.02))(y)
    x=InstanceNormalization()(x)
    x=ReLU()(x)
    x=Conv2D(filters=256,kernel_size=3,strides=1,padding="same",kernel_initializer=RandomNormal(0,0.02))(x)
    x=InstanceNormalization()(x)
    out=Add()([x,y])
    return out

def active(x,type):
    x=InstanceNormalization()(x)
    if type=="ReLU":
        x=ReLU()(x)
    if type=="LeakyReLU":
        x=LeakyReLU(0.2)(x)
    return x

trainA,trainB=load_data("horse2zebra")

cycle_gan=CycleGAN()
cycle_gan.train(trainA,trainB,epochs=100,batch_size=1,sample_interval=10)
print("end")
