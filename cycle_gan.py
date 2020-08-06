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
from tensorflow.keras.callbacks import LearningRateScheduler
import datetime
import matplotlib.pyplot as plt
import sys
from tensorflow_addons.layers import InstanceNormalization
from PIL import Image



#モデル
class CycleGAN:
    def __init__(self,identity=False):

        #入力画像のサイズ
        self.img_rows=128
        self.img_cols=128
        self.channels=3
        self.img_shape=(self.img_rows,self.img_cols,self.channels)

        #discriminator出力のサイズ
        self.patch_rows=self.img_rows//16
        self.patch_cols=self.img_cols//16

        #ハイパーパラメータ
        self.lambda_cycle=10.0#adversarial_lossに比べたcycle_consistancy_lossの比率(論文と同数値)
        self.lambda_id=0.5*self.lambda_cycle#adversarial_lossに比べたidentity_lossの比率(論文と同数値)
        self.leaky=0.2#LeakyReLUの傾き(論文と同数値)
        self.stddev=0.02**0.5#重みのガウス分布初期化の標準偏差(論文は分散が0.02だったのでおそらく同数値)
        self.identity=identity

        #最適化アルゴリズム(論文と同数値)
        self.optimizer1=Adam(0.0002,0.5)
        self.optimizer2=Adam(0.0002,0.5)
        self.optimizer3=Adam(0.0002,0.5)
        self.optimizer4=Adam(0.0002,0.5)
        self.optimizers=[self.optimizer1,self.optimizer2,self.optimizer3,self.optimizer4]

        #モデル構築
        self.d_A=self.build_discriminator()
        self.d_B=self.build_discriminator()
        self.d_A.compile(loss="binary_crossentropy",optimizer=self.optimizer1)
        self.d_B.compile(loss="binary_crossentropy",optimizer=self.optimizer2)
        self.g_AB=self.build_generator()
        self.g_BA=self.build_generator()
        img_A=Input(shape=self.img_shape)
        img_B=Input(shape=self.img_shape)
        fake_B=self.g_AB(img_A)
        fake_A=self.g_BA(img_B)
        reconstr_A=self.g_BA(fake_B)
        reconstr_B=self.g_AB(fake_A)
        if self.identity:
            img_A_id=self.g_BA(img_A)
            img_B_id=self.g_AB(img_B)
        self.d_A.trainable=False
        self.d_B.trainable=False
        valid_A=self.d_A(fake_A)
        valid_B=self.d_B(fake_B)
        if self.identity:
            self.combined=Model(inputs=[img_A,img_B],outputs=[valid_A,valid_B,reconstr_A,reconstr_B,img_A_id,img_B_id])
            self.combined.compile(loss=["binary_crossentropy","binary_crossentropy","mae","mae","mae","mae"],loss_weights=[1,1,self.lambda_cycle,self.lambda_cycle,self.lambda_id,self.lambda_id],optimizer=self.optimizer3)
        else:
            self.combined=Model(inputs=[img_A,img_B],outputs=[valid_A,valid_B,reconstr_A,reconstr_B])
            self.combined.compile(loss=["binary_crossentropy","binary_crossentropy","mae","mae"],loss_weights=[1,1,self.lambda_cycle,self.lambda_cycle],optimizer=self.optimizer4)

        #res-netブロックアーキテクチャ(論文ではInstancenormalizationが入ってなかったかも)
    def resblock(self,y):
        x=Conv2D(filters=256,kernel_size=3,strides=1,padding="same",kernel_initializer=RandomNormal(0,self.stddev))(y)
        x=InstanceNormalization()(x)
        x=ReLU()(x)
        x=Conv2D(filters=256,kernel_size=3,strides=1,padding="same",kernel_initializer=RandomNormal(0,self.stddev))(x)
        x=InstanceNormalization()(x)
        out=Add()([x,y])
        return out


        #generatorアーキテクチャ(論文通り)
    def build_generator(self):
        img=Input(shape=self.img_shape)
        x=Conv2D(filters=64,kernel_size=7,strides=1,padding="same",kernel_initializer=RandomNormal(0,self.stddev))(img)
        x=InstanceNormalization()(x)
        x=ReLU()(x)
        x=Conv2D(filters=128,kernel_size=3,strides=2,padding="same",kernel_initializer=RandomNormal(0,self.stddev))(x)
        x=InstanceNormalization()(x)
        x=ReLU()(x)
        x=Conv2D(filters=256,kernel_size=3,strides=2,padding="same",kernel_initializer=RandomNormal(0,self.stddev))(x)
        x=InstanceNormalization()(x)
        x=ReLU()(x)
        for i in range(6):
            x=self.resblock(x)
        x=Conv2DTranspose(filters=128,kernel_size=3,strides=2,padding="same",kernel_initializer=RandomNormal(0,self.stddev))(x)
        x=InstanceNormalization()(x)
        x=ReLU()(x)
        x=Conv2DTranspose(filters=64,kernel_size=3,strides=2,padding="same",kernel_initializer=RandomNormal(0,self.stddev))(x)
        x=InstanceNormalization()(x)
        x=ReLU()(x)
        x=Conv2D(filters=3,kernel_size=7,strides=1,padding="same",kernel_initializer=RandomNormal(0,self.stddev))(x)
        x=InstanceNormalization()(x)
        x=ReLU()(x)
        return Model(img,x)


        #discriminatorアーキテクチャ(論文通り)
    def build_discriminator(self):

        img=Input(shape=self.img_shape)
        x=Conv2D(filters=64,kernel_size=4,strides=2,padding="same",kernel_initializer=RandomNormal(0,self.stddev))(img)
        x=LeakyReLU(self.leaky)(x)
        x=Conv2D(filters=128,kernel_size=4,strides=2,padding="same",kernel_initializer=RandomNormal(0,self.stddev))(x)
        x=InstanceNormalization()(x)
        x=LeakyReLU(self.leaky)(x)
        x=Conv2D(filters=256,kernel_size=4,strides=2,padding="same",kernel_initializer=RandomNormal(0,self.stddev))(x)
        x=InstanceNormalization()(x)
        x=LeakyReLU(self.leaky)(x)
        x=Conv2D(filters=512,kernel_size=4,strides=2,padding="same",kernel_initializer=RandomNormal(0,self.stddev))(x)
        x=InstanceNormalization()(x)
        x=LeakyReLU(self.leaky)(x)
        x=Conv2D(filters=1,kernel_size=4,strides=1,padding="same",kernel_initializer=RandomNormal(0,self.stddev))(x)
        # x=Reshape(target_shape=(x.shape[1]*x.shape[2]*x.shape[3],1))(x)
        # x=AveragePooling1D(pool_size=x.shape[1])(x)
        # x=Flatten()(x)

        return Model(img,x)



    def train(self,trainA,trainB,epochs,batch_size=1):

        #正解ラベル
        valid=np.ones((batch_size,self.patch_rows,self.patch_cols,1))
        fake=np.zeros((batch_size,self.patch_rows,self.patch_cols,1))

        #lossのグラフ用データ格納リスト
        self.epoch_x=[]
        self.g_loss_y=[]
        self.d_loss_y=[]

        #訓練
        for epoch in range(epochs):
            self.epoch_x.append(epoch)
            print("epoch:"+str(epoch))

            #10エポックごとに画像書き出し
            if epoch%10==9:
                imgs_A=trainA[np.random.randint(0,trainA.shape[0],size=1)]
                fake_B=self.g_AB.predict(imgs_A)
                imgs_A=np.clip(imgs_A,0,1)
                fake_B=np.clip(fake_B,0,1)
                plt.imsave("output/trueA/epoch_"+str(epoch)+".png" ,imgs_A.reshape(128,128,3) )
                plt.imsave( "output/fakeB/epoch_"+str(epoch)+".png" ,fake_B.reshape(128,128,3) )
                imgs_B=trainB[np.random.randint(0,trainA.shape[0],size=1)]
                fake_A=self.g_BA.predict(imgs_B)
                imgs_B=np.clip(imgs_B,0,1)
                fake_A=np.clip(fake_A,0,1)
                plt.imsave("output/trueB/epoch_"+str(epoch)+".png" , imgs_B.reshape(128,128,3) )
                plt.imsave("output/fakeA/epoch_"+str(epoch)+".png" , fake_A.reshape(128,128,3) )

            if epoch>=100:
                for optimizer in self.optimizers:
                    optimizer.lr=0.0002*(2-epoch/100)
            #訓練メインパート
            d_loss_y=0
            g_loss_y=0
            for batch_i in range(995//batch_size):
                imgs_A=trainA[np.random.randint(0,trainA.shape[0],size=batch_size)]
                imgs_B=trainB[np.random.randint(0,trainB.shape[0],size=batch_size)]
                fake_B=self.g_AB.predict(imgs_A)
                fake_A=self.g_BA.predict(imgs_B)

                dA_loss_real=self.d_A.train_on_batch(imgs_A,valid)
                dA_loss_fake=self.d_A.train_on_batch(fake_A,fake)
                # dA_loss=0.5*np.add(dA_loss_real,dA_loss_fake)
                dA_loss=0.5*(dA_loss_real+dA_loss_fake)
                dB_loss_real=self.d_B.train_on_batch(imgs_B,valid)
                dB_loss_fake=self.d_B.train_on_batch(fake_B,fake)
                # dB_loss=0.5*np.add(dB_loss_real,dB_loss_fake)
                dB_loss=0.5*(dB_loss_real+dB_loss_fake)
                # d_loss=0.5*np.add(dA_loss,dB_loss)
                d_loss=0.5*(dA_loss+dB_loss)
                if self.identity:
                    g_loss=self.combined.train_on_batch([imgs_A,imgs_B],[valid,valid,imgs_A,imgs_B,imgs_A,imgs_B])
                else:
                    g_loss=self.combined.train_on_batch([imgs_A,imgs_B],[valid,valid,imgs_A,imgs_B])
                g_loss=g_loss[0]
                d_loss_y+=d_loss
                g_loss_y+=g_loss
            d_loss_y/=(995//batch_size)
            g_loss_y/=(995//batch_size)
            self.d_loss_y.append(d_loss_y)
            self.g_loss_y.append(g_loss_y)

    def loss_graph(self):

        plt.plot(self.epoch_x,self.g_loss_y,label='g_loss')
        plt.plot(self.epoch_x,self.d_loss_y,label='d_loss')
        plt.legend()
        plt.savefig('output/figure.png')
#データロード用
def load_data(dataset_name):
    if not os.path.exists("datasets/"+dataset_name+"/numpy_data"):
        img_list_A = glob('datasets/'+dataset_name+'/trainA/*.' + "jpg")
        img_list_B = glob('datasets/'+dataset_name+'/trainB/*.' + "jpg")
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
        os.mkdir("datasets/"+dataset_name+"/numpy_data")
        name=dataset_name.split("2")
        np.save("datasets/"+dataset_name+"/numpy_data/"+name[0]+"_numpy",trainA)
        np.save("datasets/"+dataset_name+"/numpy_data/"+name[1]+"_numpy",trainB)
    else:
      name=dataset_name.split("2")
      trainA=np.load("datasets/"+dataset_name+"/numpy_data/"+name[0]+"_numpy.npy")
      trainB=np.load("datasets/"+dataset_name+"/numpy_data/"+name[1]+"_numpy.npy")
    return trainA,trainB





#=================================================================================


#データロード
trainA,trainB=load_data("horse2zebra")

#モデル定義
cycle_gan=CycleGAN()

#エポック数入力
epochs=int(input("epochs:"))
batch_size=int(input("batch_size:"))

#バッチサイズ1での訓練
cycle_gan.train(trainA,trainB,epochs=epochs,batch_size=batch_size)
print("end")

cycle_gan.loss_graph()
