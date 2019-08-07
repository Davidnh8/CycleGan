import scipy
import keras
import tensorflow as tf
from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import glob
import imageio

def img_range(img):
    return ((img+1)*127.5).astype(np.int32)

def load_data(pathA, pathB, row_shape, col_shape, batch_size=32):
    """pathA(str): path that contains image set A
        pathB(str): path that contains image set B"""
    # list of file names
    path_imgA=glob.glob(pathA)
    path_imgB=glob.glob(pathB)
    
    n_images = min(len(path_imgA), len(path_imgB))
    n_batch = int(n_images//batch_size)
    # list of chosen filenames
    chosen_imgA = np.random.choice(path_imgA, n_images, replace=False)
    chosen_imgB = np.random.choice(path_imgB, n_images, replace=False)
    
    for i in range(n_batch-1):
        imgsA=[]

        imgA=chosen_imgA[i*batch_size : (i+1)*batch_size]
        for j in imgA:
            img=imageio.imread(j)
            img=scipy.misc.imresize(img, size=(row_shape, col_shape,3))
            imgsA.append(img)
        
        imgsB=[]
        imgB=chosen_imgB[i*batch_size : (i+1)*batch_size]
        for j in imgB:
            img=imageio.imread(j)
            img=scipy.misc.imresize(img, size=(row_shape, col_shape,3))
            imgsB.append(img)
        
        imgsA=np.array(imgsA)/127.5-1
        imgsB=np.array(imgsB)/127.5-1
        
        yield imgsA,imgsB
        


class CycleGan():
    def __init__(self, path, img_rows, img_cols, img_channels, cycle_lambda):
        
        # image parameters
        self.path=path
        self.img_rows=img_rows
        self.img_cols=img_cols
        self.channels=img_channels
        self.img_shape=(self.img_rows, self.img_cols, self.channels)
        
        
        
        # build discriminator and generator
        self.optimizer=Adam(0.0002, 0.5)
        self.Dx=self.build_discriminator()
        self.Dx.compile(loss='mse', optimizer=self.optimizer, metrics=['accuracy'])
        self.Dy=self.build_discriminator()
        self.Dy.compile(loss='mse', optimizer=self.optimizer, metrics=['accuracy'])
        
        self.genXY=self.build_generator()
        self.genYX=self.build_generator()
        
        # weights
        self.cycle_lambda = cycle_lambda
        self.id_lambda = 0.5 * self.cycle_lambda
        
        self.build_gan()
        
    def train(self, epochs=10, batch_size=32, sample_interval=10):
        
        real = 0.9 * np.ones((batch_size,)+(self.img_rows//16,self.img_cols//16,1))
        fake = np.zeros((batch_size,)+(self.img_rows//16,self.img_cols//16,1))
        for epoch in range(epochs):
            
            for imgX,imgY in load_data(self.path+"\\trainA\\*.jpg", self.path+"\\trainB\\*.jpg", self.img_rows, self.img_cols,batch_size=batch_size):
                '''train discriminator'''
                fakeY = self.genXY.predict(imgX)
                fakeX = self.genYX.predict(imgY)
                
                # real images
                self.Dy.train_on_batch(imgY, real)
                self.Dx.train_on_batch(imgX, real)
                
                # fake images
                self.Dy.train_on_batch(fakeY, fake)
                self.Dx.train_on_batch(fakeX, fake)
            
                '''train generator'''
                self.gan.train_on_batch([imgX, imgY],
                                  [real, real,
                                      imgX, imgY,
                                          imgX, imgY])
            if epoch%sample_interval==0:
                print(epoch)
                self.plot_image(epoch, self.path)
                
        return self.gan, self.genXY, self.genYX
    
    
    def encode(self, input_layer, filters):
        layer = Conv2D(filters, kernel_size=5, strides=2, padding='same')(input_layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = InstanceNormalization()(layer)
        return layer
    
    def decode(self, input_layer, forward_layer, filters):
        layer = UpSampling2D(size=2)(input_layer)
        layer = Conv2D(filters, kernel_size=5, strides=1, padding='same', activation='relu')(layer)
        layer = InstanceNormalization()(layer)
        layer = Concatenate()([layer, forward_layer])
        return layer
    
    def build_gan(self):
        
        '''cycle'''
        # image parameter
        imageX=Input(shape=self.img_shape)
        imageY=Input(shape=self.img_shape)
        # generated images
        fakeX=self.genYX(imageY)
        fakeY=self.genXY(imageX)
        # cyclded images images
        cycleX =self.genYX(fakeY)
        cycleY =self.genXY(fakeX)
        
        '''identity map'''
        # identity map: used to preserve the color, prevent color reverse
        id_imageX=self.genYX(imageX)
        id_imageY=self.genXY(imageY)
        
        # do not train discriminator
        self.Dx.trainable=False
        self.Dy.trainable=False
        
        validX=self.Dx(fakeX)
        validY=self.Dy(fakeY)
        
        self.gan= Model(inputs=[imageX, imageY],
                           outputs=[validX, validY,
                                       cycleX, cycleY,
                                           id_imageX, id_imageY])
        self.gan.compile(loss=['mse','mse',
                                   'mae', 'mae',
                                       'mae', 'mae'],
                             loss_weights=[1,1,
                                           self.cycle_lambda, self.cycle_lambda,
                                           self.id_lambda, self.id_lambda],
                                 optimizer=self.optimizer)
        
    def build_discriminator(self):
        disc_input=Input(shape=self.img_shape)
        model = Conv2D(64, kernel_size=5, strides=2, padding='same')(disc_input)
        model = LeakyReLU(alpha=0.5)(model)
        model = Conv2D(128, kernel_size=5, strides=2, padding='same')(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = InstanceNormalization()(model)
        model = Conv2D(256, kernel_size=5, strides=2, padding='same')(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = InstanceNormalization()(model)
        model = Conv2D(512, kernel_size=5, strides=2, padding='same')(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = InstanceNormalization()(model)        
        model = Conv2D(1, kernel_size=5, strides=1, padding='same')(model)
        
        disc_model= Model(disc_input, model)
        #disc_model.summary()
        return disc_model
    
    def build_generator(self):
        gen_input=Input(shape=self.img_shape)
        
        encode_1 = self.encode(gen_input, 32)
        encode_2 = self.encode(encode_1, 64)
        encode_3 = self.encode(encode_2, 128)
        encode_4 = self.encode(encode_3, 256)
        
        decode_1 = self.decode(encode_4, encode_3, 128)
        decode_2 = self.decode(decode_1, encode_2, 64)
        decode_3 = self.decode(decode_2, encode_1, 32)
        
        
        gen_model=UpSampling2D(size = 2)(decode_3)
        gen_model=Conv2D(self.channels, kernel_size=5, strides=1, padding='same', activation='tanh')(gen_model)
        
        final_gen_model = Model(gen_input, gen_model)
        #final_gen_model.summary()
        
        return final_gen_model
    
    def plot_image(self,epoch, path):
        imgX_random=np.random.choice(glob.glob(path+'trainA\\*.jpg'),1)
        imX=imageio.imread(imgX_random[0])
        imX=scipy.misc.imresize(imX, (self.img_rows,self.img_cols,3))
        imX=imX/127.5-1

        imgY_random=np.random.choice(glob.glob(path+'trainB\\*.jpg'),1)
        imY=imageio.imread(imgY_random[0])
        imY=scipy.misc.imresize(imY, (self.img_rows,self.img_cols,3))
        imY=imY/127.5-1

        generated_imY=self.genXY.predict(imX.reshape(1,self.img_rows,self.img_cols,3))
        generated_imX=self.genYX.predict(imY.reshape(1,self.img_rows,self.img_cols,3))
        reconstructed_imX=self.genYX.predict(generated_imY)
        reconstructed_imY=self.genXY.predict(generated_imX)
        
        # reshape
        generated_imX=generated_imX.reshape(self.img_rows,self.img_cols,3)
        generated_imY=generated_imY.reshape(self.img_rows,self.img_cols,3)
        reconstructed_imX=reconstructed_imX.reshape(self.img_rows,self.img_cols,3)
        reconstructed_imY=reconstructed_imY.reshape(self.img_rows,self.img_cols,3)
        
        fig,ax=plt.subplots(2,3, figsize=[24,16])
        ax[0,0].imshow(img_range(imX))
        ax[1,0].imshow(img_range(imY))
        
        ax[0,1].imshow(img_range(generated_imY))
        ax[1,1].imshow(img_range(generated_imX))
        
        ax[0,2].imshow(img_range(reconstructed_imX))
        ax[1,2].imshow(img_range(reconstructed_imY))
        
        ax[0,0].set_title("original")
        ax[0,1].set_title("transform")
        ax[0,2].set_title("reconstructed")
        
        fig.savefig(path+"save\\epoch=%s.jpg" %epoch)
        plt.show()