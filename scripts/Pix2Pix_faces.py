import numpy as np
from numpy.random import randint
import os
import cv2

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Input, Concatenate, LeakyReLU, Conv2DTranspose
from tensorflow.keras.optimizers import Adam

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

# Saved Generator path

generator_name = 'smiling_lady_gen'
current_directory = os.getcwd() 
generator_path = os.path.join(current_directory, r'../saved_generator/'+generator_name)

# Create generator directory if needed
if not os.path.exists(generator_path):
   os.makedirs(generator_path)



# load data from two folders
n = 658 # CHANGE PROGRAMATICALLY
size = 256
trainImgs = []
trainFaces = []

def load_original_images_from_folder(folder):
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img = cv2.resize(img, (256, 256))
        if img is not None:
            trainImgs.append(img)
    return trainImgs

def load_face_images_from_folder(folder):
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img = cv2.resize(img, (256, 256))
        if img is not None:
            trainFaces.append(img)
    return trainFaces



trainImgs = np.array(load_original_images_from_folder("../training_datasets/smiling_lady/output_color"))
trainFaces = np.array(load_face_images_from_folder("../training_datasets/smiling_lady/output_mask"))
trainImgs = trainImgs/255.0
trainFaces = trainFaces/255.0

print(trainImgs.shape)

import matplotlib.pyplot as plt
import random
plt.figure(figsize=(15, 15))
n = 3
x = 1
for i in range(n):
    ax = plt.subplot(3, 2, x)
    x = x + 1
    plt.imshow(trainFaces[i])
    plt.axis("off")
    ax = plt.subplot(3, 2, x)
    x = x + 1
    plt.imshow(trainImgs[i])
    plt.axis("off")
plt.show()

# recalcuate to (-1,1)
trainImgs = (trainImgs - 0.5) / 0.5
trainFaces = (trainFaces - 0.5) / 0.5

#DISCRIMINATOR MODEL
from tensorflow.keras.initializers import RandomNormal
# define the discriminator model
def define_discriminator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image input A
    in_src_image = Input(shape=image_shape)
    # target image input B
    in_target_image = Input(shape=image_shape)
    # concatenate images channel-wise
    merged = Concatenate()([in_src_image, in_target_image])
    # C64
    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image], patch_out)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model
d_model = define_discriminator((256,256,3))
print(d_model.summary())

# define an encoder block
def encoder_block(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g

# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # merge with skip connection
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation(tf.nn.leaky_relu)(g) #CHANGED FROM RELU TO LEAKY
    return g

#GENERATOR MODEL
# define the standalone generator model
def define_generator(image_shape=(256,256,3)):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # encoder model
    e1 = encoder_block(in_image, 64, batchnorm=False)
    e2 = encoder_block(e1, 128)
    e3 = encoder_block(e2, 256)
    e4 = encoder_block(e3, 512)
    e5 = encoder_block(e4, 512)
    e6 = encoder_block(e5, 512)
    e7 = encoder_block(e6, 512)
    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
    b = Activation(tf.nn.leaky_relu)(b) #CHANGED FROM RELU TO LEAKY
    # decoder model
    d1 = decoder_block(b, e7, 512)
    d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(d2, e5, 512)
    d4 = decoder_block(d3, e4, 512, dropout=True)
    d5 = decoder_block(d4, e3, 256, dropout=True)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)
    # output
    g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('tanh')(g)
    # define model
    model = Model(in_image, out_image)
    return model
    
g_model = define_generator()
print(g_model.summary())

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
    # make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    # define the source image
    in_src = Input(shape=image_shape)
    # connect the source image to the generator input
    gen_out = g_model(in_src)
    # connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src, gen_out])
    # src image as input, generated image and classification output
    model = Model(in_src, [dis_out, gen_out])
    opt = Adam(lr=0.001, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
    return model
gan_model = define_gan(g_model, d_model, (256,256,3))

# select a batch of random samples, returns images and target
def generate_real_samples(samples):
    iImg = randint(0, trainImgs.shape[0], samples)
    #added line
    #iFace = randint(0, trainFaces.shape[0], samples)
    #PREVIOUS SOLUTION
    X1, X2 = trainFaces[iImg], trainImgs[iImg]
    #SOLUTION FOR ONE IMAGE (this will probably suck)
    #X1, X2 = trainFaces[iImg], trainImgs[0] 
    return X1, X2

def show_results(step, g_model, samples=3):
    # select a sample of input images
    realFaces, realImgs = generate_real_samples(samples)
    fakeImg = g_model.predict(realFaces[:samples])
    realFaces = (realFaces+1.0)/2.0
    realImgs = (realImgs+1)/2
    fakeImg = (fakeImg+1)/2
    
    for i in range(samples):
        plt.figure(figsize=(15, 15))
        ax = plt.subplot(1, 3, 1)
        plt.imshow(realFaces[i])
        ax = plt.subplot(1, 3, 2)
        plt.imshow(realImgs[i])
        ax = plt.subplot(1, 3, 3)
        plt.imshow(fakeImg[i])
    plt.show()
show_results(0,g_model,3)

# train pix2pix model
def train(d_model, g_model, gan_model, epochs=100, batch=1):
    # determine the output square shape of the discriminator
    patch = d_model.output_shape[1]
    steps = int(len(trainImgs) / batch)
    all_ones = np.ones((batch, patch, patch, 1))
    all_zeros = np.zeros((batch, patch, patch, 1))
    # manually enumerate epochs
    for epoch in range(epochs):
        #show_results(epoch, g_model, samples=1)
        print(f"Calculating next {steps} batches of size {batch}")
        for i in range(steps):
            # select a batch of real samples
            realFaces, realImgs = generate_real_samples(batch)
            # generate a batch of fake samples
            fakeImg = g_model.predict(realFaces)
            # update discriminator for real samples
            d_loss_real = d_model.train_on_batch([realFaces, realImgs], all_ones )
            # update discriminator for generated samples
            d_loss_fake = d_model.train_on_batch([realFaces, fakeImg], all_zeros)
            # update the generator
            g_loss, _, _ = gan_model.train_on_batch(realFaces, [all_ones, realImgs])
            #print(f"Iteration {i}/{n_steps} g_loss={g_loss:.3f}, d_loss_real={d_loss_real:.3f}, d_loss_fake={d_loss_fake:.3f}")
            print(".",end='')
        print()    
        print(f"Epoch {epoch} g_loss={g_loss:.3f}, d_loss_real={d_loss_real:.3f}, d_loss_fake={d_loss_fake:.3f}")

def save_generator(g_model, generator_path):
    g_model.save(generator_path)
    
# load image data
image_shape = (256,256,3) #dataset[0].shape[1:]
# define the models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)

# train model
train(d_model, g_model, gan_model, 100, 32)

show_results(0,g_model,10)

save_generator(g_model,generator_path)