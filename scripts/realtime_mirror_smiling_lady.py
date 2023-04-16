import cv2
import numpy as np
import imutils
import dlib
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Input, Concatenate, LeakyReLU, Conv2DTranspose
from tensorflow.keras.optimizers import Adam


# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")

# Load the model
generator_name = 'smiling_lady_gen'
current_directory = os.path.join(os.getcwd(), '../')
generator_path = os.path.join(current_directory, r'saved_generator/'+generator_name)
g_model = tf.keras.models.load_model(generator_path, compile=False)

target_image = cv2.imread('../training_datasets/smiling_lady/output_color/test_1.png')
target_image = cv2.resize(target_image, (256, 256))

g_model.summary()

def show_generated_frame(targetImg, realImg, realMask, g_model):
    realImg = realImg.astype(np.float32) / 127.5 - 1.0
    realImg = np.expand_dims(realImg, axis=0) # add batch dimension

    fakeImg = g_model.predict([realImg, realMask])
    fakeImg = np.squeeze(fakeImg, axis=0)
    fakeImg = (fakeImg + 1.0) * 127.5
    fakeImg = np.clip(fakeImg, 0, 255).astype(np.uint8)

    realImg = np.squeeze(realImg, axis=0)
    realImg = (realImg + 1.0) * 127.5
    realImg = np.clip(realImg, 0, 255).astype(np.uint8)

    #What is this for?
    # realMask = (realMask+1.0)/2.0
    # realImg = (realImg+1)/2
    # fakeImg = (fakeImg+1)/2
    
    plt.figure(figsize=(15, 15))
    ax = plt.subplot(1, 3, 1)
    plt.imshow(targetImg)
    ax = plt.subplot(1, 3, 2)
    plt.imshow(realImg)
    ax = plt.subplot(1, 3, 3)
    plt.imshow(fakeImg)
    plt.show()

camera = cv2.VideoCapture(0)

length = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(camera.get(cv2.CAP_PROP_FPS))

def __get_data__():
    _, fr = camera.read()
    #fr = cv2.resize(fr, (256, 256)) # I don't think that's needed
    fr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    
    return fr, gray

#read images indefinitely
while True:
    #stop executing
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ##CREATE A MASK
    realImg, gray_fr = __get_data__()

    face_cascade = cv2.CascadeClassifier("../haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray_fr, scaleFactor=1.1, minNeighbors=5)

    # Extract the face from the image and crop the image to focus on the face
    for (x, y, w, h) in faces:
        face_image = realImg[y:y+h, x:x+w]
        break

    # Resize the image to 256x256
    resized_image = cv2.resize(face_image, (256, 256))
    gray_resized_img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_resized_img)

    #DEBUGGING PUPROSES
    #cv2.imshow('readFace',resized_image)
    #DEBUGGING PUPROSES
    #cv2.imshow('readGrayFace',gray_resized_img)

    #TODO: optimization - I could probably use found mask from 'detectMultiScale' to draw a mask
    # instead of using detector

    realMask = np.zeros((resized_image.shape[0], resized_image.shape[1], 3), np.uint8)
    print("SHAPE 1: ", resized_image.shape)
    for face in faces:
        x1 = face.left() # left point
        y1 = face.top() # top point
        x2 = face.right() # right point
        y2 = face.bottom() # bottom point

        # Create landmark object
        landmarks = predictor(image=gray_resized_img, box=face) #gray_fr to gray_resized_img might fuck up
        
        # Loop through all the points
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            #Create mask image
            cv2.circle(img=realMask, center=(x, y), radius=3, color=255, thickness=-1)
        realMask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to 1 channel
    # might not be needed
    # realMask = np.array(realMask)
    # realImg = np.array(realImg)
    #image recalculation? (might not be needed)
    # realMask = realMask/255.0
    # realImg = realImg/255.0
    # recalcuate to (-1,1)
    # realMask = (realMask - 0.5) / 0.5
    # realImg = (realImg - 0.5) / 0.5
    
    #Added use of mask in generator!
    show_generated_frame(target_image, resized_image, realMask, g_model)  
        

