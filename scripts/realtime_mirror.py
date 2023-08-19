import cv2
import numpy as np
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
generator_name = 'mom_1.3_50'
current_directory = os.path.join(os.getcwd(), '../')
generator_path = os.path.join(current_directory, r'saved_generator/'+generator_name)
g_model = tf.keras.models.load_model(generator_path, compile=False)

#target_image = cv2.imread('../training_datasets/smiling_lady/output_color/test_1.png')
#target_image = cv2.resize(target_image, (256, 256))

#TODO: I think my main problem is that model is training with small faces
# and is focusing on everything around it as well
# i think it would be better to extract face and fill the whole image with it 

# pad = 200
# height_shift_up = 100
# width_shift_right = 40
# pad = 0
# height_shift_up = 0
# width_shift_right = 0

g_model.summary()

def show_generated_frame(realImg, realMask, g_model):
    # realImg = realImg.astype(np.float32) / 127.5 - 1.0
    realMaskExpanded = np.expand_dims(realMask, axis=0) # add batch dimension
    print(realMask.shape)
    fakeImg = g_model.predict([realMaskExpanded])
    realMask = (realMask+1.0)/2.0
    realImg = (realImg+1)/2
    fakeImg = (fakeImg+1)/2
    print(fakeImg[0].shape)
    
    cv2.imshow("Mask", realMask)
    cv2.imshow("RealImage", realImg)
    cv2.imshow("FakeImage", fakeImg[0])
    cv2.waitKey(1)

camera = cv2.VideoCapture(0)

length = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(camera.get(cv2.CAP_PROP_FPS))

def __get_data__():
    _, fr = camera.read()
    cv2.imshow("RealImage", fr)
    gray = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)
    
    return fr, gray

#read images indefinitely
while True:
    #stop executing
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    realImg, gray_fr = __get_data__()

    face_cascade = cv2.CascadeClassifier("../haarcascade_frontalface_default.xml") #TODO: move up 
    faces = face_cascade.detectMultiScale(gray_fr, scaleFactor=1.2, minNeighbors=3)

    # Extract the face from the image and crop the image to focus on the face
    # for (x, y, w, h) in faces:
    #     face_image = realImg[y:y+h, x:x+w]
    #     break

    
    print(faces)
    if len(faces) > 0:
        # Extract the face from the image and crop the image to focus on the face
        for (x, y, w, h) in faces:
            face_image = realImg[y-int((0.2*(h-y))):y+h+int(0.2*(h-y)), x-int(0.2*(w-x)):x+w+int(0.2*(w-x))] #extend the crop to a set amount of pixels each side
            break

        
        if face_image.any():
            # Resize the image to 256x256
            resized_image = cv2.resize(face_image, (256, 256))
            cv2.imshow("RealImage", resized_image)
            cv2.waitKey(0)

            #Create mask for detected face
            gray_resized_img = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
            faces = detector(gray_resized_img)
            # TODO: why tf am i using detector if I already have got faces using detectMultiScale???? 
            # If I understand correctly (cuz im a bit tired) I already have face centered and I force another algo to extract the face again
            # CHECK IT 

            realMask = np.zeros((resized_image.shape[0], resized_image.shape[1], 3), np.uint8)
            print("SHAPE 1: ", resized_image.shape)
            if not faces:
                print('face not found!')
                continue
            for face in faces:
                # Create landmark object
                landmarks = predictor(image=gray_resized_img, box=face)
                
                # Loop through all the points
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    #Create mask image
                    cv2.circle(img=realMask, center=(x, y), radius=2, color=(255,255,255), thickness=-1)
            print("SHAPE 2: ", realMask.shape)
            #TODO: test if this is required
            resized_image = resized_image/255.0
            realMask = realMask/255.0
            resized_image = (resized_image - 0.5)/ 0.5
            realMask = (realMask - 0.5)/ 0.5
            show_generated_frame(resized_image, realMask, g_model)      
    else:
        print('face not found')

    
        

