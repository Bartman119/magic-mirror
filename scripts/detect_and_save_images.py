import cv2
import numpy as np
import dlib
import sys
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#TODO: input video has too much side movement and face detection and stretch algorithm is making mistakes!

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")

OUTPUT_FACE_MASK_PATH = "../training_datasets/unshaved_me/output_face_mask"
OUTPUT_FACE_COLOR_PATH = "../training_datasets/unshaved_me/output_face_color"
RECORDING_PATH = '../raw_recordings/unshaved_me/test3_closeup.mp4'

#OPTION 1: DEFINE PADDING FOR IMAGE

# Define the amount of padding to be added
pad = 50
height_shift_up = 200
width_shift_right = 40

if not os.path.exists(OUTPUT_FACE_COLOR_PATH):
    os.makedirs(OUTPUT_FACE_COLOR_PATH)

if not os.path.exists(OUTPUT_FACE_MASK_PATH):
    os.makedirs(OUTPUT_FACE_MASK_PATH)

basename = "test"
image_count = 0
def load_img(indir):
    global image_count
    samples = []
    for file in os.listdir(indir):
        image = cv2.imread("{}/{}".format(indir,file))
        image = cv2.resize(image, (256,256))
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # plt.figure(figsize=(5, 5))
        # ax = plt.subplot(1, 1, 1)
        # plt.imshow(image)
        # plt.show()

        # Add padding to the image
        #padded_img = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

        samples.append(image)
        #samples.append(image)
        if image_count%10==0: print('.',end='')
        image_count = image_count + 1

    samples = np.array(samples)
    return samples

aug = ImageDataGenerator(
    rotation_range=0,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")


def create_images(dir,images,number=500):
    #images = aug.flow(images, batch_size=1, save_to_dir=dir,save_prefix="test_", save_format="png")
    total = 0
    for image in images:
        #OPTION 1: USE TRANSFORM AND REMOVE PADDING
        # Apply the augmentation techniques to the padded image using ImageDataGenerator
        augmented_img = aug.random_transform(image)

        # Crop the resulting image back to its original size
        #cropped_img = augmented_img[pad:-pad, pad:-pad]
        total += 1
        # plt.imshow(augmented_img)
        # plt.axis('off')
        # plt.savefig( OUTPUT_FACE_COLOR_PATH + "/{}_{}.png".format(basename, total), bbox_inches='tight')
        # plt.clf()
        cv2.imwrite(OUTPUT_FACE_COLOR_PATH + "/{}_{}.png".format(basename, total), augmented_img)
        if total == number: 
            print("{} images generated to {}".format(total,dir))
            break
        if total%(number/100)==0: print('.',end='')   

rgb = cv2.VideoCapture(RECORDING_PATH)
length = int(rgb.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(rgb.get(cv2.CAP_PROP_FPS))

def __get_data__():
    _, fr = rgb.read()
    gray = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)
    #grayConverted = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    #frConverted = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
    
    return fr, gray

ix = 0
width  = int(rgb.get(3)) # float
height = int(rgb.get(4))
#fourcc = cv2.VideoWriter_fourcc(*'vp90')
#PATH = '../raw_recordings/smiling_lady_face_mask.webm'
#output = cv2.VideoWriter(PATH,fourcc, fps, (width,height))

for sm in range(1,length-1):
        ix += 1
        fr, gray_fr = __get_data__()

        #add preprocessing of an image:
        #extract just the face and stretch it to 256x256 

        # Use a face detection cascade classifier to detect the face
        face_cascade = cv2.CascadeClassifier("../haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray_fr, scaleFactor=1.5, minNeighbors=5)

        # Extract the face from the image and crop the image to focus on the face
        #TODO: training videos have no space to zoom out! Would have to reshoot for better results
        for (x, y, w, h) in faces:
            #face_image = fr[y-pad+height_shift_up:y+h+pad+height_shift_up, x-pad-width_shift_right:x+w+pad-width_shift_right] #extend the crop to a set amount of pixels each side
            face_image = fr[(y-pad+height_shift_up):y+h+pad+height_shift_up, x-pad:x+w+pad] #extend the crop to a set amount of pixels each side
            #print('{}, {}, {}, {}'.format(x, y, w, h))
            break

        # Resize the image to 256x256
        if face_image.any():
            resized_image = cv2.resize(face_image, (256, 256))
            cv2.imwrite(OUTPUT_FACE_COLOR_PATH + "/{}_{}.png".format(basename, ix), resized_image)

            # #Save images
            # plt.imshow(resized_image)
            # #print(resized_image.shape)
            # plt.axis('off')
            # plt.savefig( OUTPUT_FACE_COLOR_PATH + "/{}_{}.png".format(basename, ix), bbox_inches='tight')
            # plt.clf()
            sys.stdout.write(f"writing...{int((sm/length)*100)+1}%\n")
            sys.stdout.flush()

# images = load_img(OUTPUT_FACE_COLOR_PATH)
#Create augments from extracted images, append them to directory
# create_images(OUTPUT_FACE_COLOR_PATH, images, image_count*10)

i = 0
#Create mask for each image
for file in os.listdir(OUTPUT_FACE_COLOR_PATH):
    resized_augmented_image = cv2.imread("{}/{}".format(OUTPUT_FACE_COLOR_PATH,file))
    print(resized_augmented_image.shape)

    resized_augmented_gray = cv2.cvtColor(resized_augmented_image, cv2.COLOR_RGB2GRAY)
    print(resized_augmented_gray.shape)
    i += 1
    faces = detector(resized_augmented_gray)
    #print(faces)
    # plt.figure(figsize=(5, 5))
    # ax = plt.subplot(1, 1, 1)
    # plt.imshow(resized_augmented_converted_gray)
    # plt.show()
    if not faces:
        print('face not found for image {}!'.format(i))
    for face in faces:
        x1 = face.left() # left point
        y1 = face.top() # top point
        x2 = face.right() # right point
        y2 = face.bottom() # bottom point

        # Create landmark object
        landmarks = predictor(image=resized_augmented_gray, box=face)

        image = np.zeros((resized_augmented_image.shape[0], resized_augmented_image.shape[1], 3), np.uint8)
        # Loop through all the points
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            
            cv2.circle(img=image, center=(x, y), radius=2, color=(255,255,255), thickness=-1)
        cv2.imwrite(OUTPUT_FACE_MASK_PATH + "/{}".format(file), image)

        # plt.clf()
        # plt.imshow(image)
        # plt.axis('off')
        # plt.savefig(OUTPUT_FACE_MASK_PATH + "/{}".format(file), bbox_inches='tight')

rgb.release()

# Delay between every fram
cv2.waitKey(delay=0)

# Close all windows
cv2.destroyAllWindows()