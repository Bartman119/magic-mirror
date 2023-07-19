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

OUTPUT_FACE_MASK_PATH = "../training_datasets/mom/output_face_mask"
OUTPUT_FACE_COLOR_PATH = "../training_datasets/mom/output_face_color"
OUTPUT_FACE_MASK_PATH_AUGMENTED = "../training_datasets/mom/output_face_mask_augmented"
OUTPUT_FACE_COLOR_PATH_AUGMENTED = "../training_datasets/mom/output_face_color_augmented"
OUTPUT_FACE_MASK_COMBINED_PATH = "../training_datasets/mom/output_face_mask_combined"
OUTPUT_FACE_COLOR_COMBINED_PATH = "../training_datasets/mom/output_face_color_combined"
RECORDING_PATH = '../raw_recordings/mom/mom.mp4'

# Define the amount of padding to be added
pad = 200
height_shift_up = 200
width_shift_right = 40

if not os.path.exists(OUTPUT_FACE_COLOR_PATH):
    os.makedirs(OUTPUT_FACE_COLOR_PATH)

if not os.path.exists(OUTPUT_FACE_MASK_PATH):
    os.makedirs(OUTPUT_FACE_MASK_PATH)

if not os.path.exists(OUTPUT_FACE_COLOR_PATH_AUGMENTED):
    os.makedirs(OUTPUT_FACE_COLOR_PATH_AUGMENTED)

if not os.path.exists(OUTPUT_FACE_MASK_PATH_AUGMENTED):
    os.makedirs(OUTPUT_FACE_MASK_PATH_AUGMENTED)

if not os.path.exists(OUTPUT_FACE_MASK_COMBINED_PATH):
    os.makedirs(OUTPUT_FACE_MASK_COMBINED_PATH)

if not os.path.exists(OUTPUT_FACE_COLOR_COMBINED_PATH):
    os.makedirs(OUTPUT_FACE_COLOR_COMBINED_PATH)

basename = "image"
image_count = 0
def load_img(indir):
    global image_count
    samples = []
    for file in os.listdir(indir):
        image = cv2.imread("{}/{}".format(indir,file))
        image = cv2.resize(image, (256,256))
        samples.append(image)
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


def create_images_augmented(basename, dir,images,number=500):
    total = 0
    for image in images:
        # Apply the augmentation techniques to the padded image using ImageDataGenerator
        augmented_img = aug.random_transform(image)

        total += 1
        cv2.imwrite(dir + "/{}_{}.png".format(basename, total), augmented_img)
        if total == number: 
            print("{} images generated to {}".format(total,dir))
            break
        if total%(number/100)==0: print('.',end='')

def create_masks(dir, dirOutput):
    i = 0
    #Create mask for each image
    for file in os.listdir(dir):
        resized_augmented_image = cv2.imread("{}/{}".format(dir,file))
        #print(resized_augmented_image.shape)

        resized_augmented_gray = cv2.cvtColor(resized_augmented_image, cv2.COLOR_RGB2GRAY)
        #print(resized_augmented_gray.shape)
        i += 1
        faces = detector(resized_augmented_gray)
        if not faces:
            print('face not found for image {}!'.format(i))
        for face in faces:
            # Create landmark object
            landmarks = predictor(image=resized_augmented_gray, box=face)
            image = np.zeros((resized_augmented_image.shape[0], resized_augmented_image.shape[1], 3), np.uint8)
            # Loop through all the points
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                
                cv2.circle(img=image, center=(x, y), radius=2, color=(255,255,255), thickness=-1)
            cv2.imwrite(dirOutput + "/{}".format(file), image)
            print('face {} saved successfully!'.format(i))

def combine_original_images_and_augmented_images():
    #iterate through mask directories and extract mask and its corresponding image
    for file in os.listdir(OUTPUT_FACE_MASK_PATH):
        read_mask_file = cv2.imread("{}/{}".format(OUTPUT_FACE_MASK_PATH,file))
        read_face_file = cv2.imread("{}/{}".format(OUTPUT_FACE_COLOR_PATH,file))
        cv2.imwrite(OUTPUT_FACE_MASK_COMBINED_PATH + "/{}".format(file), read_mask_file)
        cv2.imwrite(OUTPUT_FACE_COLOR_COMBINED_PATH + "/{}".format(file), read_face_file)

    for file in os.listdir(OUTPUT_FACE_MASK_PATH_AUGMENTED):
        read_mask_augmented_file = cv2.imread("{}/{}".format(OUTPUT_FACE_MASK_PATH_AUGMENTED,file))
        read_face_augmented_file = cv2.imread("{}/{}".format(OUTPUT_FACE_COLOR_PATH_AUGMENTED,file))
        cv2.imwrite(OUTPUT_FACE_MASK_COMBINED_PATH + "/{}".format(file), read_mask_augmented_file)
        cv2.imwrite(OUTPUT_FACE_COLOR_COMBINED_PATH + "/{}".format(file), read_face_augmented_file)

rgb = cv2.VideoCapture(RECORDING_PATH)
length = int(rgb.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(rgb.get(cv2.CAP_PROP_FPS))

def __get_data__():
    _, fr = rgb.read()
    gray = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)
    
    return fr, gray

ix = 0
width  = int(rgb.get(3)) # float
height = int(rgb.get(4))

images= []
for sm in range(1,length-1):
        ix += 1
        fr, gray_fr = __get_data__()

        # Use a face detection cascade classifier to detect the face
        face_cascade = cv2.CascadeClassifier("../haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray_fr, scaleFactor=1.5, minNeighbors=5)

        # Extract the face from the image and crop the image to focus on the face
        print(faces)
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_image = fr[(y-pad-height_shift_up):y+h+pad+height_shift_up, x-pad:x+w+pad] #extend the crop to a set amount of pixels each side
                break

            # Resize the image to 256x256
            if face_image.any():
                resized_image = cv2.resize(face_image, (256, 256))
                cv2.imwrite(OUTPUT_FACE_COLOR_PATH + "/{}_{}.png".format(basename, ix), resized_image)
                sys.stdout.write(f"writing...{int((sm/length)*100)+1}%\n")
                sys.stdout.flush()
                images.append(resized_image)
        else:
            print('face not found')

#create augmented dataset using extracted images
create_images_augmented('augment', OUTPUT_FACE_COLOR_PATH_AUGMENTED, images=images)

create_masks(OUTPUT_FACE_COLOR_PATH, OUTPUT_FACE_MASK_PATH)

create_masks(OUTPUT_FACE_COLOR_PATH_AUGMENTED, OUTPUT_FACE_MASK_PATH_AUGMENTED)

combine_original_images_and_augmented_images()

rgb.release()

# Delay between every fram
cv2.waitKey(delay=0)

# Close all windows
cv2.destroyAllWindows()