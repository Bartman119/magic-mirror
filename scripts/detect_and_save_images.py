import cv2
import numpy as np
import dlib
import sys
import matplotlib.pyplot as plt

#TODO: input video has too much side movement and face detection and stretch algorithm is making mistakes!

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")

rgb = cv2.VideoCapture('../raw_recordings/smiling_lady.mp4')
length = int(rgb.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(rgb.get(cv2.CAP_PROP_FPS))

def __get_data__():
    _, fr = rgb.read()
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    
    return fr, gray

ix = 0
width  = int(rgb.get(3)) # float
height = int(rgb.get(4))
fourcc = cv2.VideoWriter_fourcc(*'vp90')
PATH = '../raw_recordings/smiling_lady_face_mask.webm'
output = cv2.VideoWriter(PATH,fourcc, fps, (width,height))

for sm in range(1,length-1):
        ix += 1
        fr, gray_fr = __get_data__()

        #add preprocessing of an image:
        #extract just the face and stretch it to 256x256 

        # Use a face detection cascade classifier to detect the face
        face_cascade = cv2.CascadeClassifier("../haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray_fr, scaleFactor=1.1, minNeighbors=5)

        # Extract the face from the image and crop the image to focus on the face
        for (x, y, w, h) in faces:
            face_image = fr[y:y+h, x:x+w]
            break

        # Resize the image to 256x256
        resized_image = cv2.resize(face_image, (256, 256))

        faces = detector(resized_image)
        for face in faces:
            x1 = face.left() # left point
            y1 = face.top() # top point
            x2 = face.right() # right point
            y2 = face.bottom() # bottom point

            # Create landmark object
            landmarks = predictor(image=resized_image, box=face)

            image = np.zeros((resized_image.shape[0], resized_image.shape[1], 3), np.uint8)
            # Loop through all the points
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y

                # Draw a circle
                #cv2.circle(img=fr, center=(x, y), radius=3, color=(255, 255, 255), thickness=-1)

                #Create mask image
                
                cv2.circle(img=image, center=(x, y), radius=3, color=(255, 255, 255), thickness=-1)


        sys.stdout.write(f"writing...{int((sm/length)*100)+1}%\n")
        sys.stdout.flush()
        output.write(image)
        plt.imshow(resized_image)
        #data = plt.imread(gray_fr)
        basename = "test"
        plt.axis('off')
        plt.savefig("../training_datasets/smiling_lady/output_face_color/{}_{}.png".format(basename, ix), bbox_inches='tight')
        plt.clf()
        plt.imshow(image)
        plt.axis('off')
        plt.savefig("../training_datasets/smiling_lady/output_face_mask/{}_{}.png".format(basename, ix), bbox_inches='tight')
rgb.release()

# Delay between every fram
cv2.waitKey(delay=0)

# Close all windows
cv2.destroyAllWindows()