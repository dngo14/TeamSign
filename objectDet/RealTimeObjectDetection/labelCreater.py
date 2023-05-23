<<<<<<< HEAD
import cv2
import os
import time
import uuid

IMG_PATH = 'Tensorflow/workspace/images/collectedimages'

labels = ['hello', 'thankyou', 'yes', 'no', 'iloveyou']
number_imgs = 15

for label in labels:
    !mkdir {'Tensorflow\workspace\images\collectedimages\\'+label}
    cap = cv2.VideoCapture(0)
    print(f'Collecting images for {label}')
    time.sleep(5)
    for img_num in range(number_imgs):
        _, frame = cap.read()
        image_name = os.path.join(IMG_PATH, label, label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(image_name, frame)
        cv2.imshow('frame', frame)
        time.sleep(2)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
=======
import cv2
import os
import time
import uuid

IMG_PATH = 'Tensorflow/workspace/images/collectedimages'

labels = ['hello', 'thankyou', 'yes', 'no', 'iloveyou']
number_imgs = 15

for label in labels:
    !mkdir {'Tensorflow\workspace\images\collectedimages\\'+label}
    cap = cv2.VideoCapture(0)
    print(f'Collecting images for {label}')
    time.sleep(5)
    for img_num in range(number_imgs):
        _, frame = cap.read()
        image_name = os.path.join(IMG_PATH, label, label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(image_name, frame)
        cv2.imshow('frame', frame)
        time.sleep(2)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
>>>>>>> ca217e5eede55c7bb75c01b4f962b9f24f19a443
            break