import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

<<<<<<< HEAD
# Prepare data generator for standardizing frames before sending them into the model.
data_generator = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
=======
data_gen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
>>>>>>> ca217e5eede55c7bb75c01b4f962b9f24f19a443

# Loading the model.
MODEL_NAME = 'CNN_model.h5'  #########################################
#MODEL_NAME = "RNN_model.h5" #########################################
model = load_model(MODEL_NAME)

# Setting up the input image size and frame crop size.
<<<<<<< HEAD
IMAGE_SIZE = 50
CROP_SIZE = 300
=======
INPUT_SIZE = 50
FRAME_SIZE = 300
>>>>>>> ca217e5eede55c7bb75c01b4f962b9f24f19a443

# Creating list of available classes stored in classes.txt.
classes_file = open("classes.txt")
classes_string = classes_file.readline()
classes = classes_string.split()
classes.sort()  # The predict function sends out output in sorted order.

# Preparing cv2 for webcam feed
<<<<<<< HEAD
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame.
    ret, frame = cap.read()

    # Target area where the hand gestures should be.
    cv2.rectangle(frame, (0, 0), (CROP_SIZE, CROP_SIZE), (0, 255, 0), 3)
    
    # Preprocessing the frame before input to the model.
    cropped_image = frame[0:CROP_SIZE, 0:CROP_SIZE]
    resized_frame = cv2.resize(cropped_image, (IMAGE_SIZE, IMAGE_SIZE))
    reshaped_frame = (np.array(resized_frame)).reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3)) #CNN
    #reshaped_frame = (np.array(resized_frame)).reshape((1, 1, IMAGE_SIZE, IMAGE_SIZE, 3)) #LSTM
    frame_for_model = data_generator.standardize(np.float64(reshaped_frame))

    # Predicting the frame.
    prediction = np.array(model.predict(frame_for_model))
    predicted_class = classes[prediction.argmax()]      # Selecting the max confidence index.
=======
capture = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame.
    ret, frame = capture.read()

    # Target area where the hand gestures should be.
    cv2.rectangle(frame, (0, 0), (FRAME_SIZE, FRAME_SIZE), (0, 255, 0), 3)
    
    cropped_image = frame[0:FRAME_SIZE, 0:FRAME_SIZE]
    resized_frame = cv2.resize(cropped_image, (INPUT_SIZE, INPUT_SIZE))
    reshaped_frame = (np.array(resized_frame)).reshape((1, INPUT_SIZE, INPUT_SIZE, 3)) #CNN
    #reshaped_frame = (np.array(resized_frame)).reshape((1, 1, INPUT_SIZE, INPUT_SIZE, 3)) #Add for LSTM
    frame_for_model = data_gen.standardize(np.float64(reshaped_frame))

    # Predicting the frame.
    prediction = np.array(model.predict(frame_for_model))
    predicted_class = classes[prediction.argmax()]   
>>>>>>> ca217e5eede55c7bb75c01b4f962b9f24f19a443

    # Preparing output based on the model's confidence.
    prediction_probability = prediction[0, prediction.argmax()]
    if prediction_probability > 0.5:
        # High confidence.
        cv2.putText(frame, '{} - {:.2f}%'.format(predicted_class, prediction_probability * 100), 
                                    (10, 450), 1, 2, (255, 255, 0), 2, cv2.LINE_AA)
    elif prediction_probability > 0.2 and prediction_probability <= 0.5:
        # Low confidence.
        cv2.putText(frame, 'Maybe {}... - {:.2f}%'.format(predicted_class, prediction_probability * 100), 
                                    (10, 450), 1, 2, (0, 255, 255), 2, cv2.LINE_AA)
    else:
        # No confidence.
        cv2.putText(frame, classes[-2], (10, 450), 1, 2, (255, 255, 0), 2, cv2.LINE_AA)

<<<<<<< HEAD
    # Display the image with prediction.
    cv2.imshow('frame', frame)

    # Press q to quit
=======
    
    cv2.imshow('frame', frame)

>>>>>>> ca217e5eede55c7bb75c01b4f962b9f24f19a443
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

<<<<<<< HEAD
# When everything done, release the capture.
cap.release()
=======
capture.release()
>>>>>>> ca217e5eede55c7bb75c01b4f962b9f24f19a443
cv2.destroyAllWindows()