<<<<<<< HEAD
import tensorflow as tf
import cv2 
import numpy as np
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
IMAGE_PATH = WORKSPACE_PATH+'/images'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'

labels = [{'name':'hello', 'id':1},
           {'name':'yes', 'id':2},
           {'name':'no', 'id':3},
           {'name':'thank you', 'id':4},
           {'name':'i love you', 'id':5},]

with open(ANNOTATION_PATH + '\label_map.pbtxt', 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 

CONFIG_PATH = MODEL_PATH+'/'+CUSTOM_MODEL_NAME+'/pipeline.config'

config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(CONFIG_PATH, "r") as f:                                                                                                                                                                                                                     
    proto_str = f.read()                                                                                                                                                                                                                                          
    text_format.Merge(proto_str, pipeline_config) 

pipeline_config.model.ssd.num_classes = len(labels)
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = PRETRAINED_MODEL_PATH+'/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-2'
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= ANNOTATION_PATH + '/label_map.pbtxt'
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/train.record']
pipeline_config.eval_input_reader[0].label_map_path = ANNOTATION_PATH + '/label_map.pbtxt'
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/test.record']

config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
with tf.io.gfile.GFile(CONFIG_PATH, "wb") as f:                                                                                                                                                                                                                     
    f.write(config_text)   

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-2')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')

# Setup capture
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
detection_box_size = 0
count = 0


while True: 
    ret, frame = cap.read()
    image_np = np.array(frame)

    # Apply blurring
    image_np_blurred = cv2.GaussianBlur(image_np, (5, 5), 0)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np_blurred, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                None,
                None,
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=1,
                min_score_thresh=.92,
                agnostic_mode=False)

    # Set the threshold for object detection
    threshold = 0.92


    ####IMPORTANT
    #For integration purposes, the following code includes a count variable that ensures every image detected is saved into the cropped_imgs directory. This is mostly for testing purposes. When this code is used with the CNN or RNN model, the count variable should either be gotten rid of or made static with a value of 0 or somehting of the like. THis way, you could feed 'cropped_img/detected_object_0.jpg' into your CNN model, and the img will always be the most recent detection. Alternatively, you could feed multiple images (say 10) into the model and average the results over a given time period (easy to calculate with frames). Just something to keep in mind. -- Ethan

    # Iterate over the detected objects
    for i in range(min(num_detections, 3)):
        if detections['detection_scores'][i] >= threshold:
            # Extract the coordinates of the bounding box
            ymin, xmin, ymax, xmax = detections['detection_boxes'][i]

            # Convert the box coordinates from normalized values to pixel values
            im_height, im_width, _ = image_np.shape
            (left, right, top, bottom) = (int(xmin * im_width), int(xmax * im_width),
                                        int(ymin * im_height), int(ymax * im_height))

            # Crop the detected object from the image
            object_image = image_np[top:bottom, left:right]

            # Save the cropped object image
            image_filename = f'cropped_imgs/detected_object_{count}.jpg'
            cv2.imwrite(image_filename, object_image)

            # Convert the object image to grayscale
            object_image_gray = cv2.cvtColor(object_image, cv2.COLOR_BGR2GRAY)

            # Resize the grayscale image to 200x200
            object_image_resized = cv2.resize(object_image_gray, (200, 200))

            # Save the resized grayscale image
            image_filename = f'cropped_imgs/detected_object_grey_{count}.jpg'
            cv2.imwrite(image_filename, object_image_resized)

            # Display the saved image filenamel
            print(f'Saved image: {image_filename}')
            count +=1
    

    #resets count so that no more than 100 imgs are saved to the computer at one time. Safer that way.
    if count >= 100:
        count = 0

   #### Older Implimentation
   # if num_detections > i and detections['detection_scores'][i] >= 0.50:
   #         # Extract the coordinates of the bounding box for the i-th detection
   #     ymin, xmin, ymax, xmax = detections['detection_boxes'][i]
   #         # Convert the coordinates from relative to absolute pixel values
   #     im_height, im_width, _ = image_np.shape
   #     (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
   #                                     ymin * im_height, ymax * im_height)
   #         # Crop the image based on the bounding box coordinates
   #     cropped_image = image_np[int(top):int(bottom), int(left):int(right)]
   #         # Save the cropped image to a file
   #     cv2.imwrite("cropped_imgs/cropped_image_{}.jpg".format(i), cropped_image)
   #     i+=1

    cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break

=======
import tensorflow as tf
import cv2 
import numpy as np
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
IMAGE_PATH = WORKSPACE_PATH+'/images'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'

labels = [{'name':'hello', 'id':1},
           {'name':'yes', 'id':2},
           {'name':'no', 'id':3},
           {'name':'thank you', 'id':4},
           {'name':'i love you', 'id':5},]

with open(ANNOTATION_PATH + '\label_map.pbtxt', 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 

CONFIG_PATH = MODEL_PATH+'/'+CUSTOM_MODEL_NAME+'/pipeline.config'

config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(CONFIG_PATH, "r") as f:                                                                                                                                                                                                                     
    proto_str = f.read()                                                                                                                                                                                                                                          
    text_format.Merge(proto_str, pipeline_config) 

pipeline_config.model.ssd.num_classes = len(labels)
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = PRETRAINED_MODEL_PATH+'/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-2'
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= ANNOTATION_PATH + '/label_map.pbtxt'
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/train.record']
pipeline_config.eval_input_reader[0].label_map_path = ANNOTATION_PATH + '/label_map.pbtxt'
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/test.record']

config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
with tf.io.gfile.GFile(CONFIG_PATH, "wb") as f:                                                                                                                                                                                                                     
    f.write(config_text)   

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-2')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')

# Setup capture
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
detection_box_size = 0
count = 0


while True: 
    ret, frame = cap.read()
    image_np = np.array(frame)

    # Apply blurring
    image_np_blurred = cv2.GaussianBlur(image_np, (5, 5), 0)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np_blurred, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                None,
                None,
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=1,
                min_score_thresh=.92,
                agnostic_mode=False)

    # Set the threshold for object detection
    threshold = 0.92


    ####IMPORTANT
    #For integration purposes, the following code includes a count variable that ensures every image detected is saved into the cropped_imgs directory. This is mostly for testing purposes. When this code is used with the CNN or RNN model, the count variable should either be gotten rid of or made static with a value of 0 or somehting of the like. THis way, you could feed 'cropped_img/detected_object_0.jpg' into your CNN model, and the img will always be the most recent detection. Alternatively, you could feed multiple images (say 10) into the model and average the results over a given time period (easy to calculate with frames). Just something to keep in mind. -- Ethan

    # Iterate over the detected objects
    for i in range(min(num_detections, 3)):
        if detections['detection_scores'][i] >= threshold:
            # Extract the coordinates of the bounding box
            ymin, xmin, ymax, xmax = detections['detection_boxes'][i]

            # Convert the box coordinates from normalized values to pixel values
            im_height, im_width, _ = image_np.shape
            (left, right, top, bottom) = (int(xmin * im_width), int(xmax * im_width),
                                        int(ymin * im_height), int(ymax * im_height))

            # Crop the detected object from the image
            object_image = image_np[top:bottom, left:right]

            # Save the cropped object image
            image_filename = f'cropped_imgs/detected_object_{count}.jpg'
            cv2.imwrite(image_filename, object_image)

            # Convert the object image to grayscale
            object_image_gray = cv2.cvtColor(object_image, cv2.COLOR_BGR2GRAY)

            # Resize the grayscale image to 200x200
            object_image_resized = cv2.resize(object_image_gray, (200, 200))

            # Save the resized grayscale image
            image_filename = f'cropped_imgs/detected_object_grey_{count}.jpg'
            cv2.imwrite(image_filename, object_image_resized)

            # Display the saved image filenamel
            print(f'Saved image: {image_filename}')
            count +=1
    

    #resets count so that no more than 100 imgs are saved to the computer at one time. Safer that way.
    if count >= 100:
        count = 0

   #### Older Implimentation
   # if num_detections > i and detections['detection_scores'][i] >= 0.50:
   #         # Extract the coordinates of the bounding box for the i-th detection
   #     ymin, xmin, ymax, xmax = detections['detection_boxes'][i]
   #         # Convert the coordinates from relative to absolute pixel values
   #     im_height, im_width, _ = image_np.shape
   #     (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
   #                                     ymin * im_height, ymax * im_height)
   #         # Crop the image based on the bounding box coordinates
   #     cropped_image = image_np[int(top):int(bottom), int(left):int(right)]
   #         # Save the cropped image to a file
   #     cv2.imwrite("cropped_imgs/cropped_image_{}.jpg".format(i), cropped_image)
   #     i+=1

    cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break

>>>>>>> ca217e5eede55c7bb75c01b4f962b9f24f19a443
    detections = detect_fn(input_tensor)