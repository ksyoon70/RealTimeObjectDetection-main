# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 19:50:19 2022

@author: headway
"""
import os,sys
import cv2
import colorsys
import time 
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import matplotlib.image as Image
import time


#========================
# 여기의 내용을 용도에 맞게 수정한다.
dataset_category='car-plate'
test_dir_name = 'test'
show_image = True
save_image = True
#========================


ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR) 


WORKSPACE_PATH = os.path.join(ROOT_DIR,'Tensorflow','workspace')
ANNOTATION_PATH = os.path.join(WORKSPACE_PATH,'annotations')
IMAGE_PATH =  os.path.join(WORKSPACE_PATH,'images',dataset_category)
MODEL_PATH = os.path.join(WORKSPACE_PATH , 'models')
CONFIG_PATH = os.path.join( MODEL_PATH ,'my_ssd_mobnet','pipeline.config')
CHECKPOINT_PATH = os.path.join( MODEL_PATH , 'my_ssd_mobnet')


#테스트할 이미지 디렉토리
images_dir = os.path.join(IMAGE_PATH,test_dir_name)
result_dir = os.path.join(IMAGE_PATH,'result')

if not os.path.isdir(result_dir):
	os.mkdir(result_dir)



# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-11')).expect_partial()



@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


category_index = label_map_util.create_category_index_from_labelmap(os.path.join(ANNOTATION_PATH, 'label_map.pbtxt'))


def drawImage(image_path):
    image  = cv2.imread(image_path)
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(imgRGB)
    plt.pause(0.5)
    plt.show()

fileNum = len(os.listdir(images_dir))
cnt = 0;
for filename in os.listdir(images_dir):
    cnt += 1
    image_path = os.path.join(images_dir,filename)
    if os.path.exists(image_path) :
        image  = cv2.imread(image_path)
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_np = np.array(imgRGB)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
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
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=.5,
                    agnostic_mode=False)
        
        
        save_fig_path = os.path.join(result_dir,filename)
        
        resultImage = cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB)
        
        if save_image :
            if not (save_fig_path is None):
                plt.imsave(save_fig_path, resultImage)

        
        if show_image :
            plt.imshow(resultImage)
            plt.show()
            





