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
from plate_number_infer import *
from plate_recog_common import *
from plate_char_infer import *


#========================
# 여기의 내용을 용도에 맞게 수정한다.
dataset_category='plate'
test_dir_name = 'test'
show_image = True
save_image = True
THRESH_HOLD = 0.8
#========================

WORKSPACE_PATH = os.path.join(ROOT_DIR,'Tensorflow','workspace')
ANNOTATION_PATH = os.path.join(WORKSPACE_PATH,'annotations')
IMAGE_PATH =  os.path.join(WORKSPACE_PATH,'images',dataset_category)
MODEL_PATH = os.path.join(WORKSPACE_PATH , 'models',dataset_category)
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
#ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-101')).expect_partial()
#restore latest checkpoint
ckpt.restore(tf.train.latest_checkpoint(CHECKPOINT_PATH))
category_index = label_map_util.create_category_index_from_labelmap(os.path.join(ANNOTATION_PATH, 'platelabel_map.pbtxt'))

#문자 번호 인식용 모델 load
ndet_model, ncat_index = number_det_init_fn()      # ncat_index 글자 추출 카테고리 인덱스

#문자 모델 초기화
char_model =  char_det_init_fn()
    
fileNum = len(os.listdir(images_dir))
cnt = 0;

models = [ndet_model, char_model]

for filename in os.listdir(images_dir):
    cnt += 1
    image_path = os.path.join(images_dir,filename)
    if os.path.exists(image_path) :
        imgRGB  = imread(image_path)
        #imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_np = np.array(imgRGB)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor, detection_model)
        
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        
        label_id_offset = 1
        image_np_with_detections = image_np.copy()
        
        #인식율이 일정값 이상이면 번호판을 추출한다.

        if detections['detection_scores'][0] > THRESH_HOLD :
            class_index = detections['detection_classes'][0]+label_id_offset
            print("'클래스:{0} 번호판 타입 {1} 확률:{2:.3f}".format(class_index,category_index[class_index]['name'],detections['detection_scores'][0]))
            print('box= {}'.format(detections['detection_boxes'][0]))
            box = list(range(0,4))
            box = detections['detection_boxes'][0]
            height, width, ch = image_np.shape
            box_sy = int(height*box[0])
            box_sx= int(width*box[1])
            box_ey = int(height*box[2])
            box_ex= int(width*box[3])
            plate_np = image_np[box_sy:box_ey,box_sx:box_ex,:]
            
            plate_img = cv2.cvtColor(plate_np, cv2.COLOR_BGR2RGB)                
            #번호판을 320x320 크기로 정규화 한다.
            RESIZE_IMAGE_WIDTH = 320
            RESIZE_IMAGE_HEIGHT = 320
            desired_size = max(RESIZE_IMAGE_WIDTH,RESIZE_IMAGE_HEIGHT)
            old_size = [plate_img.shape[1],plate_img.shape[0]]
            ratio = float(desired_size)/max(old_size)
            new_size = tuple([int(x*ratio) for x in old_size])
            #원영상에서 ratio 만큼 곱하여 리싸이즈한 번호판 영상을 얻는다.
            cropped_img = cv2.resize(plate_img,new_size,interpolation=cv2.INTER_LINEAR)
            plate_new_img_np = np.zeros((desired_size, desired_size, 3), dtype = "uint8")
            plate_new_img = cv2.cvtColor(plate_new_img_np, cv2.COLOR_BGR2RGB)
            h = new_size[1]
            w = new_size[0]
            yoff = round((desired_size-h)/2)
            xoff = round((desired_size-w)/2)
            #320x320영상에 번호판을 붙여 넣는다.
            plate_new_img[yoff:yoff+h, xoff:xoff+w , :] = cropped_img
            # plt.imshow(plate_new_img)
            # plt.show()
            #번호판에 대하여 문자 및 번호를 인식한다.
            plate_number_detect_fn(models,plate_new_img,ncat_index)
                
            
            
            
            #numpy에서 번호판 영상을 오려낸다.
            
        
        
        # viz_utils.visualize_boxes_and_labels_on_image_array(
        #             image_np_with_detections,
        #             detections['detection_boxes'],
        #             detections['detection_classes']+label_id_offset,
        #             detections['detection_scores'],
        #             category_index,
        #             use_normalized_coordinates=True,
        #             max_boxes_to_draw=20,
        #             min_score_thresh=.10,
        #             agnostic_mode=False)
        
        
        # save_fig_path = os.path.join(result_dir,filename)
        
        # resultImage = cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB)
        
        # if save_image :
        #     if not (save_fig_path is None):
        #         plt.imsave(save_fig_path, resultImage)

        
        
            





