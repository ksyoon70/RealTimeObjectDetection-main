# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 19:50:19 2022

@author: headway
이 파일은 차량영상 및 번호판을 받으면, 번호판을 검지하여 인식을 시도하는 역할을 한다. 
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
import shutil

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import matplotlib.image as Image
import time
from plate_number_infer import *
from moto_plate_number_infer import *
from plate_recog_common import *
from plate_char_infer import *
from plate_hr_infer import *
from plate_vr_infer import *
from plate_or_infer import *
from label_tools import *
#로그에서 warining을 삭제할때 아래 코드를 사용한다.
import logging
logging.getLogger('tensorflow').disabled = True
import re

#========================
dataset_category = 'plate'
RESIZE_IMAGE_WIDTH = 320
RESIZE_IMAGE_HEIGHT = 320
#========================

WORKSPACE_PATH = os.path.join(ROOT_DIR,'Tensorflow','workspace')
ANNOTATION_PATH = os.path.join(WORKSPACE_PATH,'annotations')
IMAGE_PATH =  os.path.join(WORKSPACE_PATH,'images',dataset_category)
MODEL_PATH = os.path.join(WORKSPACE_PATH , 'models',dataset_category)
CONFIG_PATH = os.path.join( MODEL_PATH ,'my_ssd_mobnet','pipeline.config')
CHECKPOINT_PATH = os.path.join( MODEL_PATH , 'my_ssd_mobnet')

result_dir = os.path.join(IMAGE_PATH,'result')


plate_det_model = None

ndet_model = None
char_model = None
hr_model = None
vr_model = None
or_model = None
ncat_index = None
models = []

#번호판을 찾는 모듈을 초기화 하는 함수이다.
def plate_det_init_fn() :
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
    global plate_det_model
    plate_det_model = model_builder.build(model_config=configs['model'], is_training=False)
    
    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=plate_det_model)
    #ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-101')).expect_partial()
    #restore latest checkpoint
    ckpt.restore(tf.train.latest_checkpoint(CHECKPOINT_PATH))
    category_index = label_map_util.create_category_index_from_labelmap(os.path.join(ANNOTATION_PATH, 'platelabel_map.pbtxt'))
    
    return plate_det_model

# 차량 영상과 번호판 영상을 입력 받아 번호인식을 시도한다.
# 번호판을 기준으로 글자의 상대좌표를 반환한다.
# plate_label 은 번호판의 레이블이다. type1 ~ type13
def plateDetection(models, ncat_index, image_np, category, filename ,plate_np = None, plate_label = None) :
    
    plate_str = None
    plateTable = None
    category_index_temp = None
    CLASS_DIC = None
    class_index = None
    plate_box = None
    
    result_path = os.path.join(result_dir,filename)
    global plate_det_model
    if plate_det_model is None:
            plate_det_model = plate_det_init_fn()
            
    
    
    if image_np is not None and plate_np is None:
        # image_np는 차량영상 이다.
        src_height, src_width, scr_ch = image_np.shape
        src_box = [0,0,1,1]
        #pad 가 True이면 영상 아래 위로 black pad가 들어감.
        InsertPad = False
        #det_image_np = extract_sub_image(image_np,src_box,RESIZE_IMAGE_WIDTH,RESIZE_IMAGE_WIDTH,pad=InsertPad)
        #plt.imshow(image_np)
        #plt.show()
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = inner_detect_fn(input_tensor, plate_det_model)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        
        label_id_offset = 1
        image_np_with_detections = image_np.copy()
        
        #인식율이 일정값 이상이면 번호판을 추출한다.

        if detections['detection_scores'][0] > PLATE_THRESH_HOLD : #THRESH_HOLD :
            class_index = detections['detection_classes'][0]+label_id_offset #여기에서는 type13이 나오지 않는다 Error --> 추후 수정
            #print("'클래스:{0} 번호판 타입 {1} 확률:{2:.3f}".format(class_index,category_index[class_index]['name'],detections['detection_scores'][0]))
            #print('box= {}'.format(detections['detection_boxes'][0]))
            box = list(range(0,4))
            box = detections['detection_boxes'][0]
            height, width, ch = image_np.shape
            if InsertPad:
                if src_width >= src_height :
                    # x 좌표는 그대로 쓴다.
                    box_sx= int(width*box[1])
                    box_ex= int(width*box[3])
                    # y 좌표는 수정한다.
                    # 상위 black 부분 
                    up_black = (src_width - src_height)/2.0
                    box_sy = int(box[0]*src_width - up_black)
                    box_ey = int(box[2]*src_width - up_black)
                else :
                    # y 좌표는 그대로 쓴다.
                    box_sy= int(height*box[0])
                    box_ey= int(height*box[2])
                    # y 좌표는 수정한다.
                    # 좌측 black 부분 
                    left_black = (src_height - src_width)/2.0
                    box_sx = int(box[1]*src_height - left_black)
                    box_ex = int(box[3]*src_height - left_black)
            else:
                    # x 좌표는 그대로 쓴다.
                    box_sx= int(width*box[1])
                    box_ex= int(width*box[3])
                    # y 좌표는 그대로 쓴다.
                    box_sy= int(height*box[0])
                    box_ey= int(height*box[2])

            plate_img = image_np[box_sy:box_ey,box_sx:box_ex,:]
            plate_box = [[box_sx, box_ex, box_ex, box_sx],[box_sy,box_sy,box_ey,box_ey]]
            #plt.imshow(plate_np)
            #plt.show()
            #plate_img = cv2.cvtColor(plate_np, cv2.COLOR_BGR2RGB)                
            #번호판을 320x320 크기로 정규화 한다.
            desired_size = max(RESIZE_IMAGE_WIDTH,RESIZE_IMAGE_HEIGHT)
            old_size = [plate_img.shape[1],plate_img.shape[0]]
            ratio = float(desired_size)/max(old_size)
            new_size = tuple([int(x*ratio) for x in old_size])
            #원영상에서 ratio 만큼 곱하여 리싸이즈한 번호판 영상을 얻는다.
            cropped_img = cv2.resize(plate_img,new_size,interpolation=cv2.INTER_LINEAR)
            plate_new_img_np = np.zeros((desired_size, desired_size, 3), dtype = "uint8")
            h = new_size[1]
            w = new_size[0]
            yoff = round((desired_size-h)/2)
            xoff = round((desired_size-w)/2)
            #320x320영상에 번호판을 붙여 넣는다.
            plate_new_img_np[yoff:yoff+h, xoff:xoff+w , :] = cropped_img            
            #번호판에 대하여 문자 및 번호를 인식한다.
            if category == 'motorcycle':
                plate_str, plateTable,category_index_temp, CLASS_DIC,class_index = moto_plate_number_detect_fn(models,plate_new_img_np,ncat_index, platetype_index=class_index,result_path=result_path)
            else:
                plate_str, plateTable,category_index_temp, CLASS_DIC,class_index = plate_number_detect_fn(models,plate_new_img_np,ncat_index, platetype_index=class_index,result_path=result_path)
            
            half_dummy_ratio = float(yoff) / desired_size
            src_ratio = h / desired_size
            for ix in range(len(plateTable)):
                plateTable[ix][2] = (plateTable[ix][2] - half_dummy_ratio)/src_ratio
                plateTable[ix][4] = (plateTable[ix][4] - half_dummy_ratio)/src_ratio
            
        else:
            return plate_str, plateTable,category_index_temp, CLASS_DIC,class_index, plate_box
    
    elif plate_np is not None:
         #번호판을 320x320 크기로 정규화 한다.
            desired_size = max(RESIZE_IMAGE_WIDTH,RESIZE_IMAGE_HEIGHT)
            plate_img = plate_np
            #plate_img = cv2.cvtColor(plate_np, cv2.COLOR_BGR2RGB) 
            old_size = [plate_img.shape[1],plate_img.shape[0]]
            ratio = float(desired_size)/max(old_size)
            new_size = tuple([int(x*ratio) for x in old_size])
            #원영상에서 ratio 만큼 곱하여 리싸이즈한 번호판 영상을 얻는다.
            cropped_img = cv2.resize(plate_img,new_size,interpolation=cv2.INTER_LINEAR)
            plate_new_img_np = np.zeros((desired_size, desired_size, 3), dtype = "uint8")
            h = new_size[1]
            w = new_size[0]
            yoff = round((desired_size-h)/2)
            xoff = round((desired_size-w)/2)
            #320x320영상에 번호판을 붙여 넣는다.
            plate_new_img_np[yoff:yoff+h, xoff:xoff+w , :] = cropped_img            
            #번호판에 대하여 문자 및 번호를 인식한다.
            if plate_label is not None:
                if 'type' in plate_label:
                    class_index = int(re.sub(r'[^0-9]', '', plate_label))
                else:
                    class_index = 1
            else:
                class_index = 1
            
            if category == 'motorcycle':
                    plate_str, plateTable,category_index_temp, CLASS_DIC,class_index = moto_plate_number_detect_fn(models,plate_new_img_np,ncat_index, platetype_index=class_index,result_path=result_path)
            else:
                plate_str, plateTable,category_index_temp, CLASS_DIC,class_index = plate_number_detect_fn(models,plate_new_img_np,ncat_index, platetype_index=class_index,result_path=result_path)
            #plateTable 을 320x320 크기에서 원래 싸이즈로 바꾼다.
            # y 좌표는 pad가 들었 갔으므로 수정한다.
            half_dummy_ratio = float(yoff) / desired_size
            src_ratio = h / desired_size
            for ix in range(len(plateTable)):
                plateTable[ix][2] = (plateTable[ix][2] - half_dummy_ratio)/src_ratio
                plateTable[ix][4] = (plateTable[ix][4] - half_dummy_ratio)/src_ratio
            
            #plate_box 는 번호판 박스의 좌표
    return plate_str, plateTable,category_index_temp, CLASS_DIC,class_index, plate_box


def coordinationTrans(sx, sy, box) :
    
    arr = np.array(box)
    newBox = [arr[0] + sx, arr[1] + sy]
    return newBox
    
