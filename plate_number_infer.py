# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 16:16:47 2022

@author: headway
"""

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
import copy

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import matplotlib.image as Image
import time
from plate_recog_common import *
from plate_char_infer import *
from plate_hr_infer import *
from plate_vr_infer import *
from plate_or_infer import *
from label_tools import predictPlateNumber, predictPlateNumberODAPI


#========================
# 여기의 내용을 용도에 맞게 수정한다.
dataset_category='plateimage'
test_dir_name = 'test'
show_image = True
save_image = True
#========================
WORKSPACE_PATH = os.path.join(ROOT_DIR,'Tensorflow','workspace')
ANNOTATION_PATH = os.path.join(WORKSPACE_PATH,'annotations')
PIMAGE_PATH =  os.path.join(WORKSPACE_PATH,'images',dataset_category)
PMODEL_PATH = os.path.join(WORKSPACE_PATH , 'models',dataset_category)
PCONFIG_PATH = os.path.join( PMODEL_PATH ,'my_ssd_mobnet','pipeline.config')
PCHECKPOINT_PATH = os.path.join( PMODEL_PATH , 'my_ssd_mobnet')

category_index = None

def number_det_init_fn():
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(PCONFIG_PATH)
    number_det_model = model_builder.build(model_config=configs['model'], is_training=False)
    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=number_det_model)
    #ckpt.restore(os.path.join(CHECKPOINT_PAdTH, 'ckpt-101')).expect_partial()
    #restore latest checkpoint
    ckpt.restore(tf.train.latest_checkpoint(PCHECKPOINT_PATH))
    global ANNOTATION_PATH
    category_index = label_map_util.create_category_index_from_labelmap(os.path.join(ANNOTATION_PATH, 'char_number_label_map.pbtxt'))

    fLabels = pd.read_csv(DEFAULT_LABEL_FILE, header = None )
    global LABEL_FILE_CLASS
    LABEL_FILE_CLASS = fLabels[0].values.tolist()
    LABEL_FILE_HUMAN_NAMES = fLabels[1].values.tolist()
    global CLASS_DIC    
    CLASS_DIC = dict(zip(LABEL_FILE_CLASS, LABEL_FILE_HUMAN_NAMES))

    return number_det_model, category_index

@tf.function
def number_det_fn(image, detection_model):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections



def plate_number_detect_fn(models, imageRGB, category_index,platetype_index) :

    image_np = imageRGB
    ndet_model = models[0]
    cdet_model = models[1]
    hr_det_model = models[2]
    vr_det_model = models[3]
    or_det_model = models[4]
    
    pinput_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = number_det_fn(pinput_tensor,ndet_model)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    
    """
    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=20,
                min_score_thresh=.10,
                agnostic_mode=False)
    """
    #검지 class가 'char' 이면 문자 검출을 한다.
    # 객체 인식율을 정수로 변환
    # scores = detections['detection_scores'][0:num_detections]
    # class_ids = detections['detection_classes']+label_id_offset
    # intscore = list(map(int, [x*y for x,y in zip(scores, [100] * len(scores))]))
    
    #Char, vReg, hReg, oReg, 
    ch = None
    category_index_temp = copy.deepcopy(category_index)
    for index, cindex in enumerate(detections['detection_classes']+label_id_offset) :
        if category_index[cindex]['name'] == 'Char' :
            det_image_np = extract_sub_image(image_np,detections['detection_boxes'][index],IMG_SIZE,IMG_SIZE,fixratio=False)
            ch = char_det_fn(cdet_model,det_image_np)
            category_index_temp[cindex]['name'] = ch
        if category_index[cindex]['name'] == 'hReg' :
            det_image_np = extract_sub_image(image_np,detections['detection_boxes'][index],IMG_SIZE,IMG_SIZE,fixratio=False)
            ch = hr_det_fn(hr_det_model,det_image_np)
            category_index_temp[cindex]['name'] = ch
        if category_index[cindex]['name'] == 'vReg' :
            det_image_np = extract_sub_image(image_np,detections['detection_boxes'][index],IMG_SIZE,IMG_SIZE,fixratio=False)
            ch = vr_det_fn(vr_det_model,det_image_np)
            category_index_temp[cindex]['name'] = ch
        if category_index[cindex]['name'] == 'oReg' :
            det_image_np = extract_sub_image(image_np,detections['detection_boxes'][index],IMG_SIZE,IMG_SIZE,fixratio=False)
            ch = or_det_fn(or_det_model,det_image_np)
            category_index_temp[cindex]['name'] = ch
    
    plate_str =  predictPlateNumberODAPI(detections,platetype_index,category_index_temp, CLASS_DIC)
  
  
    
    if show_image :
        plt.imshow(image_np_with_detections)
        plt.show()
        
    return plate_str
            





