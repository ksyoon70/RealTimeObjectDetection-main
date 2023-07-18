# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 15:34:48 2022
# 이륜차 번호판을 인식하기 위한 모듈이다.
@author: 윤경섭
"""

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
from moto_plate_char_infer import *
from moto_plate_hr_infer import *
from label_tools import *
from CRNN_Model import *
import pandas as pd


#========================
# 여기의 내용을 용도에 맞게 수정한다.
dataset_category='mplateimage'
test_dir_name = 'test'
show_image = False
save_image = True
save_char = False                # 문자영역을 저장할지 여부
CHAR_SAVE_FOLDER_NAME = 'char'
CRNN_MODEL_USE = False           # CRNN 모델을 사용할지 여부
REG_CRNN_MODEL_USE = False       #지역번판에 CRNN 사용여부
crnn_categories = []
crnn_cat_filename = 'chcrnn_categories.txt'
reg_crnn_categories = []
reg_crnn_cat_filename = 'regcrnn_categories.txt'
CHAR_CRNN_MODEL_DIR = 'm_char_crnn_model'      #CRNN 모델 위치 
REG_CRNN_MODEL_DIR = 'm_reg_crnn_model'      #CRNN 모델 위치 
CH_THRESH_HOLD = 0.7
MOTO_CH_THRESH_HOLD = 0.4
HR_THRESH_HOLD = 0.5
MOTO_HR_THRESH_HOLD = 0.4
DEFAULT_LABEL_FILE = "./LPR_PlateImage_Labels.txt"  #라벨 파일이름
#========================
WORKSPACE_PATH = os.path.join(ROOT_DIR,'Tensorflow','workspace')
ANNOTATION_PATH = os.path.join(WORKSPACE_PATH,'annotations')
PIMAGE_PATH =  os.path.join(WORKSPACE_PATH,'images',dataset_category)
PMODEL_PATH = os.path.join(WORKSPACE_PATH , 'models',dataset_category)
PCONFIG_PATH = os.path.join( PMODEL_PATH ,'my_ssd_mobnet','pipeline.config')
PCHECKPOINT_PATH = os.path.join( PMODEL_PATH , 'my_ssd_mobnet')

category_index = None

crnn_model = None
CHAR_CRNN_MODEL_PATH = None
CHAR_CRNN_WEIGHT_PATH = None

reg_crnn_model = None
REG_CRNN_MODEL_PATH = None
REG_CRNN_WEIGHT_PATH = None
def moto_number_det_init_fn():
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
    LABEL_FILE_CLASS.append('x')
    LABEL_FILE_HUMAN_NAMES = fLabels[1].values.tolist()
    LABEL_FILE_HUMAN_NAMES.append('x')
    global CLASS_DIC
    
    CLASS_DIC = dict(zip(LABEL_FILE_CLASS, LABEL_FILE_HUMAN_NAMES))
    
    global REV_CLASS_DIC
    REV_CLASS_DIC = dict(zip(LABEL_FILE_HUMAN_NAMES[LABEL_FILE_HUMAN_NAMES.index('가'):LABEL_FILE_HUMAN_NAMES.index('○') + 1] + LABEL_FILE_HUMAN_NAMES[LABEL_FILE_HUMAN_NAMES.index('강'):LABEL_FILE_HUMAN_NAMES.index('흥') + 1] ,LABEL_FILE_CLASS[LABEL_FILE_CLASS.index('Ga'):LABEL_FILE_CLASS.index('Cml') + 1] + LABEL_FILE_CLASS[LABEL_FILE_CLASS.index('Gang'):LABEL_FILE_CLASS.index('Heung') + 1]))
    REV_CLASS_DIC['x'] = 'x'
    
    global REV_HCLASS_DIC
    REV_HCLASS_DIC = dict(zip(LABEL_FILE_HUMAN_NAMES[LABEL_FILE_HUMAN_NAMES.index('서울'):LABEL_FILE_HUMAN_NAMES.index('울산') + 1],LABEL_FILE_CLASS[LABEL_FILE_CLASS.index('hSeoul'):LABEL_FILE_CLASS.index('hUlSan') + 1]))
    REV_HCLASS_DIC['x'] = 'x'
    
    global crnn_model
    global crnn_categories
    global reg_crnn_model
    global reg_crnn_categories
    if CRNN_MODEL_USE:
        #용도문자 CRNN 설정
        filelist =  os.listdir(os.path.join(ROOT_DIR,CHAR_CRNN_MODEL_DIR))
        
        for fn in filelist :
            if 'model' in fn.lower() :
                CHAR_CRNN_MODEL_PATH = os.path.join(ROOT_DIR,CHAR_CRNN_MODEL_DIR,fn)
               
            if 'weight' in fn.lower():
                #read weight value from trained dir
                CHAR_CRNN_WEIGHT_PATH = os.path.join(ROOT_DIR,CHAR_CRNN_MODEL_DIR,fn)
                
        
        
        CRNN_CATEGORIES_FILE_PATH = os.path.join(ROOT_DIR,CHAR_CRNN_MODEL_DIR,crnn_cat_filename)
    
        file = open(CRNN_CATEGORIES_FILE_PATH, "r")
        while True:
            line = file.readline()
            if not line:
                break
            crnn_categories.append(line.strip())
    
        file.close()
        crnn_model = CRNN_Model(model_path=CHAR_CRNN_MODEL_PATH,weight_path=CHAR_CRNN_WEIGHT_PATH,characters = crnn_categories ,max_length=1)
    #지역 CRNN 설청
    if REG_CRNN_MODEL_USE :
        filelist =  os.listdir(os.path.join(ROOT_DIR,REG_CRNN_MODEL_DIR))
        for fn in filelist :
            if 'model' in fn :
                REG_CRNN_MODEL_PATH = os.path.join(ROOT_DIR,REG_CRNN_MODEL_DIR,fn)
           
            if 'weight' in fn:
                #read weight value from trained dir
                REG_CRNN_WEIGHT_PATH = os.path.join(ROOT_DIR,REG_CRNN_MODEL_DIR,fn)
                
        REG_CRNN_CATEGORIES_FILE_PATH = os.path.join(ROOT_DIR,REG_CRNN_MODEL_DIR,reg_crnn_cat_filename)

        file = open(REG_CRNN_CATEGORIES_FILE_PATH, "r")
        while True:
            line = file.readline()
            if not line:
                break
            reg_crnn_categories.append(line.strip())

        file.close()
        reg_crnn_model = CRNN_Model(model_path=REG_CRNN_MODEL_PATH,weight_path=REG_CRNN_WEIGHT_PATH,characters = reg_crnn_categories ,max_length=2)

    return number_det_model, category_index

@tf.function
def moto_number_det_fn(image, detection_model):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections



def moto_plate_number_detect_fn(models, imageRGB, category_index,platetype_index,result_path) :

    image_np = imageRGB
    ndet_model = models[5]
    cdet_model = models[6]
    hr_det_model = models[7]

    
    pinput_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = moto_number_det_fn(pinput_tensor,ndet_model)
    
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
                max_boxes_to_draw=20,
                min_score_thresh=.10,
                agnostic_mode=False)
    
    #검지 class가 'char' 이면 문자 검출을 한다.
    # 객체 인식율을 정수로 변환
    # scores = detections['detection_scores'][0:num_detections]
    # class_ids = detections['detection_classes']+label_id_offset
    # intscore = list(map(int, [x*y for x,y in zip(scores, [100] * len(scores))]))
    
    #Char, vReg, hReg, oReg, 
    ch = None
    twoLinePlate = False
    category_index_temp = copy.deepcopy(category_index)
    plate_class_id = []  #인식한 문자들...숫자로가지고 있어야...
    for index, cindex in enumerate(detections['detection_classes']+label_id_offset) :
        if category_index[cindex]['name'] == 'Char' :
            det_image_np = extract_sub_image(image_np,detections['detection_boxes'][index],IMG_SIZE,IMG_SIZE,pad=False)
            if CRNN_MODEL_USE:
                #CRNN 모델을 사용하여 문자를 추출합니다.
                crnn_image_np = np.swapaxes(det_image_np,0,1)
                crnn_image_np = np.expand_dims(crnn_image_np,0)
                ch_crnn, probs = crnn_model.predict(crnn_image_np)
                ch = ch_crnn[0]
                if ch == '[UNK]' or probs[0] <= MOTO_CH_THRESH_HOLD:
                    ch = 'x'
                category_index_temp[cindex]['name'] = REV_CLASS_DIC[ch]
                #검시 확률을 업데이트 한다.
                detections['detection_scores'][index] = probs[0] 
                print('한글인식 {} 확률 {:.2f}'.format(ch,probs[0]*100))
            else:
                ch, prob = moto_char_det_fn(cdet_model,det_image_np,ch_thresh_hold=MOTO_CH_THRESH_HOLD,predict_anyway=save_char)
                plate_class_id.append(LABEL_FILE_CLASS.index(REV_CLASS_DIC[ch]))
            if save_char:
                # 문자영상을 저장하고 싶으면 여기서 저장한다.
                # 저장 경로 루트
                result_path_root, filename = os.path.split(result_path)   # 경로와 파일 분리
                result_path_char = os.path.join(result_path_root,CHAR_SAVE_FOLDER_NAME)
                if not os.path.isdir(result_path_char):
                    os.mkdir(result_path_char)
                basefilename, ext = os.path.splitext(filename)
                result_save_filename_ch = basefilename + '_' + ch + ext  # 저장할 파일명을 만든다.
                result_save_fullpath_ch = os.path.join(result_path_char,result_save_filename_ch)
                det_image_np = cv2.cvtColor(det_image_np, cv2.COLOR_RGB2BGR)
                imwrite( result_save_fullpath_ch, det_image_np)
                
        elif category_index[cindex]['name'] == 'hReg' :
            det_image_np = extract_sub_image(image_np,detections['detection_boxes'][index],IMG_SIZE,IMG_SIZE,pad=False)
            if REG_CRNN_MODEL_USE :
                #CRNN 모델을 사용하여 문자를 추출합니다.
                crnn_image_np = np.swapaxes(det_image_np,0,1)
                crnn_image_np = np.expand_dims(crnn_image_np,0)
                ch_crnn, probs = reg_crnn_model.predict(crnn_image_np)
                ch = ch_crnn[0]
                if ch == '[UNK]' or probs[0] <= MOTO_HR_THRESH_HOLD:
                    ch = 'x'
                else:
                    find, ch = checkKeyinRegionDictionary(REV_HCLASS_DIC,ch)
                    if not find :
                        ch = 'x'

                category_index_temp[cindex]['name'] = REV_HCLASS_DIC[ch]
                #검시 확률을 업데이트 한다.
                detections['detection_scores'][index] = probs[0] 
                print('H지역 {} 확률 {:.2f}'.format(ch,probs[0]*100))
            else :
                ch, prob = moto_hr_det_fn(hr_det_model,det_image_np,hr_thresh_hold=MOTO_HR_THRESH_HOLD)
                plate_class_id.append(LABEL_FILE_CLASS.index(REV_HCLASS_DIC[ch]))
                
        else:
            plate_class_id.append(cindex - 1)
            
    twoLinePlate = True
    platetype_index = 13   #type 13 번호판
    
    plate_str, plateTable =  moto_predictPlateNumberODAPI(detections,plate_class_id,category_index_temp, CLASS_DIC, LABEL_FILE_CLASS,twoLinePlate=twoLinePlate)
  
    plateTable = list(plateTable)
    for i in range(0,len(plateTable)) :
        plateTable[i] = list(plateTable[i])
        plateTable[i][-1] = LABEL_FILE_CLASS[int(plateTable[i][-1])]
    
    if show_image :
        plt.imshow(image_np_with_detections)
        plt.show()
        
    return plate_str, plateTable, category_index_temp, CLASS_DIC, platetype_index
            





