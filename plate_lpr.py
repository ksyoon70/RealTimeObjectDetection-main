# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 19:50:19 2022

@author: headway
이 파일은 차량과 번호판 인식 시작을 하는 파일이다. 
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
from plate_recog_common import *
from plate_char_infer import *
from plate_hr_infer import *
from plate_vr_infer import *
from plate_or_infer import *
from label_tools import *
from plate_det import *
#로그에서 warining을 삭제할때 아래 코드를 사용한다.
import logging
logging.getLogger('tensorflow').disabled = True

#========================
# 여기의 내용을 용도에 맞게 수정한다.
dataset_category='car-plate'  #plate
test_dir_name = 'test'
show_image = True
save_image = True
save_true_recog_image = False          #정인식 영상 저장 여부
THRESH_HOLD = 0.1
IS_RESULT_DIR_REMOVE = True #결과 디렉토리 삭제 여부
MAKE_JSON_FILE  = True                 #json 파일 생성 여부
REMOVE_SRC_IMAGE = False                #원본영상 삭제여부
RESIZE_IMAGE_WIDTH = 640
RESIZE_IMAGE_HEIGHT = 640
DEFAULT_LABEL_FILE = 'LPR_Car-Plate_Labels.txt'
#========================
CLASS_DIC = {}
LABEL_FILE_CLASS = []
LABEL_FILE_HUMAN_NAMES = []
if dataset_category == 'car-plate':
    Labels = pd.read_csv(DEFAULT_LABEL_FILE, header = None )
    LABEL_FILE_CLASS = Labels[0].values.tolist()
    LABEL_FILE_HUMAN_NAMES = Labels[1].values.tolist()
    CLASS_DIC = dict(zip(LABEL_FILE_CLASS, LABEL_FILE_HUMAN_NAMES))

WORKSPACE_PATH = os.path.join(ROOT_DIR,'Tensorflow','workspace')
ANNOTATION_PATH = os.path.join(WORKSPACE_PATH,'annotations')
IMAGE_PATH =  os.path.join(WORKSPACE_PATH,'images',dataset_category)
MODEL_PATH = os.path.join(WORKSPACE_PATH , 'models',dataset_category)
CONFIG_PATH = os.path.join( MODEL_PATH ,'my_ssd_mobnet','pipeline.config')
CHECKPOINT_PATH = os.path.join( MODEL_PATH , 'my_ssd_mobnet')


#테스트할 이미지 디렉토리
images_dir = os.path.join(IMAGE_PATH,test_dir_name)
result_dir = os.path.join(IMAGE_PATH,'result')
no_recog_dir = os.path.join(result_dir,'no_recog')
wrong_recog_dir = os.path.join(result_dir,'wrong_recog') #오인식
json_dir = os.path.join(IMAGE_PATH,'json')      #json 파일 생성 디렉토리

#result 디렉토리 삭제여부
if not os.path.isdir(result_dir):
	os.mkdir(result_dir)
    
if IS_RESULT_DIR_REMOVE :
    shutil.rmtree(result_dir)

if not os.path.isdir(result_dir):
	os.mkdir(result_dir)
    
#미인식 이면 미인식 폴더에 넣는다.
if not os.path.isdir(no_recog_dir):
	os.mkdir(no_recog_dir)
    
#오인식 이면 오인식 폴더에 넣는다.
if not os.path.isdir(wrong_recog_dir):
	os.mkdir(wrong_recog_dir)
# json 디렉토리가 있으면 삭제한다. 
if not os.path.isdir(json_dir):
    	os.mkdir(json_dir)

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
plate_det_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=plate_det_model)
#ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-101')).expect_partial()
#restore latest checkpoint
ckpt.restore(tf.train.latest_checkpoint(CHECKPOINT_PATH))
category_index = label_map_util.create_category_index_from_labelmap(os.path.join(ANNOTATION_PATH, 'platelabel_map.pbtxt'))

#번호판 찾기 인식용 모델 load
plate_det_init_fn()

#문자 번호 인식용 모델 load
ndet_model, ncat_index = number_det_init_fn()      # ncat_index 글자 추출 카테고리 인덱스
#ncat_index
#{1: {'id': 1, 'name': 'n1'}, 2: {'id': 2, 'name': 'n2'}, 3: {'id': 3, 'name': 'n3'}, 4: {'id': 4, 'name': 'n4'}, 5: {'id': 5, 'name': 'n5'}, 6: {'id': 6, 'name': 'n6'}, 7: {'id': 7, 'name': 'n7'}, 8: {'id': 8, 'name': 'n8'}, 9: {'id': 9, 'name': 'n9'}, 10: {'id': 10, 'name': 'n0'}, 11: {'id': 11, 'name': 'Char'}, 12: {'id': 12, 'name': 'vReg'}, 13: {'id': 13, 'name': 'hReg'}, 14: {'id': 14, 'name': 'oReg'}}

#문자 모델 초기화
char_model =  char_det_init_fn()

#hr 문자모델 초기화
hr_model = hr_det_init_fn()

#vr 문자모델 초기화
vr_model = vr_det_init_fn()

#or 문자모델 초기화
or_model = or_det_init_fn()

    
image_ext = ['jpg','JPG','png','PNG']
files = [fn for fn in os.listdir(images_dir)
                  if any(fn.endswith(ext) for ext in image_ext)]

total_test_files = len(files)
cnt = 0;

models = [ndet_model, char_model, hr_model, vr_model, or_model]


print('테스트용 이미지 갯수:',total_test_files)

recog_count = 0
fail_count = 0
false_recog_count = 0  #오인식 카운트
true_recog_count = 0
start_time = time.time() # strat time



try:

    for filename in files:
        cnt += 1
        image_path = os.path.join(images_dir,filename)
        result_path = os.path.join(result_dir,filename)
        basefilename, ext = os.path.splitext(filename)
        right_recog = False
        print('Processing {}'.format(filename))
        if os.path.exists(image_path) :
            imgRGB  = imread(image_path)
            #imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_np = np.array(imgRGB)
            src_height, src_width, scr_ch = image_np.shape

            src_box = [0,0,1,1]
            #pad 가 True이면 영상 아래 위로 black pad가 들어감.
            InsertPad = False
            det_image_np = extract_sub_image(image_np,src_box,RESIZE_IMAGE_WIDTH,RESIZE_IMAGE_WIDTH,pad=InsertPad)
            #plt.imshow(det_image_np)
            #plt.show()
            input_tensor = tf.convert_to_tensor(np.expand_dims(det_image_np, 0), dtype=tf.float32)
            detections = detect_fn(input_tensor, plate_det_model)
            
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
            
            label_id_offset = 1
            image_np_with_detections = image_np.copy()
            
            #현재 모델이 
            obj_class = []
            obj_score = []
            obj_boxes = []
            obj_labels = []
            # 카테고리가 'car-plate'이면 이륜차 검지를 위한 별도의 처리를 한다.
            if dataset_category == 'car-plate' :
                obj_class = detections['detection_classes']
                obj_boxes = detections['detection_boxes']
                obj_score = detections['detection_scores']
                #번호판만 따로 모은다.
                plate_info = []
                for index in range(num_detections) :
                    if LABEL_FILE_CLASS[obj_class[index]] == 'plate':
                        pbox = detections['detection_boxes'][index]
                        plate_info.append([pbox,False])
                
                for index in range(num_detections) :
                    if LABEL_FILE_CLASS[obj_class[index]] == 'car' or LABEL_FILE_CLASS[obj_class[index]] == 'truck' or LABEL_FILE_CLASS[obj_class[index]] == 'bus' or LABEL_FILE_CLASS[obj_class[index]] == 'bike' :
                        
                        if detections['detection_scores'][index] > THRESH_HOLD :
                            # 그 영상만 오려낸다.
                            box = detections['detection_boxes'][index]
                            height, width, ch = image_np.shape
                            # x 좌표는 그대로 쓴다.
                            box_sx= int(width*box[1])
                            box_ex= int(width*box[3])
                            # y 좌표는 그대로 쓴다.
                            box_sy= int(height*box[0])
                            box_ey= int(height*box[2])
                            obj_np = image_np[box_sy:box_ey,box_sx:box_ex,:]
                            obj_box = [[box_sx, box_ex, box_ex, box_sx],[box_sy,box_sy,box_ey,box_ey]]
                            plt.imshow(obj_np)
                            plt.show()
                            obj_img = cv2.cvtColor(obj_np, cv2.COLOR_BGR2RGB)
                            category = LABEL_FILE_CLASS[obj_class[index]]
                            
                            for index, [ pbox, IsUse ] in enumerate(plate_info) :
                                if not IsUse and isInside( pbox, box ) :  #박스를 사용하지 않았고, 차안에 번호판이 있으면...
                                    pbox_sx= int(width*pbox[1])
                                    pbox_ex= int(width*pbox[3])
                                    # y 좌표는 그대로 쓴다.
                                    pbox_sy= int(height*pbox[0])
                                    pbox_ey= int(height*pbox[2])
                                    pobj_np = image_np[pbox_sy:pbox_ey,pbox_sx:pbox_ex,:]
                                    pobj_box = [[pbox_sx, pbox_ex, pbox_ex, pbox_sx],[pbox_sy,pbox_sy,pbox_ey,pbox_ey]]
                                    plate_info[index][1] = True  #이 아이템을 사용했음을 표시함.
                                    #plt.imshow(pobj_np)
                                    #plt.show()
                                    plate_str, plateTable,category_index_temp, CLASS_DIC,class_index = plateDetection(models, ncat_index, obj_np, category, filename, plate_np = pobj_np)
                                    #하나라도 차량에 속하는 번호판을 찾으면 빠져나간다.
                                    break
                            
                        else:
                            print('{} : {} is below Threshold{} < {}!'.format(filename,LABEL_FILE_CLASS[obj_class[index]],detections['detection_scores'][index],THRESH_HOLD))
                            continue
                        
                for index, [ pbox, IsUse ] in enumerate(plate_info) :
                    if IsUse == False :
                        #아직 차량과 매칭이 안되고 남은 번호판이 있으면...
                        pbox_sx= int(width*pbox[1])
                        pbox_ex= int(width*pbox[3])
                        # y 좌표는 그대로 쓴다.
                        pbox_sy= int(height*pbox[0])
                        pbox_ey= int(height*pbox[2])
                        pobj_np = image_np[pbox_sy:pbox_ey,pbox_sx:pbox_ex,:]
                        pobj_box = [[pbox_sx, pbox_ex, pbox_ex, pbox_sx],[pbox_sy,pbox_sy,pbox_ey,pbox_ey]]
                        plate_info[index][1] = True  #이 아이템을 사용했음을 표시함.
                        #plt.imshow(pobj_np)
                        #plt.show()
                        plate_str, plateTable,category_index_temp, CLASS_DIC,class_index = plateDetection(models, ncat_index, None, 'plate', filename, plate_np = pobj_np)
                        
                # 1개 파일에 대한 모든 조회가 끝남.
     
    
except Exception as e:
             pass

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 19:50:19 2022

@author: headway
이 파일은 차량과 번호판 인식 시작을 하는 파일이다. 
번호판도 인식하고, 해당 json 파일을 만든다.
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
from plate_recog_common import *
from plate_char_infer import *
from plate_hr_infer import *
from plate_vr_infer import *
from plate_or_infer import *
from label_tools import *
from plate_det import *
from JsonMng import *
#로그에서 warining을 삭제할때 아래 코드를 사용한다.
import logging
logging.getLogger('tensorflow').disabled = True

#========================
# 여기의 내용을 용도에 맞게 수정한다.
dataset_category='car-plate'  #plate
test_dir_name = 'test'
show_image = True
save_image = True
save_true_recog_image = False          #정인식 영상 저장 여부
THRESH_HOLD = 0.1
IS_RESULT_DIR_REMOVE = True #결과 디렉토리 삭제 여부
MAKE_JSON_FILE  = True                 #json 파일 생성 여부
REMOVE_SRC_IMAGE = False                #원본영상 삭제여부
RESIZE_IMAGE_WIDTH = 640
RESIZE_IMAGE_HEIGHT = 640
DEFAULT_LABEL_FILE = 'LPR_Car-Plate_Labels.txt'
#========================
CLASS_DIC = {}
LABEL_FILE_CLASS = []
LABEL_FILE_HUMAN_NAMES = []
if dataset_category == 'car-plate':
    Labels = pd.read_csv(DEFAULT_LABEL_FILE, header = None )
    LABEL_FILE_CLASS = Labels[0].values.tolist()
    LABEL_FILE_HUMAN_NAMES = Labels[1].values.tolist()
    CLASS_DIC = dict(zip(LABEL_FILE_CLASS, LABEL_FILE_HUMAN_NAMES))

WORKSPACE_PATH = os.path.join(ROOT_DIR,'Tensorflow','workspace')
ANNOTATION_PATH = os.path.join(WORKSPACE_PATH,'annotations')
IMAGE_PATH =  os.path.join(WORKSPACE_PATH,'images',dataset_category)
MODEL_PATH = os.path.join(WORKSPACE_PATH , 'models',dataset_category)
CONFIG_PATH = os.path.join( MODEL_PATH ,'my_ssd_mobnet','pipeline.config')
CHECKPOINT_PATH = os.path.join( MODEL_PATH , 'my_ssd_mobnet')


#테스트할 이미지 디렉토리
images_dir = os.path.join(IMAGE_PATH,test_dir_name)
result_dir = os.path.join(IMAGE_PATH,'result')
no_recog_dir = os.path.join(result_dir,'no_recog')
wrong_recog_dir = os.path.join(result_dir,'wrong_recog') #오인식
json_dir = os.path.join(IMAGE_PATH,'json')      #json 파일 생성 디렉토리

#result 디렉토리 삭제여부
if not os.path.isdir(result_dir):
	os.mkdir(result_dir)
    
if IS_RESULT_DIR_REMOVE :
    shutil.rmtree(result_dir)

if not os.path.isdir(result_dir):
	os.mkdir(result_dir)
    
#미인식 이면 미인식 폴더에 넣는다.
if not os.path.isdir(no_recog_dir):
	os.mkdir(no_recog_dir)
    
#오인식 이면 오인식 폴더에 넣는다.
if not os.path.isdir(wrong_recog_dir):
	os.mkdir(wrong_recog_dir)
# json 디렉토리가 있으면 삭제한다. 
if not os.path.isdir(json_dir):
    	os.mkdir(json_dir)

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
plate_det_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=plate_det_model)
#ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-101')).expect_partial()
#restore latest checkpoint
ckpt.restore(tf.train.latest_checkpoint(CHECKPOINT_PATH))
category_index = label_map_util.create_category_index_from_labelmap(os.path.join(ANNOTATION_PATH, 'platelabel_map.pbtxt'))

#번호판 찾기 인식용 모델 load
plate_det_init_fn()

#문자 번호 인식용 모델 load
ndet_model, ncat_index = number_det_init_fn()      # ncat_index 글자 추출 카테고리 인덱스
#ncat_index
#{1: {'id': 1, 'name': 'n1'}, 2: {'id': 2, 'name': 'n2'}, 3: {'id': 3, 'name': 'n3'}, 4: {'id': 4, 'name': 'n4'}, 5: {'id': 5, 'name': 'n5'}, 6: {'id': 6, 'name': 'n6'}, 7: {'id': 7, 'name': 'n7'}, 8: {'id': 8, 'name': 'n8'}, 9: {'id': 9, 'name': 'n9'}, 10: {'id': 10, 'name': 'n0'}, 11: {'id': 11, 'name': 'Char'}, 12: {'id': 12, 'name': 'vReg'}, 13: {'id': 13, 'name': 'hReg'}, 14: {'id': 14, 'name': 'oReg'}}

#문자 모델 초기화
char_model =  char_det_init_fn()

#hr 문자모델 초기화
hr_model = hr_det_init_fn()

#vr 문자모델 초기화
vr_model = vr_det_init_fn()

#or 문자모델 초기화
or_model = or_det_init_fn()

    
image_ext = ['jpg','JPG','png','PNG']
files = [fn for fn in os.listdir(images_dir)
                  if any(fn.endswith(ext) for ext in image_ext)]

total_test_files = len(files)
cnt = 0;

models = [ndet_model, char_model, hr_model, vr_model, or_model]


print('테스트용 이미지 갯수:',total_test_files)

recog_count = 0
fail_count = 0
false_recog_count = 0  #오인식 카운트
true_recog_count = 0
start_time = time.time() # strat time



try:

    for filename in files:
        cnt += 1
        image_path = os.path.join(images_dir,filename)
        result_path = os.path.join(result_dir,filename)
        basefilename, ext = os.path.splitext(filename)
        right_recog = False
        print('Processing {}'.format(filename))
        if os.path.exists(image_path) :
            
            
            imgRGB  = imread(image_path)
            #imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_np = np.array(imgRGB)
            src_height, src_width, scr_ch = image_np.shape
            jsnonMng = JsonMng(json_dir,image_np.shape,filename)

            src_box = [0,0,1,1]
            #pad 가 True이면 영상 아래 위로 black pad가 들어감.
            InsertPad = False
            det_image_np = extract_sub_image(image_np,src_box,RESIZE_IMAGE_HEIGHT,RESIZE_IMAGE_WIDTH,pad=InsertPad)
            #plt.imshow(det_image_np)
            #plt.show()
            input_tensor = tf.convert_to_tensor(np.expand_dims(det_image_np, 0), dtype=tf.float32)
            detections = detect_fn(input_tensor, plate_det_model)
            
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
            
            label_id_offset = 1
            image_np_with_detections = image_np.copy()
            
            #현재 모델이 
            obj_class = []
            obj_score = []
            obj_boxes = []
            obj_labels = []
            # 카테고리가 'car-plate'이면 이륜차 검지를 위한 별도의 처리를 한다.
            if dataset_category == 'car-plate' :
                obj_class = detections['detection_classes']
                obj_boxes = detections['detection_boxes']
                obj_score = detections['detection_scores']
                #번호판만 따로 모은다.
                plate_info = []
                for index in range(num_detections) :
                    if LABEL_FILE_CLASS[obj_class[index]] == 'plate':
                        pbox = detections['detection_boxes'][index]
                        plate_info.append([pbox,False])
                
                for index in range(num_detections) :
                    if LABEL_FILE_CLASS[obj_class[index]] == 'car' or LABEL_FILE_CLASS[obj_class[index]] == 'truck' or LABEL_FILE_CLASS[obj_class[index]] == 'bus' or LABEL_FILE_CLASS[obj_class[index]] == 'bike' :
                        
                        if detections['detection_scores'][index] > THRESH_HOLD :
                            # 그 영상만 오려낸다.
                            box = detections['detection_boxes'][index]
                            height, width, ch = image_np.shape
                            # x 좌표는 그대로 쓴다.
                            box_sx= int(width*box[1])
                            box_ex= int(width*box[3])
                            # y 좌표는 그대로 쓴다.
                            box_sy= int(height*box[0])
                            box_ey= int(height*box[2])
                            obj_np = image_np[box_sy:box_ey,box_sx:box_ex,:]
                            obj_box = [[box_sx, box_ex, box_ex, box_sx],[box_sy,box_sy,box_ey,box_ey]]
                            #plt.imshow(obj_np)
                            #plt.show()
                            obj_img = cv2.cvtColor(obj_np, cv2.COLOR_BGR2RGB)
                            category = LABEL_FILE_CLASS[obj_class[index]]
                            jsnonMng.addObject(box=obj_box, label = category)
                            for index, [ pbox, IsUse ] in enumerate(plate_info) :
                                if not IsUse and isInside( pbox, box ) :  #박스를 사용하지 않았고, 차안에 번호판이 있으면...
                                    pbox_sx= int(width*pbox[1])
                                    pbox_ex= int(width*pbox[3])
                                    # y 좌표는 그대로 쓴다.
                                    pbox_sy= int(height*pbox[0])
                                    pbox_ey= int(height*pbox[2])
                                    pobj_np = image_np[pbox_sy:pbox_ey,pbox_sx:pbox_ex,:]
                                    pobj_box = [[pbox_sx, pbox_ex, pbox_ex, pbox_sx],[pbox_sy,pbox_sy,pbox_ey,pbox_ey]]
                                    plate_info[index][1] = True  #이 아이템을 사용했음을 표시함.
                                    #plt.imshow(pobj_np)
                                    #plt.show()
                                    ratios = [float(src_width)/pobj_np.shape[1],float(src_height)/pobj_np.shape[0]]
                                    plate_str, plateTable,category_index_temp, CLASS_DIC,class_index = plateDetection(models, ncat_index, obj_np, category, filename, plate_np = pobj_np)
                                    #하나라도 차량에 속하는 번호판을 찾으면 빠져나간다.
                                    jsnonMng.addPlate(plateTable=plateTable,category_index=category_index_temp,CLASS_DIC=CLASS_DIC,platebox=pobj_box,plateIndex=class_index,plate_shape=pobj_np.shape)
                                    break
                            
                        else:
                            print('{} : {} is below Threshold{} < {}!'.format(filename,LABEL_FILE_CLASS[obj_class[index]],detections['detection_scores'][index],THRESH_HOLD))
                            continue
                        
                for index, [ pbox, IsUse ] in enumerate(plate_info) :
                    if IsUse == False :
                        #아직 차량과 매칭이 안되고 남은 번호판이 있으면...
                        pbox_sx= int(width*pbox[1])
                        pbox_ex= int(width*pbox[3])
                        # y 좌표는 그대로 쓴다.
                        pbox_sy= int(height*pbox[0])
                        pbox_ey= int(height*pbox[2])
                        pobj_np = image_np[pbox_sy:pbox_ey,pbox_sx:pbox_ex,:]
                        pobj_box = [[pbox_sx, pbox_ex, pbox_ex, pbox_sx],[pbox_sy,pbox_sy,pbox_ey,pbox_ey]]
                        plate_info[index][1] = True  #이 아이템을 사용했음을 표시함.
                        #plt.imshow(pobj_np)
                        #plt.show()
                        plate_str, plateTable,category_index_temp, CLASS_DIC,class_index = plateDetection(models, ncat_index, None, 'plate', filename, plate_np = pobj_np)
                        
                # 1개 파일에 대한 모든 조회가 끝남.
                jsnonMng.save()
     
    
except Exception as e:
             pass

