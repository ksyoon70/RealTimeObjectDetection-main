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
from moto_plate_char_infer import *
from moto_plate_hr_infer import *
#로그에서 warining을 삭제할때 아래 코드를 사용한다.
import logging
logging.getLogger('tensorflow').disabled = True

#========================
# 여기의 내용을 용도에 맞게 수정한다.
dataset_category='car-plate'  #plate
test_dir_name = 'test'
show_image = True
save_image = False
save_true_recog_image = False          #정인식 영상 저장 여부
add_platename = True               #번호인식 결과를 파일이름에 붙일지 여부
THRESH_HOLD = 0.5
PLATE_THRESH_HOLD = 0.4             
IS_RESULT_DIR_REMOVE = True #결과 디렉토리 삭제 여부
MAKE_JSON_FILE  = True                 #json 파일 생성 여부
MOVE_SRC_IMAGE = True                #원본 영상 이동 여부
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
obj_det_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=obj_det_model)
#ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-101')).expect_partial()
#restore latest checkpoint
ckpt.restore(tf.train.latest_checkpoint(CHECKPOINT_PATH))
category_index = label_map_util.create_category_index_from_labelmap(os.path.join(ANNOTATION_PATH, 'platelabel_map.pbtxt'))

#번호판 찾기 인식용 모델 load
plate_det_init_fn()

#문자 번호 인식용 모델 load
ndet_model, ncat_index = number_det_init_fn()       # ncat_index 글자 추출 카테고리 인덱스
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

#이륜차 모델 초기화
moto_ndet_model, moto_ncat_index = moto_number_det_init_fn()      # ncat_index 글자 추출 카테고리 인덱스
moto_char_model = moto_char_det_init_fn()
moto_hr_model = moto_hr_det_init_fn()

    
image_ext = ['jpg','JPG','jpeg','JPEG','png','PNG']
files = [fn for fn in os.listdir(images_dir)
                  if any(fn.endswith(ext) for ext in image_ext)]

total_test_files = len(files)
cnt = 0;

models = [ndet_model, char_model, hr_model, vr_model, or_model, moto_ndet_model,moto_char_model, moto_hr_model]


print('테스트용 이미지 갯수:',total_test_files)

recog_count = 0
fail_count = 0
false_recog_count = 0  #오인식 카운트
true_recog_count = 0
start_time = time.time() # strat time

try:
    start_time = time.time() # strat time
    
    for filename in files:
        cnt += 1
        image_path = os.path.join(images_dir,filename)
        result_path = os.path.join(result_dir,filename)
        json_image_path  = os.path.join(json_dir,filename)
        wrong_recog_path = os.path.join(wrong_recog_dir,filename)
        no_recog_path = os.path.join(no_recog_dir,filename)
        basefilename, ext = os.path.splitext(filename)
        right_recog = False
        recog_list = []
        recog_index = -1     #recog_list에서 인식한 번호 인덱스
        plate_str = None
        false_recog = False #오인식 실패 여부
        fail_recog = False  #인식 실패 여부
        print('Processing {}'.format(filename))
        if os.path.exists(image_path) :
            
            imgBRB  = imread(image_path)
            #imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_np = np.array(imgBRB)
            height, width, ch = image_np.shape
            jsonMng = JsonMng(json_dir,image_np.shape,filename)

            src_box = [0,0,1,1]
            #pad 가 True이면 영상 아래 위로 black pad가 들어감.
            InsertPad = False
            det_image_np = extract_sub_image(image_np,src_box,RESIZE_IMAGE_HEIGHT,RESIZE_IMAGE_WIDTH,pad=InsertPad)
            #plt.imshow(det_image_np)
            #plt.show()
            input_tensor = tf.convert_to_tensor(np.expand_dims(det_image_np, 0), dtype=tf.float32)
            detections = detect_fn(input_tensor, obj_det_model)
            
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
            
            label_id_offset = 1
            image_np_with_detections = image_np.copy()
            if show_image:              
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
                plt.imshow(image_np_with_detections)
                plt.show()
            
            
            #현재 모델이 
            obj_class = []
            obj_score = []
            obj_boxes = []
            obj_labels = []
            helmet_box_list = []
            plate_box_list = []
            # 카테고리가 'car-plate'이면 이륜차 검지를 위한 별도의 처리를 한다.
            if dataset_category == 'car-plate' :
                obj_class = detections['detection_classes']
                obj_boxes = detections['detection_boxes']
                obj_score = detections['detection_scores']
                #번호판만 따로 모은다.
                plate_info = [] 
                vehi_box_list = []
                #추후 번호판이나 차량등이 겹치면 삭제해 주어야 한다.
                for index in range(num_detections) :
                    if detections['detection_scores'][index] > THRESH_HOLD :
                        label = LABEL_FILE_CLASS[obj_class[index]]
                        #if LABEL_FILE_CLASS[obj_class[index]] == 'plate':
                        if 'type' in LABEL_FILE_CLASS[obj_class[index]]:  # type1 ~ type13
                            pbox = detections['detection_boxes'][index]
                            
                            box_sx= int(width*pbox[1])
                            box_ex= int(width*pbox[3])
                            # y 좌표는 그대로 쓴다.
                            box_sy= int(height*pbox[0])
                            box_ey= int(height*pbox[2])
                            #이전 박스와 겹치는지 확인한다.
                            checkOberlapped = overlabCheck([box_sy,box_sx,box_ey,box_ex], plate_box_list)
                            
                            if not checkOberlapped:
                                plate_box_list.append([box_sy,box_sx,box_ey,box_ex])
                                plate_info.append([pbox,False,label])
                            
                        elif LABEL_FILE_CLASS[obj_class[index]] == 'helmet':
                            category = LABEL_FILE_CLASS[obj_class[index]]
                            helmet_box_ratio = detections['detection_boxes'][index]
                            box_sx= int(width*helmet_box_ratio[1])
                            box_ex= int(width*helmet_box_ratio[3])
                            # y 좌표는 그대로 쓴다.
                            box_sy= int(height*helmet_box_ratio[0])
                            box_ey= int(height*helmet_box_ratio[2])
                            helmet_box = [[box_sx, box_ex, box_ex, box_sx],[box_sy,box_sy,box_ey,box_ey]]
                            helmet_box_list.append(helmet_box)  # helmet 리스트에 추가한다.
                            jsonMng.addObject(box=helmet_box, label = category)
                        elif LABEL_FILE_CLASS[obj_class[index]] == 'bicycle':
                            category = LABEL_FILE_CLASS[obj_class[index]]
                            bicycle_box_ratio = detections['detection_boxes'][index]
                            box_sx= int(width*bicycle_box_ratio[1])
                            box_ex= int(width*bicycle_box_ratio[3])
                            # y 좌표는 그대로 쓴다.
                            box_sy= int(height*bicycle_box_ratio[0])
                            box_ey= int(height*bicycle_box_ratio[2])
                            bicycle_box = [[box_sx, box_ex, box_ex, box_sx],[box_sy,box_sy,box_ey,box_ey]]
                            vehi_box_list.append([box_sy,box_sx,box_ey,box_ex])
                            jsonMng.addObject(box=bicycle_box, label = category)

                
                for index in range(num_detections) :
                    if LABEL_FILE_CLASS[obj_class[index]] == 'car' or LABEL_FILE_CLASS[obj_class[index]] == 'truck' or LABEL_FILE_CLASS[obj_class[index]] == 'bus' or LABEL_FILE_CLASS[obj_class[index]] == 'motorcycle' :
                        
                        if detections['detection_scores'][index] > THRESH_HOLD :
                            # 그 영상만 오려낸다.
                            box = detections['detection_boxes'][index]  # box 0 ~ 1
                            
                            # x 좌표는 그대로 쓴다.
                            box_sx= int(width*box[1])
                            box_ex= int(width*box[3])
                            # y 좌표는 그대로 쓴다.
                            box_sy= int(height*box[0])
                            box_ey= int(height*box[2])
                            obj_np = image_np[box_sy:box_ey,box_sx:box_ex,:]  #차량영상
                            obj_box = [[box_sx, box_ex, box_ex, box_sx],[box_sy,box_sy,box_ey,box_ey]]
                            #이전 박스와 겹치는지 확인한다.
                            checkOberlapped = overlabCheck([box_sy,box_sx,box_ey,box_ex], vehi_box_list)
                                    
                            if checkOberlapped:
                                continue;
                            
                            vehi_box_list.append([box_sy,box_sx,box_ey,box_ex])
                            #if show_image:
                            #    plt.imshow(obj_np)
                            #    plt.show()
                            obj_img = obj_np
                            #obj_img = cv2.cvtColor(obj_np, cv2.COLOR_BGR2RGB)
                            category = LABEL_FILE_CLASS[obj_class[index]]
                            jsonMng.addObject(box=obj_box, label = category)
                            findPlate = False #번호판을 찾았음을 나타내는 플래그
                            for index, [ pbox, IsUse, label ] in enumerate(plate_info) :
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
                                    ratios = [float(width)/pobj_np.shape[1],float(height)/pobj_np.shape[0]]
                                    plate_str, plateTable,category_index_temp, CLASS_DIC,class_index,_ = plateDetection(models, ncat_index, obj_np, category, filename, plate_np = pobj_np, plate_label = label)
                                    
                                    if plate_str:
                                        recog_list.append(plate_str)
                                    
                                    #하나라도 차량에 속하는 번호판을 찾으면 빠져나간다.
                                    if plateTable is not None :
                                        jsonMng.addPlate(plateTable=plateTable,category_index=category_index_temp,CLASS_DIC=CLASS_DIC,platebox=pobj_box,plateIndex=class_index,plate_shape=pobj_np.shape)
                                        findPlate = True
                                    break
                                
                            if not findPlate :
                                #좀 더 범위를 좁혀서 찾아본다.
                                plate_str, plateTable,category_index_temp, CLASS_DIC,class_index,pobj_box_pt = plateDetection(models, ncat_index, obj_np, category, filename, plate_np = None, plate_label = None)
                                
                                if plate_str:
                                    recog_list.append(plate_str)
                                
                                if plateTable is not None :
                                    pobj_box_pt = coordinationTrans(box_sx,box_sy,pobj_box_pt) #pobj_box는 픽셀 좌표이다.
                                    pbox_sx= int(pobj_box_pt[0][0])
                                    pbox_ex= int(pobj_box_pt[0][1])
                                    # y 좌표는 그대로 쓴다.
                                    pbox_sy= int(pobj_box_pt[1][0])
                                    pbox_ey= int(pobj_box_pt[1][2])
                                    pobj_np = image_np[pbox_sy:pbox_ey,pbox_sx:pbox_ex,:]
                                    jsonMng.addPlate(plateTable=plateTable,category_index=category_index_temp,CLASS_DIC=CLASS_DIC,platebox=pobj_box_pt,plateIndex=class_index,plate_shape=pobj_np.shape)
                        else:
                            print('{} : {} is below Threshold {:.3f} < {:.3f} !'.format(filename,LABEL_FILE_CLASS[obj_class[index]],detections['detection_scores'][index],THRESH_HOLD))
                            continue
                        
                for index, [ pbox, IsUse, label ] in enumerate(plate_info) :
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
                        #오토바이번호판인지 일반 번호판인지 알수 없으므로 일단 헬멧이 있으면 그것 아래에 있는지 확인하고 그러면 이륜차 아니면 일반차 번호판으로 인식을 시도 해본다.
                        IsMotorsycle = False
                        
                        if len(helmet_box_list) > 0 :
                            for hBox in helmet_box_list :
                               if isUnderBox(pobj_box , hBox) :
                                   IsMotorsycle = True
                                   break;
                               
                        if IsMotorsycle:
                            #오토바이가 확실하므로 type13을 넣는다.
                            plate_str, plateTable,category_index_temp, CLASS_DIC,class_index,_ = plateDetection(models, ncat_index, None, 'motorcycle', filename, plate_np = pobj_np, plate_label='type13')
                        else:
                            plate_str, plateTable,category_index_temp, CLASS_DIC,class_index,_ = plateDetection(models, ncat_index, None, 'plate', filename, plate_np = pobj_np, plate_label = label)
                        
                        if plate_str:
                            recog_list.append(plate_str)
                            
                        if plateTable is not None :
                            jsonMng.addPlate(plateTable=plateTable,category_index=category_index_temp,CLASS_DIC=CLASS_DIC,platebox=pobj_box,plateIndex=class_index,plate_shape=pobj_np.shape)
                
                if len(recog_list) > 0 :
                    #인식된 리스트를 돌며 인식한 문자가 파일 이름에 있는지 확인한다.
                    recog_count += 1
                    find_string_in_filename = False
                    ix = 0
                    for recog_string in recog_list :
                        if recog_string in basefilename:
                            find_string_in_filename = True
                            true_recog_count += 1
                            recog_index = ix
                            break
                        ix += 1
                    #파일에서 일치된 인식 문자를 찾지 못했을 경우.
                    if not find_string_in_filename:
                        false_recog_count += 1
                        false_recog = True
                       
                else :
                    fail_count += 1
                    fail_recog = True
                   
                                    
                # 1개 파일에 대한 모든 조회가 끝남.
                if add_platename and len(recog_list):
                    if recog_index != -1:
                        json_image_path = os.path.join(json_dir,basefilename + '_' + recog_list[recog_index] + ext)
                        jsonMng.save(add_platename,recog_list[recog_index])
                    else:
                        json_image_path = os.path.join(json_dir,basefilename + '_' + recog_list[0] + ext)
                        wrong_recog_path = os.path.join(wrong_recog_dir,basefilename + '_' + recog_list[0] + ext)
                        jsonMng.save(add_platename,recog_list[0])
                else:
                    jsonMng.save()
                
                saved_json_filename = jsonMng.getJsonFilename()
                

                #json 파일을 미인식 디렉토리로 복사한다.
                if(fail_recog) :
                    src_file = os.path.join(json_dir,saved_json_filename)
                    dst_file = os.path.join(no_recog_dir,saved_json_filename)
                    if os.path.isfile(src_file) :
                        shutil.copy(src_file,dst_file)
                    #이미지를 미인식 디렉토리로 복사한다.
                    shutil.copy(image_path,no_recog_path)
                #json 파일을 오인식 디렉토리로 복사한다.
                if(false_recog) :
                    src_file = os.path.join(json_dir,saved_json_filename)
                    dst_file = os.path.join(wrong_recog_dir,saved_json_filename)
                    if os.path.isfile(src_file) :
                        shutil.copy(src_file,dst_file)
                        #이미지 파일을 오인식 디렉토리로 복사한다
                    if os.path.isfile(image_path) :
                        shutil.copy(image_path,wrong_recog_path)
                        
                #이미지 파일을 json 디렉토리로 복사한다.
                if MOVE_SRC_IMAGE :
                    #원본 영상 이동 옵션이면...
                    shutil.move(image_path,json_image_path)
                else:
                    shutil.copy(image_path,json_image_path)
                
    
    end_time = time.time()
    print("총: {}장".format(total_test_files))        
    print("수행시간: {:.2f}".format(end_time - start_time))
    print("건당 수행시간 : {:.2f}".format((end_time - start_time)/total_test_files))             
    print('인식률: {:}'.format(recog_count) +'  ({:.2f})'.format(recog_count*100/total_test_files) + ' %')
    print('정인식: {:}'.format(true_recog_count) +'  ({:.2f})'.format(true_recog_count*100/recog_count) + ' %')
    print('오인식: {:}'.format(false_recog_count) +'  ({:.2f})'.format(false_recog_count*100/recog_count) + ' %')
    print('인식실패: {}'.format(fail_count) +'  ({:.2f})'.format(fail_count*100/total_test_files) + ' %') 
    print('정인식율: {}'.format(true_recog_count) +'  ({:.2f})'.format(true_recog_count*100/total_test_files) + ' %') 
 
    
except Exception as e:
             pass

    