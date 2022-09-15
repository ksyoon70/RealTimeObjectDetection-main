# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 19:50:19 2022

@author: headway
이 파일은 인식 시작을 하는 파일이다. 
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

#로그에서 warining을 삭제할때 아래 코드를 사용한다.
import logging
logging.getLogger('tensorflow').disabled = True


#========================
# 여기의 내용을 용도에 맞게 수정한다.
dataset_category='plate'
test_dir_name = 'test'
show_image = True
save_image = True
save_true_recog_image = False          #정인식 영상 저장 여부
THRESH_HOLD = 0.1
IS_RESULT_DIR_REMOVE = True #결과 디렉토리 삭제 여부
MAKE_JSON_FILE  = True                 #json 파일 생성 여부
REMOVE_SRC_IMAGE = False                #원본영상 삭제여부
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

#hr 문자모델 초기화
hr_model = hr_det_init_fn()

#vr 문자모델 초기화
vr_model = vr_det_init_fn()

#or 문자모델 초기화
or_model = or_det_init_fn()

    
total_test_files = len(os.listdir(images_dir))
cnt = 0;

models = [ndet_model, char_model, hr_model, vr_model, or_model]


print('테스트용 이미지 갯수:',total_test_files)

recog_count = 0
fail_count = 0
false_recog_count = 0  #오인식 카운트
true_recog_count = 0
start_time = time.time() # strat time

RESIZE_IMAGE_WIDTH = 320
RESIZE_IMAGE_HEIGHT = 320

try:

    for filename in os.listdir(images_dir):
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
            det_image_np = extract_sub_image(image_np,src_box,RESIZE_IMAGE_WIDTH,RESIZE_IMAGE_WIDTH,fixratio=True)
            #plt.imshow(det_image_np)
            #plt.show()
            input_tensor = tf.convert_to_tensor(np.expand_dims(det_image_np, 0), dtype=tf.float32)
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
                #print("'클래스:{0} 번호판 타입 {1} 확률:{2:.3f}".format(class_index,category_index[class_index]['name'],detections['detection_scores'][0]))
                #print('box= {}'.format(detections['detection_boxes'][0]))
                box = list(range(0,4))
                box = detections['detection_boxes'][0]
                height, width, ch = image_np.shape
                # box_sy = int(height*box[0])
                # box_sx= int(width*box[1])
                # box_ey = int(height*box[2])
                # box_ex= int(width*box[3])
                
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
                    box_sy= int(width*box[0])
                    box_ey= int(width*box[2])
                    # y 좌표는 수정한다.
                    # 좌측 black 부분 
                    left_black = (src_height - src_width)/2.0
                    box_sx = int(box[1]*src_height - left_black)
                    box_ex = int(box[3]*src_height - left_black)
                
                plate_np = image_np[box_sy:box_ey,box_sx:box_ex,:]
                plate_box = [[box_sx, box_ex, box_ex, box_sx],[box_sy,box_sy,box_ey,box_ey]]
                #plt.imshow(plate_np)
                #plt.show()
                
                plate_img = cv2.cvtColor(plate_np, cv2.COLOR_BGR2RGB)                
                #번호판을 320x320 크기로 정규화 한다.
                desired_size = max(RESIZE_IMAGE_WIDTH,RESIZE_IMAGE_HEIGHT)
                old_size = [plate_img.shape[1],plate_img.shape[0]]
                ratio = float(desired_size)/max(old_size)
                new_size = tuple([int(x*ratio) for x in old_size])
                #원영상에서 ratio 만큼 곱하여 리싸이즈한 번호판 영상을 얻는다.
                cropped_img = cv2.resize(plate_img,new_size,interpolation=cv2.INTER_AREA)
                plate_new_img_np = np.zeros((desired_size, desired_size, 3), dtype = "uint8")
                h = new_size[1]
                w = new_size[0]
                yoff = round((desired_size-h)/2)
                xoff = round((desired_size-w)/2)
                #320x320영상에 번호판을 붙여 넣는다.
                plate_new_img_np[yoff:yoff+h, xoff:xoff+w , :] = cropped_img            
                #번호판에 대하여 문자 및 번호를 인식한다.
                plate_str, plateTable,category_index_temp, CLASS_DIC,class_index = plate_number_detect_fn(models,plate_new_img_np,ncat_index, class_index,result_path=result_path)

                plate_new_img_np = cv2.cvtColor(plate_new_img_np, cv2.COLOR_RGB2BGR)
            
                ix_gt = -1 # 정답 위치
            
                gtrue_label = filename.split('_')[ix_gt]
                
                if ix_gt == -1:
                    gtrue_label = gtrue_label[0:-4]  #왜냐하면 마지막 .jpg를 삭제해야하므로.
                                
                if gtrue_label[-1] == 'c':
                    gtrue_label = gtrue_label[0:-1]
                
                xfind = plate_str.find('x')
                
                yfind = gtrue_label.find('영')
                
                #영자를 삭제한다.
                if yfind >= 0 :
                    gtrue_label = gtrue_label.replace('영','')
                
                
                
                if xfind == -1 :
                    recog_count += 1
                    if(gtrue_label == plate_str) :
                        true_recog_count += 1
                        result_file = os.path.join(result_dir, basefilename + '_' + plate_str + ext)
                        right_recog = True  #정인식이 되었음.
                    else:
                        #오인식인 경우
                        false_recog_count += 1
                        result_file = os.path.join(wrong_recog_dir, basefilename + '_' + plate_str + ext)
                        shutil.copy(image_path, os.path.join(wrong_recog_dir,os.path.basename(image_path)))
                        
                else :
                    fail_count += 1
                    result_file = os.path.join(no_recog_dir, basefilename + '_' + plate_str + ext)
                
                if save_true_recog_image:  #정인식 영상 저장 이면..
                    imwrite( result_file, plate_new_img_np)
                else :
                    if not right_recog:    #정인식 영상이 아니면 저장한다.
                        imwrite( result_file, plate_new_img_np)
                        
                #json 파일을 저장한다. def makeJson(src_path, image_filename,dst_path, image_shape,category_index, CLASS_DIC,plateTable, plateNumber) 
                if MAKE_JSON_FILE :
                    makeJson(src_path=images_dir,image_filename=filename,dst_path=json_dir,image_shape=image_np.shape, 
                             category_index=category_index_temp,CLASS_DIC=CLASS_DIC,plateTable=plateTable,
                             plateNumber=plate_str,platebox = plate_box ,plateIndex = class_index,plate_shape = plate_new_img_np.shape, xratio=ratio, add_platenum=True)
                    
                    
            else :
                fail_count += 1
                result_file = os.path.join(no_recog_dir, basefilename + ext)
                shutil.copy(image_path,result_file)
                
            if REMOVE_SRC_IMAGE:
                os.remove(image_path)
            
            
                
                    
    end_time = time.time()        
    print("수행시간: {:.2f}".format(end_time - start_time))
    print("건당 수행시간 : {:.2f}".format((end_time - start_time)/total_test_files))             
    print('인식률: {:}'.format(recog_count) +'  ({:.2f})'.format(recog_count*100/total_test_files) + ' %')
    print('정인식: {:}'.format(true_recog_count) +'  ({:.2f})'.format(true_recog_count*100/recog_count) + ' %')
    print('오인식: {:}'.format(false_recog_count) +'  ({:.2f})'.format(false_recog_count*100/recog_count) + ' %')
    print('인식실패: {}'.format(fail_count) +'  ({:.2f})'.format(fail_count*100/total_test_files) + ' %')  
    
except Exception as e:
            pass            

        
        
            





