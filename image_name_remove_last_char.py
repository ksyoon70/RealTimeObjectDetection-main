# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 18:46:43 2022

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


#========================
# 여기의 내용을 용도에 맞게 수정한다.
dataset_category='plate'
test_dir_name = 'test'
show_image = True
save_image = True
THRESH_HOLD = 0.1
IS_RESULT_DIR_REMOVE = True #결과 디렉토리 삭제 여부
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

#result 디렉토리 삭제여부
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


recog_count = 0
fail_count = 0
false_recog_count = 0  #오인식 카운트
true_recog_count = 0
start_time = time.time() # strat time

RESIZE_IMAGE_WIDTH = 320
RESIZE_IMAGE_HEIGHT = 320

for filename in os.listdir(images_dir):
    image_path = os.path.join(images_dir,filename)
    basefilename, ext = os.path.splitext(filename)
    basefilename = basefilename[0:-1]
    result_file = os.path.join(images_dir, basefilename + ext)
    shutil.move(image_path,result_file)