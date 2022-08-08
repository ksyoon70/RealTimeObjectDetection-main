# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 19:33:30 2022

@author: headway
"""
import os,sys
import cv2
from shutil import copyfile
import time
import uuid
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import tensorflow.compat.v1 as tfv1
import pandas as pd

#-----------------------------------------------------------------------
# 사용자가 수정하는 부분이다.
#여기서 사용할 모델을 고른다.
#PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
#PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8'
PRETRAINED_MODEL_NAME = 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8'

CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 

fsLabelFileName = "./tracking_Label.txt"  #라벨 파일이름
filterFileName =  "filter.map"  #"LPR_Filtermap.txt"  #필터 맵 파일이다. 사용하지않으면 존재하지 않는 파일명을 넣는다.
dataset_category='tracking'
#-----------------------------------------------------------------------

ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR) 

ROOT_OF_ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

WORKSPACE_PATH = os.path.join(ROOT_DIR,'Tensorflow','workspace')
SCRIPTS_PATH = os.path.join(ROOT_DIR,'Tensorflow','scripts')
APIMODEL_PATH = os.path.join(ROOT_OF_ROOT,'models') 
ANNOTATION_PATH = os.path.join(WORKSPACE_PATH,'annotations')
IMAGE_PATH =  os.path.join(WORKSPACE_PATH,'images')
MODEL_PATH = os.path.join(WORKSPACE_PATH ,'models',dataset_category)
PRETRAINED_MODEL_PATH = os.path.join(WORKSPACE_PATH , 'pre-trained-models')
CONFIG_PATH = os.path.join( MODEL_PATH ,'my_ssd_mobnet','pipeline.config')
CHECKPOINT_PATH = os.path.join( MODEL_PATH , 'my_ssd_mobnet')



fLabels = pd.read_csv(fsLabelFileName, header = None )
CLASS_NAMES = fLabels[0].values.tolist()


#필터맵을 통하여 라별 변환할 항목을 읽는다.
FILTERMAP_FILE_PATH = os.path.join(ROOT_DIR,filterFileName)

bFilterMap = False
if  not os.path.exists(FILTERMAP_FILE_PATH):
    print("FilterMap File {0} isn't exists".format(FILTERMAP_FILE_PATH))
else :
    bFilterMap = True
    
if bFilterMap :
    ConvMap = {}
    cLabels = pd.read_csv(FILTERMAP_FILE_PATH, header = None )

    for i, value in enumerate(cLabels[0]):
        ConvMap[value] = cLabels[1][i]
    
     #필터에 있는 내용이면 레이블 이름을 변경한다.
    for index, label in enumerate(CLASS_NAMES) :
        
        if label  in ConvMap:
            CLASS_NAMES[index] = ConvMap[label]
    
    new_list = []
    for v in CLASS_NAMES:
        if v not in new_list:
            new_list.append(v)
    CLASS_NAMES = new_list

#라벨 클래스 이름에서 필터에 있는 내용이 있으면 변환한다.


#CLASS_NAMES = ['bike_front','bike_rear','bus_front','bus_rear','car_front','car_rear','truck_front','truck_rear']
#CLASS_NAMES = ['cat','dog']
#CLASS_NAMES = ['car','plate']


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


N = len(CLASS_NAMES)

#for i in range(N):
labels = [{'name':cls_name, 'id':i+1 } for i, cls_name in enumerate(CLASS_NAMES)]

with open(ANNOTATION_PATH + '\label_map.pbtxt', 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')


CONFIG_PATH = os.path.join(MODEL_PATH, CUSTOM_MODEL_NAME, 'pipeline.config')

if not os.path.isdir(os.path.join(MODEL_PATH, CUSTOM_MODEL_NAME)):
    createFolder(MODEL_PATH)
    createFolder(os.path.join(MODEL_PATH, CUSTOM_MODEL_NAME))
    
if not os.path.exists(CONFIG_PATH) :
    src_file = os.path.join(PRETRAINED_MODEL_PATH,PRETRAINED_MODEL_NAME,'pipeline.config')
    dst_file = CONFIG_PATH
    if os.path.exists(src_file):
        copyfile(src_file,dst_file)
    else:
        print("Error no {0} exists".format(src_file))
        sys.exit()

config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)


pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(CONFIG_PATH, "r") as f:                                                                                                                                                                                                                     
    proto_str = f.read()                                                                                                                                                                                                                                          
    text_format.Merge(proto_str, pipeline_config) 
    
    
pipeline_config.model.ssd.num_classes = len(CLASS_NAMES)
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(PRETRAINED_MODEL_PATH,PRETRAINED_MODEL_NAME,'checkpoint','ckpt-0')
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= os.path.join(ANNOTATION_PATH,'label_map.pbtxt')
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(ANNOTATION_PATH,'train.record')]
pipeline_config.eval_input_reader[0].label_map_path = os.path.join(ANNOTATION_PATH,'label_map.pbtxt')
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(ANNOTATION_PATH ,'valid.record')]



config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
with tf.io.gfile.GFile(CONFIG_PATH, "wb") as f:                                                                                                                                                                                                                     
    f.write(config_text)   

print("if you train your dataset, Copy & paste below message to command window")
print("""python {}/research/object_detection/model_main_tf2.py --model_dir={}/{} --pipeline_config_path={}/{}/pipeline.config --num_train_steps=5000""".format(APIMODEL_PATH, MODEL_PATH,CUSTOM_MODEL_NAME,MODEL_PATH,CUSTOM_MODEL_NAME))