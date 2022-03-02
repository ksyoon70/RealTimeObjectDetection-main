# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 19:33:30 2022

@author: headway
"""
import os,sys
import cv2
import time
import uuid
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import tensorflow.compat.v1 as tfv1

ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR) 

ROOT_OF_ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

WORKSPACE_PATH = os.path.join(ROOT_DIR,'Tensorflow','workspace')
SCRIPTS_PATH = os.path.join(ROOT_DIR,'Tensorflow','scripts')
APIMODEL_PATH = os.path.join(ROOT_OF_ROOT,'models') 
ANNOTATION_PATH = os.path.join(WORKSPACE_PATH,'annotations')
IMAGE_PATH =  os.path.join(WORKSPACE_PATH,'images')
MODEL_PATH = os.path.join(WORKSPACE_PATH , 'models')
PRETRAINED_MODEL_PATH = os.path.join(WORKSPACE_PATH , 'pre-trained-models')
CONFIG_PATH = os.path.join( MODEL_PATH ,'my_ssd_mobnet','pipeline.config')
CHECKPOINT_PATH = os.path.join( MODEL_PATH , 'my_ssd_mobnet')


CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 



CLASS_NAMES = ['bike_front','bike_rear','bus_front','bus_rear','car_front','car_rear','truck_front','truck_rear']
#CLASS_NAMES = ['cat','dog']

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
config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)


pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(CONFIG_PATH, "r") as f:                                                                                                                                                                                                                     
    proto_str = f.read()                                                                                                                                                                                                                                          
    text_format.Merge(proto_str, pipeline_config) 
    
    
pipeline_config.model.ssd.num_classes = len(CLASS_NAMES)
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(PRETRAINED_MODEL_PATH,'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8','checkpoint','ckpt-0')
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