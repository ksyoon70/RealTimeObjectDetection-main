# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 14:42:45 2022
모든 h5로 저장한 모델을 한꺼번에 tensorflow savemodel로 컨버전 한다.
@author: headway
"""
import numpy as np
import os, shutil, sys
import tensorflow as tf
import matplotlib.pyplot as plt
# 한글 폰트 사용을 위해서 세팅
from matplotlib import font_manager, rc

from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import load_model
import natsort
import time
import pandas as pd
import argparse

from label_tools import *

font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
plt.rcParams['axes.unicode_minus'] = False  ## 추가해줍니다. 

#GPU 사용시 풀어 놓을 것
"""
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
"""
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR) 
#----------------------------
DEFAULT_OBJ_TYPES = ['ch','hr','vr','or']
SAVE_FOLDER_NAME = 'saved_model'
#----------------------------
DEFAULT_MODEL_PATH = None
DEFAULT_WEIGHT_PATH = None
CATEGORIES_FILE_NAME = None
DEFAULT_MODEL_DIR = None
filelist = None
DEFAULT_SAVEMODEL_PATH = None  #SaveModel 저장 위치
MIDDLE_PATH = 'exported-models'
DEFAULT_SAVE_MODEL_DIR = None

for DEFAULT_OBJ_TYPE in DEFAULT_OBJ_TYPES :

    if DEFAULT_OBJ_TYPE == 'ch':
        DEFAULT_MODEL_DIR = 'char_model'
        DEFAULT_SAVE_MODEL_DIR = 'ch_model'
        CATEGORIES_FILE_NAME = 'character_categories.txt'
    elif DEFAULT_OBJ_TYPE == 'vr':
        DEFAULT_MODEL_DIR = 'vreg_model'
        DEFAULT_SAVE_MODEL_DIR = 'vr_model'
        CATEGORIES_FILE_NAME = 'vregion_categories.txt'
    elif DEFAULT_OBJ_TYPE == 'hr':
        DEFAULT_MODEL_DIR = 'hreg_model'
        DEFAULT_SAVE_MODEL_DIR = 'hr_model'
        CATEGORIES_FILE_NAME = 'hregion_categories.txt'
    elif DEFAULT_OBJ_TYPE == 'or':
        DEFAULT_MODEL_DIR = 'oreg_model'
        DEFAULT_SAVE_MODEL_DIR = 'or_model'
        CATEGORIES_FILE_NAME = 'oregion_categories.txt'    

    filelist =  os.listdir(os.path.join(ROOT_DIR,DEFAULT_MODEL_DIR))    
    for fn in filelist :
        if 'model' in fn :
            DEFAULT_MODEL_PATH = os.path.join(ROOT_DIR,DEFAULT_MODEL_DIR,fn)
            
        if 'weight' in fn:
            #read weight value from trained dir
            DEFAULT_WEIGHT_PATH = os.path.join(ROOT_DIR,DEFAULT_MODEL_DIR,fn)



    DEFAULT_SAVEMODEL_PATH = os.path.join(ROOT_DIR,MIDDLE_PATH,DEFAULT_SAVE_MODEL_DIR,SAVE_FOLDER_NAME)

    # Initiate argument parser
    parser = argparse.ArgumentParser(
        description="object split and save in jpeg and annotation files")
    parser.add_argument("-t",
                        "--modeltype",
                        help="Set model type example ch, hr, or",
                        type=str,default=DEFAULT_OBJ_TYPE)
    parser.add_argument("-m",
                        "--modelfile",
                        help="Label model file where the text files are stored.",
                        type=str,default=DEFAULT_MODEL_PATH)
    parser.add_argument("-w",
                        "--weightfile",
                        help="Label weight file where the text files are stored.",
                        type=str,default=DEFAULT_WEIGHT_PATH)

    parser.add_argument("-d",
                        "--destpath",
                        help="Destination path",
                        type=str,default=DEFAULT_SAVEMODEL_PATH)


    args = parser.parse_args()



        
    #read model
    model = load_model(args.modelfile)
    #read weight value from trained dir
    weight_path = args.weightfile
    model.load_weights(weight_path)

    if not os.path.isdir(args.destpath):
        createFolder(args.destpath)

    model.save(args.destpath)

    #카테고리 파일을 복사한다.
    src = os.path.join(ROOT_DIR,DEFAULT_MODEL_DIR,CATEGORIES_FILE_NAME)
    dst = os.path.join(ROOT_DIR,MIDDLE_PATH,DEFAULT_SAVE_MODEL_DIR,CATEGORIES_FILE_NAME)
    shutil.copyfile(src,dst)

    print('Tensorflow SaveModel로 {}에 저장했습니다'.format(args.destpath))
        





