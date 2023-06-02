
import numpy as np
import os, shutil
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
from plate_recog_common import *

#----------------------------
DEFAULT_LABEL_FILE = "./LPR_Labels1.txt"  #라벨 파일이름
CHAR_MODEL_DIR = 'char_model'
MODEL_FILE_NAME = None #'character_resnet50_20220820-002035_model_epoch_35_val_acc_0.9025.h5'
CMODEL_PATH = None
WEIGHT_FILE_NAME = None #'character_resnet50_20220820-001359_weights_epoch_025_val_acc_0.905.h5'
CWEIGHT_PATH = None
CATEGORIES_FILE_NAME = 'character_categories.txt'
CATEGORIES_FILE_PATH = os.path.join(ROOT_DIR,CHAR_MODEL_DIR,CATEGORIES_FILE_NAME)
categories = []
#----------------------------

#read model
def char_det_init_fn():
    
    fLabels = pd.read_csv(DEFAULT_LABEL_FILE, header = None )
    LABEL_FILE_CLASS = fLabels[0].values.tolist()
    LABEL_FILE_HUMAN_NAMES = fLabels[1].values.tolist()
    global CLASS_DIC
    CLASS_DIC = dict(zip(LABEL_FILE_CLASS, LABEL_FILE_HUMAN_NAMES))
    CLASS_DIC['x'] = 'x'
    
    filelist =  os.listdir(os.path.join(ROOT_DIR,CHAR_MODEL_DIR))

    for fn in filelist :
        if 'model' in fn :
            CMODEL_PATH = os.path.join(ROOT_DIR,CHAR_MODEL_DIR,fn)
           
        if 'weight' in fn:
            #read weight value from trained dir
            CWEIGHT_PATH = os.path.join(ROOT_DIR,CHAR_MODEL_DIR,fn)

    model = load_model(CMODEL_PATH)
    model.load_weights(CWEIGHT_PATH)
    
    global categories
    catLabels = pd.read_csv(CATEGORIES_FILE_PATH, header = None )
    categories = catLabels[0].values.tolist()
    categories.append('no_categorie')
    
    return model


def char_det_fn(model, img_np, ch_thresh_hold, predict_anyway = False) :
    img_np = np.expand_dims(img_np,axis=0)
    preds = model.predict(img_np)
    index = np.argmax(preds[0],0)
    predic_label = None
    if preds[0][index] > ch_thresh_hold :
        predic_label = CLASS_DIC[categories[index]]
        print('predict:{}'.format(predic_label))
    else:
        if predict_anyway :     #확률에 관계 없이 무조건 인식한 값을 원할 경우
            predic_label = CLASS_DIC[categories[index]]
        else :                  # 일반적으로 확률이 낮으면 x 값을 리턴한다.
            predic_label = 'x'
        
        predic_label1 = CLASS_DIC[categories[index]]
        print('predict ?:{}'.format(predic_label1))
        
    print('확률:{}%'.format(preds[0][index]*100 ))
    return  predic_label, preds[0][index]