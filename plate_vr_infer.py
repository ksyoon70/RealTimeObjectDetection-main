
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
VR_MODEL_DIR = 'vreg_model'
MODEL_FILE_NAME ='vregion_resnet50_model_epoch_37_val_acc_0.8906_20220819-102828.h5'
VR_MODEL_PATH = os.path.join(ROOT_DIR,VR_MODEL_DIR,MODEL_FILE_NAME)
WEIGHT_FILE_NAME = 'vregion_resnet50_20220819-102738_model_weights_epoch_27_val_acc_0.891.h5'
VR_WEIGHT_PATH = os.path.join(ROOT_DIR,VR_MODEL_DIR,WEIGHT_FILE_NAME)
CATEGORIES_FILE_NAME = 'vregion_categories.txt'
CATEGORIES_FILE_PATH = os.path.join(ROOT_DIR,VR_MODEL_DIR,CATEGORIES_FILE_NAME)
categories = []
VR_THRESH_HOLD = 0.5
#----------------------------

#read model
def vr_det_init_fn():
    
    fLabels = pd.read_csv(DEFAULT_LABEL_FILE, header = None )
    LABEL_FILE_CLASS = fLabels[0].values.tolist()
    LABEL_FILE_HUMAN_NAMES = fLabels[1].values.tolist()
    global CLASS_DIC
    CLASS_DIC = dict(zip(LABEL_FILE_CLASS, LABEL_FILE_HUMAN_NAMES))
    
    model = load_model(VR_MODEL_PATH)
    #read weight value from trained dir
    model.load_weights(VR_WEIGHT_PATH)
    global categories
    catLabels = pd.read_csv(CATEGORIES_FILE_PATH, header = None )
    categories = catLabels[0].values.tolist()
    categories.append('no_categorie')
    
    return model


def vr_det_fn(model, img_np) :
    img_np = np.expand_dims(img_np,axis=0)
    preds = model.predict(img_np)
    index = np.argmax(preds[0],0)
    predic_label = None
    if preds[0][index] > VR_THRESH_HOLD :
        predic_label = CLASS_DIC[categories[index]]
        print('predict:{}'.format(predic_label))
    else:
        predic_label = 'x'
        
    print('확률:{}%'.format(preds[0][index]*100 ))
    return  predic_label