# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 16:27:27 2022

@author: headway
"""
import os,sys
from uuid import RESERVED_MICROSOFT
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import matplotlib.image as Image
import numpy as np
import tensorflow as tf

IMG_SIZE = 224

ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR) 

global WORKSPACE_PATH
global ANNOTATION_PATH
global char_model
global CLASS_DIC
global REV_CLASS_DIC    # CLASS_DIC 반대의 형태 
global REV_VCLASS_DIC
global REV_HCLASS_DIC
global REV_OCLASS_DIC
global REV_CLASS6_DIC
global LABEL_FILE_CLASS
THRESH_HOLD = 0.8
PLATE_THRESH_HOLD = 0.35

@tf.function
def detect_fn(image, detection_model):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


@tf.function
def inner_detect_fn(image, detection_model):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def drawImage(image_path):
    image  = cv2.imread(image_path)
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(imgRGB)
    plt.pause(0.5)
    plt.show()
    
    
def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None

def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False
    
# def extract_sub_image(src_np, box, width, height, pad=False):
#     src_height, src_width, ch = src_np.shape
#     box_sy = int(src_height*box[0])
#     box_sx= int(src_width*box[1])
#     box_ey = int(src_height*box[2])
#     box_ex= int(src_width*box[3])
#     obj_img = src_np[box_sy:box_ey,box_sx:box_ex,:]
    
#     #번호판을 320x320 크기로 정규화 한다.
#     if fixratio :
#         desired_size = max(height,width)
#         old_size = [obj_img.shape[1],obj_img.shape[0]]
#         ratio = float(desired_size)/max(old_size)
#         new_size = tuple([int(x*ratio) for x in old_size])
#         #원영상에서 ratio 만큼 곱하여 리싸이즈한 번호판 영상을 얻는다.
#         cropped_img = cv2.resize(obj_img,new_size,interpolation=cv2.INTER_LINEAR)
#         dst_np = np.zeros((desired_size, desired_size, 3), dtype = "uint8")
#         #dst_np = cv2.cvtColor(dst_np, cv2.COLOR_BGR2RGB)
#         h = new_size[1]
#         w = new_size[0]
#         yoff = round((desired_size-h)/2)
#         xoff = round((desired_size-w)/2)
#         #320x320영상에 번호판을 붙여 넣는다.
#         dst_np[yoff:yoff+h, xoff:xoff+w , :] = cropped_img        
#     else :
#         desired_size = (height,width)
#         #원영상에서 ratio 만큼 곱하여 리싸이즈한 번호판 영상을 얻는다.
#         dst_np = cv2.resize(obj_img,desired_size,interpolation=cv2.INTER_LINEAR)
#         plt.imshow(dst_np)
#         plt.show()

#     return dst_np