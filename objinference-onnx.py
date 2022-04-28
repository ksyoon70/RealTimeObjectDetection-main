# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 19:50:19 2022
Save_Model을  onnx 파일로 변환하여 모델을 읽고 인퍼런스를 수행하는 프로그램이다.
@author: 윤경섭
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

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import matplotlib.image as Image
import time

import onnx
from onnx import numpy_helper

import onnxruntime as rt


#========================
# 여기의 내용을 용도에 맞게 수정한다.
dataset_category='plate'
test_dir_name = 'test'
show_image = True
save_image = True
#========================


ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR) 


WORKSPACE_PATH = os.path.join(ROOT_DIR,'Tensorflow','workspace')
ANNOTATION_PATH = os.path.join(WORKSPACE_PATH,'annotations')
IMAGE_PATH =  os.path.join(WORKSPACE_PATH,'images',dataset_category)

ONNX_MODEL_DIR = ROOT_DIR

#테스트할 이미지 디렉토리
images_dir = os.path.join(IMAGE_PATH,test_dir_name)
result_dir = os.path.join(IMAGE_PATH,'result')

if not os.path.isdir(result_dir):
	os.mkdir(result_dir)


sess = rt.InferenceSession(os.path.join(ONNX_MODEL_DIR,'model.onnx')) 
# input 이름을 읽어옴
input_name = sess.get_inputs()[0].name

output_name = []
#ouput 의 갯수
output_len = len(sess.get_outputs()[:])

# ouput 이름을 읽어 옴.
for i in range(output_len):
   output_name.append(sess.get_outputs()[i].name)
   

category_index = label_map_util.create_category_index_from_labelmap(os.path.join(ANNOTATION_PATH, 'label_map.pbtxt'))


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
    
fileNum = len(os.listdir(images_dir))
cnt = 0;
for filename in os.listdir(images_dir):
    cnt += 1
    image_path = os.path.join(images_dir,filename)
    if os.path.exists(image_path) :
        imgRGB  = imread(image_path)
        
        image_np = np.array(imgRGB)
        
        image_input = np.expand_dims(image_np.astype(np.uint8), axis=0)

       
        result = sess.run(output_name, {input_name: image_input})
        
        detections = {}
        for i in range(output_len):
           detections[output_name[i]] = result[i][0]

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        
        label_id_offset = 1
        image_np_with_detections = image_np.copy()
        
        image_width = imgRGB.shape[1]
        image_height = imgRGB.shape[0]
        
       
        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes'],
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=.5,
                    agnostic_mode=False)
        
        
        save_fig_path = os.path.join(result_dir,filename)
        
        resultImage = cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB)
        
        if save_image :
            if not (save_fig_path is None):
                plt.imsave(save_fig_path, resultImage)

        
        if show_image :
            plt.imshow(resultImage)
            plt.show()
            





