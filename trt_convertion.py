# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 09:35:05 2022

@author: headway
"""

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import time
import logging
import numpy as np

import tensorflow as tf
print("TensorFlow version: ", tf.__version__)
import tensorflow.experimental.tensorrt as trt
#from tensorflow.python.saved_model import tag_constants

output_saved_model_dir = 'C:\SPB_Data\RealTimeObjectDetection-main\exported-models\car-plate\car-plate.trt'
params = tf.experimental.tensorrt.ConversionParams(
    precision_mode='FP16')
converter = tf.experimental.tensorrt.Converter(
    input_saved_model_dir="C:\SPB_Data\RealTimeObjectDetection-main\exported-models\car-plate\saved_model\saved_model.pb", conversion_params=params)
converter.convert()
converter.save(output_saved_model_dir)
