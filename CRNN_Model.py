
import os
import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.models import load_model
import time 
from label_tools import *
# 한글 폰트 사용을 위해서 세팅
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)



class CRNN_Model():
    def __init__(self,model_path,weight_path,characters,max_length,**kwargs):
        super(CRNN_Model, self).__init__(**kwargs)
        self.prediction_model = tf.keras.models.load_model(model_path)
        self.prediction_model.load_weights(weight_path)
        self.max_length = max_length
        self.characters = characters
        
        self.char_to_num = layers.experimental.preprocessing.StringLookup(
            vocabulary=list(self.characters), num_oov_indices=0, mask_token=None
        )

        self.num_to_char = layers.experimental.preprocessing.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), num_oov_indices=0, mask_token=None, invert=True
        )
        
    def decode_batch_predictions(self,pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
            :, :self.max_length
        ]
    
        decoded = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)
        log_probs  = np.asarray(decoded[1])
        probabilities = np.exp(-log_probs)  # log probability로 반환하기 때문에 지수 함수로 바꿔줘야 한다.
        # Iterate over the results and get back the text
        output_text = []
        probs = probabilities.reshape((-1)).tolist()
        for ix, res in enumerate(results):
            ch = tf.strings.reduce_join(self.num_to_char(res)).numpy().decode('utf-8')
            output_text.append(ch)
            
        return output_text,probs
        
    def predict(self,batch_images):
        preds = self.prediction_model.predict(batch_images)
        pred_texts, probs = self.decode_batch_predictions(preds)
        
        return pred_texts, probs
        
        