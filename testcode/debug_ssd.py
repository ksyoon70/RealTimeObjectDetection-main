import os,sys
import subprocess
import tensorflow as tf
#import pdb
#from testarg import test1
sys.path.append('../models') 

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    
    for keys in keys_list:
    	FLAGS.__delattr__(keys)
        
del_all_flags(tf.flags.FLAGS)

#subprocess.call([sys.executable, 'C:\SPB_Data\models\research\object_detection\model_main_tf2.py', '--model_dir=Tensorflow\workspace\models\plate\my_ssd_mobnet', '--pipeline_config_path=Tensorflowworkspace\models\plate\my_ssd_mobnet\pipeline.config', '--num_train_steps=10000'])