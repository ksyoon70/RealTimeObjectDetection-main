""" Sample TensorFlow JSON-to-TFRecord converter

usage: generate_tfrecord.py [-h] [-x XML_DIR] [-l LABELS_PATH] [-o OUTPUT_PATH] [-i IMAGE_DIR] [-c CSV_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -x XML_DIR, --xml_dir XML_DIR
                        Path to the folder where the input .xml files are stored.
  -l LABELS_PATH, --labels_path LABELS_PATH
                        Path to the labels (.pbtxt) file.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path of output TFRecord (.record) file.
  -i IMAGE_DIR, --image_dir IMAGE_DIR
                        Path to the folder where the input image files are stored. Defaults to the same directory as XML_DIR.
  -c CSV_PATH, --csv_path CSV_PATH
                        Path of output .csv file. If none provided, then no file will be written.
                        
용도 :  영상 관련 json을 읽어서 label_map.txt를 만든다.
다만 필터파일을 읽어서 변형하여 label_map.txt을 만든다.
"""

import os,sys
import glob
import pandas as pd
import io
import xml.etree.ElementTree as ET
import argparse
import json	

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow.compat.v1 as tf
from PIL import Image
from object_detection.utils import dataset_util, label_map_util
from collections import namedtuple
import numpy as np


#========================
# 여기의 내용을 용도에 맞게 수정한다.
usage = 'train' # train or valid
dataset_category='car-plate' #plateimage
bFilterMap = None # filter map을 사용하는지 여부
#========================

if dataset_category == 'car-plate':
    label_file = 'label_map.pbtxt'
    fsLabelFileName = "LPR_Car-Plate_Labels_2.txt" #변경하고자 하는 레이블 명이다.
    filterFileName = "Car_PlateFiltermap.txt"  #필터 맵 파일이다.
    bFilterMap = True
elif dataset_category == 'plateimage':
    label_file = 'char_number_label_map.pbtxt'
    fsLabelFileName = "LPR_Labels2.txt"
    filterFileName = "LPR_Filtermap.txt"  #필터 맵 파일이다.
    bFilterMap = True
elif dataset_category == 'mplateimage':
    label_file = 'char_number_label_map.pbtxt'
    fsLabelFileName = "LPR_Labels2.txt"
    filterFileName = "LPR_Filtermap.txt"  #필터 맵 파일이다.
    bFilterMap = True    
elif dataset_category == 'plate':          # 도공 위반촬영장치와 같이 영상에 1개의 차량만 있는 경우에 사용한다. 
    label_file = 'platelabel_map.pbtxt'
    fsLabelFileName =  "LPR_Plate_Labels.txt"
    bFilterMap = False 

ROOT_DIR = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

#ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR) 

#레이블을 읽어 들여서 필요한 클래스를 파악한다.
LABEL_FILE_PATH = os.path.join(ROOT_DIR,fsLabelFileName)


if  not os.path.exists(LABEL_FILE_PATH):
    print("Label file {0} isn't exists".format(LABEL_FILE_PATH))
    sys.exit()
    
fLabels = pd.read_csv(LABEL_FILE_PATH, header = None )
CLASS_NAMES = fLabels[0].values.tolist()
    


ConvMap = {}
if bFilterMap :
    #필터맵을 통하여 라별 변환할 항목을 읽는다.
    FILTERMAP_FILE_PATH = os.path.join(ROOT_DIR,filterFileName)
    if  not os.path.exists(FILTERMAP_FILE_PATH):
        print("FilterMap File {0} isn't exists".format(FILTERMAP_FILE_PATH))
        sys.exit()
    
    cLabels = pd.read_csv(FILTERMAP_FILE_PATH, header = None )

    for i, value in enumerate(cLabels[0]):
        ConvMap[value] = cLabels[1][i]

WORKSPACE_PATH = os.path.join(ROOT_DIR,'Tensorflow','workspace')
SCRIPTS_PATH = os.path.join(ROOT_DIR,'Tensorflow','scripts')
ANNOTATION_PATH = os.path.join(WORKSPACE_PATH,'annotations')
IMAGE_PATH =  os.path.join(WORKSPACE_PATH,'images',dataset_category)
MODEL_PATH = os.path.join(WORKSPACE_PATH , 'models',dataset_category)
PRETRAINED_MODEL_PATH = os.path.join(WORKSPACE_PATH , 'pre-trained-models')
CONFIG_PATH = os.path.join( MODEL_PATH ,'my_ssd_mobnet','pipeline.config')
CHECKPOINT_PATH = os.path.join( MODEL_PATH , 'my_ssd_mobnet')



DEFAULT_ANNOTATION_PATH = os.path.join(IMAGE_PATH,usage)
DEFAULT_LABEL_PATH = os.path.join(ANNOTATION_PATH,label_file)
DEFAULT_OUPUT_PATH = os.path.join(ANNOTATION_PATH, usage + '.record')

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Sample TensorFlow JSON-to-TFRecord converter")
parser.add_argument("-j",
                    "--json_dir",
                    help="Path to the folder where the input .xml files are stored.",
                    type=str,default=DEFAULT_ANNOTATION_PATH)
parser.add_argument("-l",
                    "--labels_path",
                    help="Path to the labels (.pbtxt) file.", type=str, default=DEFAULT_LABEL_PATH)
parser.add_argument("-o",
                    "--output_path",
                    help="Path of output TFRecord (.record) file.", type=str,default=DEFAULT_OUPUT_PATH)
parser.add_argument("-i",
                    "--image_dir",
                    help="Path to the folder where the input image files are stored. "
                         "Defaults to the same directory as XML_DIR.",
                    type=str, default=None)
parser.add_argument("-c",
                    "--csv_path",
                    help="Path of output .csv file. If none provided, then no file will be "
                         "written.",
                    type=str, default=None)

args = parser.parse_args()

if args.image_dir is None:
    args.image_dir = args.json_dir

label_map = label_map_util.load_labelmap(args.labels_path)
label_map_dict = label_map_util.get_label_map_dict(label_map)


def xml_to_csv(path):
    """Iterates through all .xml files (generated by labelImg) in a given directory and combines
    them in a single Pandas dataframe.

    Parameters:
    ----------
    path : str
        The path containing the .xml files
    Returns
    -------
    Pandas DataFrame
        The produced dataframe
    """

    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def json_to_csv(path):
    """Iterates through all .xml files (generated by labelImg) in a given directory and combines
    them in a single Pandas dataframe.

    Parameters:
    ----------
    path : str
        The path containing the .xml files
    Returns
    -------
    Pandas DataFrame
        The produced dataframe
    """
    json_list = []
    boxes =[]
    width = 0
    height = 0
    for filename in glob.glob(path + '/*.json'):
        print('processing {}...'.format(filename))
        with open(filename, 'r',encoding="UTF-8") as f:
            json_data = json.load(f)
            width = json_data['imageWidth']
            height = json_data['imageHeight']
            for item, shape in enumerate(json_data['shapes']):
                points = shape['points']
                shape_type = shape['shape_type']
                label = shape['label']
                
                if label  in ConvMap:
                    label = ConvMap[label]
                if any(item == label for item in CLASS_NAMES):
                    arr =np.array(points)
                    if shape_type == 'polygon':
                        xmin = np.min(arr[:,0])
                        ymin = np.min(arr[:,1])
                        xmax = np.max(arr[:,0])
                        ymax = np.max(arr[:,1])
                        coors = [xmin, ymax]
                        points.insert(1, coors)
                        coors = [xmax, ymin]
                        points.insert(3, coors)
                        boxes.append(points)
                    elif shape_type == 'rectangle':
                        xmin = np.min(arr[:,0])
                        ymin = np.min(arr[:,1])
                        xmax = np.max(arr[:,0])
                        ymax = np.max(arr[:,1])
                        coors = [xmin, ymax]
                        points.insert(1, coors)
                        coors = [xmax, ymin]
                        points.insert(3, coors)
                        boxes.append(points)
                    else:
                        xmin = np.min(arr[:,0])
                        ymin = np.min(arr[:,1])
                        xmax = np.max(arr[:,0])
                        ymax = np.max(arr[:,1])
                        coors = [xmin, ymax]
                        points.insert(1, coors)
                        coors = [xmax, ymin]
                        points.insert(3, coors)
                        boxes.append(points)
                        
                    #필터에 있는 내용이면 레이블 이름을 변경한다.
                    shape['label'] = label  #레이블 무조건 업데이트....
                else:
                    continue
                value= (json_data['imagePath'],width,height,shape['label'],xmin,ymin,xmax,ymax)
                json_list.append(value)
                #print(value)
    
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    json_df = pd.DataFrame(json_list, columns=column_name)
    return json_df

def class_text_to_int(row_label):
    if row_label  in ConvMap:
        row_label = ConvMap[row_label]
    
    return label_map_dict[row_label]


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    try:
        with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
            encoded_jpg = fid.read()
    except Exception as e:
            print('{} 읽기 오류'.format(group.filename))
            return
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):

    writer = tf.python_io.TFRecordWriter(args.output_path)
    path = os.path.join(args.image_dir)
    examples = json_to_csv(args.json_dir)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        if tf_example!= None:
            writer.write(tf_example.SerializeToString())
    writer.close()   
    print('Successfully created the TFRecord file: {}'.format(args.output_path))
    if args.csv_path is not None:
        examples.to_csv(args.csv_path, index=None)
        print('Successfully created the CSV file: {}'.format(args.csv_path))


if __name__ == '__main__':
    tf.app.run()
