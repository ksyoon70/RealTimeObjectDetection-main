import os,sys
import json
import numpy as np
from collections import OrderedDict

# NpEncoder class ================================================================
class NpEncoder(json.JSONEncoder):
   def default(self, obj):
       if isinstance(obj, np.integer):
           return int(obj)
       elif isinstance(obj, np.floating):
           return float(obj)
       elif isinstance(obj, np.ndarray):
           return obj.tolist()
       else:
           return super(NpEncoder, self).default(obj)
#==============================================================================

class JsonMng:
    
    def __init__(self,dst_path,image_shape,image_filename):
        self.json_data = OrderedDict()
        self.json_data['version'] = '5.0.1'
        self.json_data['flags'] = {}
        shapes=[]
        self.json_data['shapes'] = shapes
        self.json_data['imageData'] = None
        self.json_data['imageHeight'] = image_shape[0]
        self.json_data['imageWidth'] = image_shape[1]
        self.json_data['imagePath'] = image_filename
        self.dst_path = dst_path
        self.image_filename = image_filename
        self.json_filename = None
        
    #plate를 추가한다.    
    def addPlate(self,plateTable,category_index,CLASS_DIC,platebox,plateIndex,plate_shape):
        
        num_detections = len(plateTable)
        shapes = self.json_data['shapes']
        # 번호판 타입을 추가한다.
        platetype = label = 'type{}'.format(plateIndex)
        self.insertlabel_with_xypoints(shapes,platebox[0],platebox[1],label=label)
        for ix in range(0,num_detections) :
            class_index = int(plateTable[ix][0])
            if platetype == 'type13':
                label = plateTable[ix][-1]
                str =  CLASS_DIC[label]
            else:
                label = category_index[class_index]['name']
                #str =  CLASS_DIC[label]
                label_index = int(plateTable[ix][-1])
                label_key = list(CLASS_DIC.keys())[label_index]
                str = CLASS_DIC[label_key]
                if str == '○':      #예외처리
                    label = 'Cml'
            if not str == 'x' :  # x가 나오면 인식한게 아니기 때문.
                if class_index == 11 :  #용도문자
                    self.json_data['usage'] = str
                    self.json_data['type'] = str # 타입숫자 ?
                if class_index >= 12 :  #지역문자
                    self.json_data['region'] = str
                if  class_index == 14 :
                    commercial = True
                x1 = plateTable[ix][3]  # 이 좌표는 전체 영상 기준으로 한다.
                x2 = plateTable[ix][5]
                x3 = plateTable[ix][5]
                x4 = plateTable[ix][3]
                y1 = plateTable[ix][2]
                y2 = plateTable[ix][4]
                y3 = plateTable[ix][4]
                y4 = plateTable[ix][2]
                points_x = [ x1, x2, x2, x1]
                points_y = [ y1, y1, y2, y2]
                plate_real_height = platebox[1][2] - platebox[1][0]  #원래 번호판 높이
                points_x = [points_x[i]*plate_shape[1] + platebox[0][0]  for i in range(len(points_x))] #320 기준으로 좌표를 변환하고, 다시 전체영상 기준으로 바꾼다.
                points_y = [points_y[i]*plate_shape[0] + platebox[1][0]  for i in range(len(points_y))]
                self.insertlabel_with_xypoints(shapes,points_x,points_y,label=label)
                
        
        self.json_data['shapes'] = shapes
                
    # points_x =[x1, x2, x3, x4] 
    # points_y =[y1, y2, y3, y4] 이런식의 입력이 들어와야 한다.  
    def insertlabel_with_xypoints(self,shapes, points_x, points_y,label, shape_type= 'polygon'):
        points_xy= [ [x,y] for x, y in zip(points_x,points_y)] 
        lable_info = {'label':label}
        lable_info['points']= points_xy
        lable_info['group_id'] = None
        lable_info["shape_type"] = shape_type
        lable_info["flags"] = {}
        shapes.append(lable_info)
    #일반적인 레이블을 추가한다.
    #box는 box[0] = x 좌표들 box[1] =y 이다.    
    def addObject(self, box, label, shape_type= 'polygon') :
        shapes = self.json_data['shapes']
        self.insertlabel_with_xypoints(shapes,box[0],box[1],label=label)
        self.json_data['shapes'] = shapes
# add_platenum 인식번호를 파일이름에 추가할지 여부
# plateNumber 인식한 번호이름    
    def save(self,add_platenum=False, plateNumber=None):
        
        basefilename, ext = os.path.splitext(self.image_filename) 
        #if basefilename[-1] == 'c':
        #    basefilename =  basefilename[:-1]
        if add_platenum and plateNumber:
            ofilename = os.path.join(self.dst_path,basefilename + '_' + plateNumber)
            self.json_data['imagePath'] = basefilename + '_' + plateNumber + ext
            self.json_filename  = basefilename + '_' + plateNumber + '.json'
        else:
            ofilename = os.path.join(self.dst_path,basefilename)
            self.json_filename  = basefilename  + '.json'
        # json 파일을 저장한다.
        with open( ofilename +'.json','w', encoding='utf-8') as f:
                json.dump(self.json_data,f,ensure_ascii=False,indent="\t" , cls=NpEncoder)
                
  #저장한 파일 이름을 반환한다.              
    def getJsonFilename(self):
        return self.json_filename