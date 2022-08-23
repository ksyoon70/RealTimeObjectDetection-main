# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:19:22 2022

@author:  윤경섭
"""

import os,sys
import pandas as pd
import cv2
import argparse
import json
from collections import OrderedDict
from PIL import Image
from shutil import copyfile
import re
import numpy as np


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

# 디렉토리 생성
# points_xy =[[x1,y1],[x2,y2],[x3,y3],[x4,y4]] 이런식의 입력이 들어와야 한다.
def insertlabel_with_points(shapes, points_xy,label, shape_type= 'polygon'):
    lable_info = {'label':label}
    lable_info['points']= points_xy
    lable_info['group_id'] = None
    lable_info["shape_type"] = shape_type
    lable_info["flags"] = {}
    shapes.append(lable_info)
    
# points_x =[x1, x2, x3, x4] 
# points_y =[y1, y2, y3, y4] 이런식의 입력이 들어와야 한다.  
def insertlabel_with_xypoints(shapes, points_x, points_y,label, shape_type= 'polygon'):
    points_xy= [ [x,y] for x, y in zip(points_x,points_y)] 
    lable_info = {'label':label}
    lable_info['points']= points_xy
    lable_info['group_id'] = None
    lable_info["shape_type"] = shape_type
    lable_info["flags"] = {}
    shapes.append(lable_info)
    
    
# 문자가 한글인지 여부를 리턴함. 한글이면 True, 아니면 False    
def isHangul(text):
    #Check the Python Version
    encText = text
    hanCount = len(re.findall(u'[\u3130-\u318F\uAC00-\uD7A3]+', encText))
    return hanCount > 0

# 파일 이름을 넣으면 인식 내용을 분할하여 리턴한다.
def splitPlateName(filename):
    #filename : 파일 이름
    sIndex = filename.rfind('_')
    eIndex = filename.rfind('.')
    print("start {0} end {1}".format(sIndex,eIndex))
    
    if sIndex >=0 and eIndex >=0 :
        recogstr = filename[sIndex + 1 : eIndex]
    else :
        recogstr = ""
    
    print("인식내용 : {0}".format(recogstr))
    
    
    hangul = []
    region = ""
    type_ch = ""
    usage = ""
    number=""
    isyoung = False
    
    for ix, ch in enumerate(recogstr):
        if isHangul(ch) :
            hangul.append(True)
        else:
            hangul.append(False)
    
    for ix, val in enumerate(hangul):
        if val == True and ix == 0 :
            region += recogstr[ix]
        elif  val == True and ix == 1:
            region += recogstr[ix]
    #지역 문자 찾기
    if len(region) == 1 and hangul[0] == True :
        region = region + 'x'
    elif len(region) == 1 and hangul[1] == True:
        region = 'x' + region
    
    #type 용도 찾기
    if len(region) > 0 :
        partial = recogstr[len(region) :]
        hangul = hangul[len(region):]
    else:
        partial = recogstr
    
            
    partial = partial[:-4]
    if len(partial) > 0 :
        usage =  partial[-1:]
    partial = partial[:-1]
    if len(partial) > 0 :
        type_ch = partial
        
    recogstrlen = len(recogstr)
    if not (recogstr is "") : #인식된 내용이 있으면.
        if recogstr[-1:] == '영' :
            number = recogstr[-5:-1]
        else :
            number = recogstr[-4:]
    
    print("지역: {0}".format(region))
    print("타입: {0}".format(type_ch))
    print("용도: {0}".format(usage))
    print("번호: {0}".format(number))
    # region: 지역문자
    # type_ch : 타입 숫자
    # usage : 용도 문자
    # number : 인식 숫자
    # isyoung : 영자 포함 여부
    # recogstr : 인식 내용
    return region, type_ch, usage, number, isyoung, recogstr


twolinePlate = [1,2,4,6,7]  #tyep9는 3자리 번호판


def IoU(box1, box2):
    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou, box1_area, box2_area,inter


def predictPlateNumber(objTable,dictlabel,class_names) :
    # objTable : numpy array
    # [ [class_id, 신뢰도, y1, x1, y2, x2] , []] 형식의 입력이다.
    plate_str = "" # 번호판 문자
    if(len(objTable) > 1):
        plate2line = False
        # 번호판 상하단 구분 위한 코드
        ref = objTable.mean(axis = 0)
        types = objTable[:,0]
        if any(type in twolinePlate  for type in types) :
            plate2line = True
            print("2line")
        else:
            print("1line")
        plateBox  =  objTable[0][2:]
        plateTable = []
        if plate2line :
            # 2line 번호판이면...
            # 1line 과 2line으로 나눈다.
            onelineTable = []
            twolineTalbe = []
            
            for table in objTable[1:]:
                if table[2] <= ref[2] :
                    onelineTable.append(list(table))
                else:
                    twolineTalbe.append(list(table))
            onelineTable = np.array(onelineTable)
            twolineTalbe = np.array(twolineTalbe)
            if onelineTable.size :
                onelineTable = onelineTable[onelineTable[:,3].argsort()] #onelineTable[:,3].argsort() 순서대로 인덱스를 반환
            if twolineTalbe.size :
                twolineTalbe = twolineTalbe[twolineTalbe[:,3].argsort()]
            if onelineTable.size and twolineTalbe.size:
                plateTable = np.append(onelineTable,twolineTalbe, axis=0)
            elif onelineTable.size:
                plateTable =  onelineTable
            elif twolineTalbe.size:
                plateTable =  twolineTalbe

        else:
                onelineTable = objTable[1:]
                plateTable = onelineTable[onelineTable[:,3].argsort()]
        """        
        #숫자가 있을 때 다른 문자 안에 포함되면 삭제한다.
        boxes = plateTable[:,2:]
        boxes[:,[0,1]] = boxes[:,[1,0]] 
        boxes[:,[2,3]] = boxes[:,[3,2]] 
        
        #print("plateTable : {0}".format(plateTable))
        
        isbreak = False
        for i in range(0,len(boxes) - 1):
            box1 = boxes[i]
            for box2 in boxes[i+1 :]:
                iou,box1_area, box2_area,inter = IoU(box1,box2)
                if iou > 0.05 and box1_area < box2_area :
                    plateTable = np.delete(plateTable, i, axis=0)
                    print("box {0} 삭제 ".format(i))
                    isbreak = True
            
            if isbreak :
                break;
        """        
        #print("plateTable : {0}".format(plateTable))
                
        classIDnum = list(map(int, plateTable[plateTable[:,0] > 0,0])) #번호판외 검지 id
    
        
        for id in classIDnum:
            plate_str = plate_str + dictlabel[class_names[id]]
    
    print("MaskRcnn 인식 내용 {0}".format(plate_str))
    return plate_str


# 포인트가 Polygon 내에 있는지 확인한다. 포인트가 폴리곤 안에 있으면 True 아니면 False를 리턴한다.
# 모든 입력 포인트는 numpy로 입력 받는다.
def PointInPolygon( tpoint, polygon) :

    bPointInPolygon = False
    iCrosses = 0
    
    #교차점 수
    polylen = len(polygon)
    
    for i, point in enumerate(polygon) :
        
        j = (i + 1) % polylen
        
        npoint = polygon[j] # next point
        ymax = np.where(npoint[1] >= point[1], npoint[1], point[1])
        ymin = np.where(npoint[1] >= point[1], point[1], npoint[1])
        
        if tpoint[1] > ymin and tpoint[1] < ymax : # 같은 크기을 때를 방지하려면 등호는 포함하지 않는다.
            Xat = (npoint[0] - point[0])/(npoint[1] - point[1])*(tpoint[1] - point[1]) + point[0] #직선의 방정식에서 겹치는 X를 구한다.
            
            if tpoint[0] > Xat :
                iCrosses = iCrosses + 1
                
    
    if (iCrosses % 2) == 0 : # iCrosses가 짝수이고 0 이상 이어야 한다.
        bPointInPolygon = False
    else :
        bPointInPolygon = True
        
    return bPointInPolygon

# 폴리콘이 폴리곤 안에 겹치는지 화인한다. 겹치는 부분이 있으면 True 아니면 False를 리턴한다
# 모든 입력 포인트는 numpy로 입력 받는다.
def PolygonOverlab(spolygon, tpolygon) :
    
    bresult = False
    
    for point in spolygon :
        
        bresult = PointInPolygon(point, tpolygon)
        
        if (bresult):
            break
        
    return bresult

# polygon 좌표가 box를 넘지 않도록 제한을 둔다.
# 단 polygon 좌표는 box 내에 최소 한점은 있다고 가정한다.
def SupressInBox(polygon , box):
    box_x = box[:,0]
    box_y = box[:,1]
    
    min_x = np.min(box_x,axis=0)
    min_y = np.min(box_y,axis=0)
    max_x = np.max(box_x,axis=0)
    max_y = np.max(box_y,axis=0)
    for point in polygon :
        if point[0] < min_x :
            point[0] = min_x
        if point[0] > max_x:
            point[0] = max_x
        if point[1] < min_y :
            point[1] = min_y
        if point[1] > max_y:
            point[1] = max_y
            
    return polygon


# box 좌표를 polygon 형태로 만든다.
def box2polygon( box):
    box_x = box[:,0]
    box_y = box[:,1]
    
    min_x = np.min(box_x,axis=0)
    min_y = np.min(box_y,axis=0)
    max_x = np.max(box_x,axis=0)
    max_y = np.max(box_y,axis=0)
    
    polygon = np.array([[min_x,min_y],[max_x,min_y],[max_x,max_y],[min_x,max_y]])
    
    return polygon

# 디렉토리 생성
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
    
            
#Object Detection API에서의  번호/문자 인식 내용을 추출 한다.    
def predictPlateNumberODAPI(detect, platetype_index, category_index, CLASS_DIC, twoLinePlate) :
    
    objTable = []
    
    num_detections = detect['num_detections']
    
    for i in range(0,num_detections) :
        box = detect['detection_boxes'][i]
        class_id = detect['detection_classes'][i] + 1
        score = detect['detection_scores'][i]
        item = [class_id, score, box[0],box[1],box[2],box[3]]
        objTable.append(item)
    
    objTable = np.array(objTable)
    
    plate_str = "" # 번호판 문자
    if(num_detections > 1):
        plate2line = False
        # 번호판 상하단 구분 위한 코드
        ref = objTable[:,2].mean(axis = 0)
        type = platetype_index
        if type in twolinePlate or twoLinePlate :
            plate2line = True
            print("2line")
        else:
            print("1line")
        plateTable = []
        if plate2line :
            # 2line 번호판이면...
            # 1line 과 2line으로 나눈다.
            onelineTable = []
            twolineTalbe = []
            
            for type in objTable:
                if type[2] <= ref :
                    onelineTable.append(list(type))
                else:
                    twolineTalbe.append(list(type))
            onelineTable = np.array(onelineTable)
            twolineTalbe = np.array(twolineTalbe)
            if onelineTable.size :
                onelineTable = onelineTable[onelineTable[:,-1].argsort()] #onelineTable[:,3].argsort() 순서대로 인덱스를 반환
                if onelineTable[0,0] == 13:  # hReg 첫글자 가로 지역문자이면...
                    res = onelineTable[1:,:]
                    if res.shape[1] > 2:
                        res = res[(-res[:,1]).argsort()[:2]] #스코어 순으로 2개만 추린다.
                        #다시 정렬한다.
                        res = res[res[:,-1].argsort()]
                        arr = np.array([onelineTable[0]])
                        arr = np.concatenate([arr,res],axis=0)
                        onelineTable = arr
                        
            if twolineTalbe.size :
                twolineTalbe = twolineTalbe[twolineTalbe[:,-1].argsort()]
                twolineTalbescore = twolineTalbe[:,0]
                result = np.where(twolineTalbescore == 11)
                #용도문자 이후 오른쪽 숫자가 4개 이상이면 스코어에 따라서 삭제한다.
                if len(result) and len(result[0]) > 0 :  # Char 첫글자 가로 지역문자이면...
                    cindex = result[0][0]
                    res = twolineTalbe[cindex + 1:,:]
                    if res.shape[1] > 4:
                        res = res[(-res[:,1]).argsort()[:4]] #스코어 순으로 4개만 추린다.
                        #다시 정렬한다.
                        res = res[res[:,-1].argsort()]
                        arr = twolineTalbe[0 : cindex + 1]
                        arr = np.concatenate([arr,res],axis=0)
                        twolineTalbe = arr
            if onelineTable.size and twolineTalbe.size:
                plateTable = np.append(onelineTable,twolineTalbe, axis=0)
            elif onelineTable.size:
                plateTable =  onelineTable
            elif twolineTalbe.size:
                plateTable =  twolineTalbe

        else:
                onelineTable = objTable
                plateTable = onelineTable[onelineTable[:,-1].argsort()]
                onelineTalbescore = plateTable[:,0]
                result = np.where(onelineTalbescore == 11)
                #용도문자 이후 오른쪽 숫자가 4개 이상이면 스코어에 따라서 삭제한다.
                if len(result) and len(result[0]) > 0 :  # Char 첫글자 가로 지역문자이면...
                    cindex = result[0][0]
                    res = plateTable[cindex + 1:,:]
                    if res.shape[1] > 4:
                        res = res[(-res[:,1]).argsort()[:4]] #스코어 순으로 4개만 추린다.
                        #다시 정렬한다.
                        res = res[res[:,-1].argsort()]
                        arr = plateTable[0 : cindex + 1]
                        arr = np.concatenate([arr,res],axis=0)
                        plateTable = arr
        """        
        #숫자가 있을 때 다른 문자 안에 포함되면 삭제한다.
        boxes = plateTable[:,2:]
        boxes[:,[0,1]] = boxes[:,[1,0]] 
        boxes[:,[2,3]] = boxes[:,[3,2]] 
        
        #print("plateTable : {0}".format(plateTable))
        
        isbreak = False
        for i in range(0,len(boxes) - 1):
            box1 = boxes[i]
            for box2 in boxes[i+1 :]:
                iou,box1_area, box2_area,inter = IoU(box1,box2)
                if iou > 0.05 and box1_area < box2_area :
                    plateTable = np.delete(plateTable, i, axis=0)
                    print("box {0} 삭제 ".format(i))
                    isbreak = True
            
            if isbreak :
                break;
        """        
        #print("plateTable : {0}".format(plateTable))
        num_detections = plateTable.shape[0] #갯수가 바뀔수 있다.
        for i in range(0,num_detections) :
            class_index = int(plateTable[i][0])
            if class_index <= 10 :
                name = category_index[class_index]['name']
                plate_str = plate_str + CLASS_DIC[name]
            else :
                plate_str = plate_str + category_index[class_index]['name']
    
    print("SSD 인식 내용 {0}".format(plate_str))
    return plate_str            
            
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

