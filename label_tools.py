# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:19:22 2022

@author:  윤경섭
"""

from genericpath import isfile
import os,sys,shutil
import pandas as pd
import cv2
import argparse
import json
from collections import OrderedDict
from PIL import Image
from shutil import copyfile
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import matplotlib.image as Image


IOU_THESHOLD = 0.3
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
    # box = (y1, x1, y2, x2)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[1], box2[1])
    y1 = max(box1[0], box2[0])
    x2 = min(box1[3], box2[3])
    y2 = min(box1[2], box2[2])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 )
    h = max(0, y2 - y1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou, box1_area, box2_area,inter

#box의 구조가 box = (y1, x1, y2, x2) 일때 사용한다.
def IoU2(box1, box2):
    # box = (y1, x1, y2, x2)
    box1_area = (box1[3] - box1[1] + 1) * (box1[2] - box1[0] + 1)
    box2_area = (box2[3] - box2[1] + 1) * (box2[2] - box2[0] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[1], box2[1])
    y1 = max(box1[0], box2[0])
    x2 = min(box1[3], box2[3])
    y2 = min(box1[2], box2[2])

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
        
#[classid, score, box[0],box[1],box[2],box[3]] 을  objTable 입력으로 받아서
# 그중에 classid가 한개만 있도록 objTable을 바꾼다. 이때 score가 가장 큰 것만 남긴다.
def classIdDoubleCheck(class_id,objTable) :
    #클라스 id가 단 1개만 있어야 하는데 2개 이상이 있으면 score에 따라 삭제 한 후 리턴한다.
    class_id_np = objTable[:,0] #클래스 id만 있는 numpy 배열을 얻어온다.
    col_size = objTable.shape[1]
    result = np.where(class_id_np == class_id)
    if len(result) and len(result[0]) > 0 : 
        if  len(result[0]) > 1: # class_id 가 1개 이상이면...   score 별로 정렬 시킨다.
            class_id_objTable = []
            for index in result[0]:
                class_id_objTable.append(objTable[index])
            #score 별로 정렬을 한다.
            class_id_objTable = np.array(class_id_objTable)
            class_id_objTable = class_id_objTable[class_id_objTable[:,1].argsort()]
            class_id_objTable = class_id_objTable[-1,:] #accendign order 이므로 score가 가장 큰 array만 취득
            
            arr = np.array([])
            for row in objTable :
                if class_id_objTable[0] != row[0] :
                    arr = np.concatenate([arr,row],axis=0)
           
            arr = np.concatenate([arr,class_id_objTable],axis=0)
            objTable = arr.reshape(-1,col_size)
            
        else: #해당 class_id가 1개 만 있다.
            return objTable
    else:
        return  objTable   
    
    return objTable    

def checkTwoNumAhead(rindex, objTable) :
    NUM_TH_HOLD = 0.7  #숫자 쓰레쉬 홀드
    if rindex <= 1 :
        return objTable
    else :
        #print('table {}'.format(objTable))
        numTable = objTable[0:rindex,:]
        if numTable[0,0] > 10 : #첫번째 오브젝트가 즉 숫자가 아니면.
            arr_index = [0]
            arr_index1 = numTable[1:,1].argsort() + 1
            if arr_index1.size > 2: #즉 숫자가 2개 이상이면...
                objTable = np.delete(objTable,arr_index1[0].item(),0)
        else : # score 별로 소팅
            arr_index1 = numTable[0:,1].argsort()
            score = objTable[arr_index1[0].item(),1]
            if arr_index1.size > 2 and score < NUM_TH_HOLD: #즉 숫자가 2개 이상이면...
                objTable = np.delete(objTable,arr_index1[0].item(),0) 
        return objTable
# 오직 1개의 region만 존재 하도록 한다.
def onlyOneRegion(objTable, twoLinePlate) :
    col_size = objTable.shape[1]
    
    regTable = [row  for row in objTable if row[0] > 11 ]
    
    if len(regTable) > 0 :
    
        regTable = np.array(regTable)
        
        #regionTable을 score로 소팅한다.
        regTable = regTable[(-regTable[:,1]).argsort()]
        regTable = regTable[0,:]
        arr = np.array([])
        if regTable[0] == 12 :
            twoLinePlate = False
        elif regTable[0] == 13 or regTable[0] == 14 :
            twoLinePlate = True
            
        for row in objTable :
            if row[0] < 12 : # region이 아니면..
                arr = np.concatenate([arr,row],axis=0)
    
        arr = np.concatenate([arr,regTable],axis=0)
        objTable = arr.reshape(-1,col_size)
        

    
    return objTable, twoLinePlate
            
#Object Detection API에서의  번호/문자 인식 내용을 추출 한다.    
def predictPlateNumberODAPI(detect, platetype_index, category_index, CLASS_DIC, twoLinePlate) :
    
    objTable = []
    
    hReg = False
    oReg = False
    vReg = False
    uChar = False
    
    upbox_avr = 0
    lobox_avr = 0
    
    num_detections = detect['num_detections']
    plate2line = False
    plateTable = []
    
    for i in range(0,num_detections) :
        box = detect['detection_boxes'][i]
        class_id = detect['detection_classes'][i] + 1
        score = detect['detection_scores'][i]
        item = [class_id, score, box[0],box[1],box[2],box[3]]
        objTable.append(item)
    
    objTable = np.array(objTable)
    
    print('번호판 글자 검지갯수 {}'.format(num_detections))

    plate_str = "" # 번호판 문자
    if(num_detections > 1):
        
        #용도 문자가 중복되는지 확인
        objTable = classIdDoubleCheck(class_id=11,objTable=objTable)
        #지역 문자가 중복되는지 확인
        objTable = classIdDoubleCheck(class_id=12,objTable=objTable)
        objTable = classIdDoubleCheck(class_id=13,objTable=objTable)
        objTable = classIdDoubleCheck(class_id=14,objTable=objTable)
        #오직 한개의 region만 존재하도록 한다.
        objTable, twoLinePlate = onlyOneRegion(objTable,twoLinePlate)
        
        if len(objTable) > 1 :
        
            plate2line = False
            # 번호판 상하단 구분 위한 코드
            #ref = objTable[:,2].mean(axis = 0)
            
            #y 높이 순으로 정렬
            v_order_arr = objTable[objTable[:,2].argsort()]
            # y 갋만 뽑음
            ycol1 = v_order_arr[:,2]
            # 한개 차이로 
            ycol2 = ycol1[1:]
            ycol2 = np.append(ycol2,ycol2[-1])
            result = ycol2 - ycol1
            ref = result.argmax()
            
            type = platetype_index
            # if type in twolinePlate or twoLinePlate :
            #     plate2line = True
            #     print("2line")
            # else:
            #     print("1line")
            
            box_height = v_order_arr[:,4] - v_order_arr[:,2]  # box 놀이를 구한다.
    
            if ref >= 0 and ref < len(result) - 1 :
                upbox_avr =  Average(box_height[:ref+1])
                lobox_avr =  Average(box_height[ref+1 :])
                if result[ref] > upbox_avr/2:
                    plate2line = True
                    print("2line")
                
            else:
                plate2line = False
                print("1line")
            
            if plate2line :
                # 2line 번호판이면...
                # 1line 과 2line으로 나눈다.
                onelineTable = []
                twolineTalbe = []
                
                for index ,type in enumerate(v_order_arr):
                    if index <= ref :
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
                        if cindex != 0 and  platetype_index != 9:
                            twolineTalbe = checkTwoNumAhead(rindex=cindex, objTable=twolineTalbe)
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
                        #if cindex != 0 and  platetype_index != 9:
                        #    plateTable = checkTwoNumAhead(rindex=cindex, objTable=plateTable)
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
            plateTable = rmOverlapBoxs(plateTable=plateTable)
            num_detections = plateTable.shape[0] #갯수가 바뀔수 있다.
            for i in range(0,num_detections) :
                class_index = int(plateTable[i][0])
                label = category_index[class_index]['name']
                plate_str = plate_str + CLASS_DIC[label]
                if class_index == 11:
                    uChar = True
                elif class_index == 12:
                    vReg = True
                elif class_index == 13:
                    hReg = True
                elif class_index == 14:
                    oReg = True
                    
        else:
            print('에러 번호판 아님')
    else:   # 예외처리
        plateTable = objTable
    
           
    #번호판 타입을 결정한다.
    if not plate2line :
        if vReg == True:
            platetype_index = 5 # type5와 type10이 있지만, type5로 일단한다.
        elif len(plate_str) == 8 and uChar == True:
            platetype_index = 9 # 번호판 글자가 8자이고 용도문자가 있으면 type9(3자리 번호)로 한다.
        elif len(plate_str) == 7 and uChar == True:
            if platetype_index != 3:  # 타입이 3이 아니면 8로 한다.
                platetype_index = 8         #7자리 이면 type8로 정한다.
        else:
            platetype_index = 3
    else :
        # 2자리 번호판이면.
        if oReg == True:        # 영 번호판이면.
            platetype_index = 6
        elif hReg == True:
            platetype_index = 1 # type1과 type2가 있지만 일단 type1로 설정
        elif lobox_avr > upbox_avr*1.5 : #윗쪽 문자 평균 놑이*1.5배 보다 아랫쪽 문자 높이가 크면
            platetype_index = 4
        else:
            platetype_index = 7  #그 외에는 type7번으로 한다.
        
    print("번호판 {} 번호 인식: {}".format('2단' if plate2line == True else '1단',plate_str))
    print('plateTable {}'.format(plateTable))
    return plate_str , plateTable ,  plate2line,  platetype_index


def moto_predictPlateNumberODAPI(detect, plate_class_id , category_index, CLASS_DIC,LABEL_FILE_CLASS,twoLinePlate) :
    
    objTable = []
    
    hReg = False
    oReg = False
    vReg = False
    uChar = False
    
    upbox_avr = 0
    lobox_avr = 0
    
    num_detections = detect['num_detections']
    plate2line = False
    plateTable = []
    
    for i in range(0,num_detections) :
        box = detect['detection_boxes'][i]
        class_id = detect['detection_classes'][i] + 1
        score = detect['detection_scores'][i]
        ch_class_id = plate_class_id[i]
        item = [class_id, score, box[0],box[1],box[2],box[3], ch_class_id]
        objTable.append(item)
    
    objTable = np.array(objTable)
    
    print('번호판 글자 검지갯수 {}'.format(num_detections))

    plate_str = "" # 번호판 문자
    if(num_detections > 1):
        
        objTable = classIdDoubleCheck(class_id=13,objTable=objTable)
        #오직 한개의 region만 존재하도록 한다.
        objTable, twoLinePlate = onlyOneRegion(objTable,twoLinePlate)
        
        plate2line = False
        # 번호판 상하단 구분 위한 코드
        #ref = objTable[:,2].mean(axis = 0)
        
        #y 높이 순으로 정렬
        v_order_arr = objTable[objTable[:,2].argsort()]
        # y 값만 뽑음
        ycol1 = v_order_arr[:,2]
        # 한개 차이로 
        ycol2 = ycol1[1:]
        ycol2 = np.append(ycol2,ycol2[-1])
        result = ycol2 - ycol1
        ref = result.argmax()
        
        box_height = v_order_arr[:,4] - v_order_arr[:,2]  # box 놀이를 구한다.

        if ref >= 0 and ref < len(result) - 1 :
            upbox_avr =  Average(box_height[:ref+1])
            lobox_avr =  Average(box_height[ref+1 :])
            if result[ref] > upbox_avr/2:
                plate2line = True
                print("2line")
            
        else:
            plate2line = False
            print("1line")
        
        if plate2line :
            # 2line 번호판이면...
            # 1line 과 2line으로 나눈다.
            onelineTable = []
            twolineTalbe = []
            
            for index ,type in enumerate(v_order_arr):
                if index <= ref :
                    onelineTable.append(list(type))
                else:
                    twolineTalbe.append(list(type))
            onelineTable = np.array(onelineTable)
            twolineTalbe = np.array(twolineTalbe)
            if onelineTable.size :
                onelineTable = onelineTable[onelineTable[:,-2].argsort()] #onelineTable[:,3].argsort() 순서대로 인덱스를 반환
                if onelineTable[0,0] == 13:  # hReg 첫글자 가로 지역문자이면...
                    res = onelineTable[1:,:]
                    if res.shape[0] > 3:
                        res = res[(-res[:,1]).argsort()[:2]] #스코어 순으로 2개만 추린다.
                        #다시 정렬한다.
                        res = res[res[:,-1].argsort()]
                        arr = np.array([onelineTable[0]])
                        arr = np.concatenate([arr,res],axis=0)
                        onelineTable = arr
                        
            if twolineTalbe.size :
                twolineTalbe = twolineTalbe[twolineTalbe[:,-2].argsort()]
                twolineTalbescore = twolineTalbe[:,0]
                result = np.where(twolineTalbescore == 11)
                #용도문자 이후 오른쪽 숫자가 4개 이상이면 스코어에 따라서 삭제한다.
                if len(result) and len(result[0]) > 0 :  # Char 첫글자 가로 지역문자이면...
                    cindex = result[0][0]
                    res = twolineTalbe[cindex + 1:,:]
                    if res.shape[0] > 4:
                        res = res[(-res[:,1]).argsort()[:4]] #스코어 순으로 4개만 추린다.
                        #다시 정렬한다.
                        res = res[res[:,-1].argsort()]
                        arr = twolineTalbe[0 : cindex + 1]
                        arr = np.concatenate([arr,res],axis=0)
                        twolineTalbe = arr
                        twolineTalbe = checkTwoNumAhead(rindex=cindex, objTable=twolineTalbe)
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

        #print("plateTable : {0}".format(plateTable))
        plateTable = rmOverlapBoxs(plateTable=plateTable)
        num_detections = plateTable.shape[0] #갯수가 바뀔수 있다.
        
        for i in range(0,num_detections) :
            plateTable[i][-1] = int(plateTable[i][-1])
            plate_str = plate_str + CLASS_DIC[LABEL_FILE_CLASS[int(plateTable[i][-1])]]

    else:   # 예외처리
        plateTable = objTable
    
        
    print("번호판 {} 번호 인식: {}".format('2단' if plate2line == True else '1단',plate_str))
    print('plateTable {}'.format(plateTable))
    return plate_str , plateTable 

# plateTabe에서 겹치는 부분은 삭제한다.
def rmOverlapBoxs(plateTable):
    #숫자가 있을 때 다른 문자 안에 포함되면 삭제한다.
    boxes = plateTable[:,2:2+4]
    scores = plateTable[:,1]
    #x,y를 바꾼다.
    #boxes[:,[0,1]] = boxes[:,[1,0]] 
    #boxes[:,[2,3]] = boxes[:,[3,2]]
   
    new_plateTable = np.zeros((1,plateTable.shape[1]))
    skip = []
    for i in range(0,len(boxes) - 1):
        if i in skip:
            continue
        box1 = boxes[i]
        box2 = boxes[i+1]
        iou,box1_area, box2_area,inter = IoU(box1,box2)
        if iou < IOU_THESHOLD:
           new_plateTable =  np.append(new_plateTable,np.expand_dims(plateTable[i],axis=0),axis=0)
        elif iou >= IOU_THESHOLD:
            if scores[i] > scores[i+1] :
                new_plateTable =  np.append(new_plateTable,np.expand_dims(plateTable[i],axis=0),axis=0)
            else : 
                new_plateTable =  np.append(new_plateTable,np.expand_dims(plateTable[i+1],axis=0),axis=0)
            skip.append(i+1)

    lastlow = len(boxes) - 1
    #마지막 줄은 항상 빠지므로 추가 해 준다.
    if not lastlow in skip :
        new_plateTable =  np.append(new_plateTable,np.expand_dims(plateTable[lastlow],axis=0),axis=0)
    new_plateTable = np.delete(new_plateTable, 0 , axis = 0)
    return new_plateTable
    
            
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
            
def equalizeHist(src) :
    # src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
    # ycrcb_planes = np.asarray(cv2.split(src_ycrcb))
    
    
    # # 밝기 성분에 대해서만 히스토그램 평활화 수행
    # ycrcb_planes[0] = cv2.equalizeHist(ycrcb_planes[0])
    
    # dst_ycrcb = cv2.merge(ycrcb_planes)
    # dst = cv2.cvtColor(dst_ycrcb, cv2.COLOR_YCrCb2BGR)
    img_yuv = cv2.cvtColor(src, cv2.COLOR_BGR2YUV)
    img_clahe = img_yuv.copy()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) #CLAHE 생성
    img_clahe[:,:,0] = clahe.apply(img_clahe[:,:,0])           #CLAHE 적용
    img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_YUV2BGR)
    
    return img_clahe


def rgb2gray(src_img):
    
    src_img[:,:,0] = (src_img[:,:,0]*0.2126 + src_img[:,:,1]* 0.7152 + src_img[:,:,2]* 0.0722)

    return src_img
# 리스트의 평균을 구하는 함수이다.
def Average(lst):
    return sum(lst) / len(lst) 

# json 파일내용에서 번호판 내용을 읽어오는 함수이다.
# json_data : json data 
# enlable : english label
# human_dic : english label to 한글 레이블 변환 딕셔너리
def GetPlateNameFromJson(json_data , enlabel, human_dic ) :
    
    class_label = enlabel[1:]
    num_char_lable = enlabel[11:]
    plate_label = enlabel[1:11]
    json_data['imageData'] = None  
    #object의 시작 위치이다.
    box_sx = None
    box_sy = None
    #object 종료 위치이다.
    box_ex = None
    box_ey = None
    #object 넓이 높이 이다.
    box_width = None
    box_height = None
    
    crop_polygon = None
    
    find_object = False
    obj_name_ext = None
        
    image_width = int (json_data['imageWidth'])
    image_height = int (json_data['imageHeight'])  

    objTable = []
    num_detections = 0 
    platetype_index = 0

    for item, shape in enumerate(json_data['shapes']):
        label = shape['label']
        
        if label in class_label:

            if label in plate_label :
                platetype_index = plate_label.index(label) + 1
            elif label in num_char_lable:

                obj_name_ext = human_dic[label]
                points = np.array(shape['points']).astype(int) # numpy로 변형
                shape_type = shape['shape_type']
                
                # rectangle 형태이면 폴리곤 타입으로 바꾸어 준다.
                tpoints = []
                if shape_type == 'rectangle':
                    tpoints = box2polygon(points) #test point를 polygon으로 만든다.
                else:
                    tpoints = points
                    
                #줄이기 전에 잘라낼 위치를 정한다.
                box_xs = points[:,0]
                box_ys = points[:,1]

                box_sx = np.min(box_xs,axis=0)
                if box_sx < 0:
                    box_sx = 0
                box_sy = np.min(box_ys,axis=0)
                if box_sy < 0:
                    box_sy = 0
                box_ex = np.max(box_xs,axis=0) 
                if box_ex >= image_width:
                    box_ex = image_width - 1
                box_ey = np.max(box_ys,axis=0)
                if box_ey >= image_height:
                    cropey = image_height - 1
                box_width =  box_ex - box_sx
                box_height = box_ey - box_sy
                
                box = [box_sy, box_sx, box_ey, box_ex]
                
                class_id = class_label.index(label)

                item = [class_id, 100, box[0],box[1],box[2],box[3]]
                num_detections += 1
                objTable.append(item)
        
    objTable = np.array(objTable)
    
    plate_str = "" # 번호판 문자
    if(num_detections > 1):
        plate2line = False
        # 번호판 상하단 구분 위한 코드
        #y 높이 순으로 정렬
        v_order_arr = objTable[objTable[:,2].argsort()]
        # y 갋만 뽑음
        ycol1 = v_order_arr[:,2]
        # 한개 차이로 
        ycol2 = ycol1[1:]
        ycol2 = np.append(ycol2,ycol2[-1])
        result = ycol2 - ycol1
        ref = result.argmax()
        
        box_height = v_order_arr[:,4] - v_order_arr[:,2]  # box 놀이를 구한다.

        if ref >= 0 and ref < len(result) - 1 :
            upbox_avr =  Average(box_height[:ref+1])
            if result[ref] > upbox_avr/2:
                plate2line = True
                #print("2line")
            
        else:
            plate2line = False
            #print("1line")

        plateTable = []
        if plate2line :
            # 2line 번호판이면...
            # 1line 과 2line으로 나눈다.
            onelineTable = []
            twolineTalbe = []
            
            for ix, type in enumerate(v_order_arr):
                if ix <= ref :
                    onelineTable.append(list(type))
                else:
                    twolineTalbe.append(list(type))
            onelineTable = np.array(onelineTable)
            twolineTalbe = np.array(twolineTalbe)
            if onelineTable.size :
                onelineTable = onelineTable[onelineTable[:,-1].argsort()] #onelineTable[:,3].argsort() 순서대로 인덱스를 반환
            if twolineTalbe.size :
                twolineTalbe = twolineTalbe[twolineTalbe[:,-1].argsort()]
            if onelineTable.size and twolineTalbe.size:
                plateTable = np.append(onelineTable,twolineTalbe, axis=0)
            elif onelineTable.size:
                plateTable =  onelineTable
            elif twolineTalbe.size:
                plateTable =  twolineTalbe

        else:
                onelineTable = objTable
                plateTable = onelineTable[onelineTable[:,-1].argsort()]    
        #print("plateTable : {0}".format(plateTable))
    
        for i in range(0,num_detections) :
            class_index = int(plateTable[i][0])
            name = class_label[class_index]
            plate_str = plate_str + human_dic[name]
    
        #print("SSD 인식 내용 {0}".format(plate_str)) 
        
        return plate_str 
    
def extract_sub_image(src_np, box, width, height, pad=False):
    src_height, src_width, ch = src_np.shape
    box_sy = int(src_height*box[0])
    box_sx= int(src_width*box[1])
    box_ey = int(src_height*box[2])
    box_ex= int(src_width*box[3])
    obj_img = src_np[box_sy:box_ey,box_sx:box_ex,:]
    
    #번호판을 320x320 크기로 정규화 한다.
    if pad :
        desired_size = max(height,width)
        old_size = [obj_img.shape[1],obj_img.shape[0]]
        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        #원영상에서 ratio 만큼 곱하여 리싸이즈한 번호판 영상을 얻는다.
        cropped_img = cv2.resize(obj_img,new_size,interpolation=cv2.INTER_LINEAR)
        dst_np = np.zeros((desired_size, desired_size, 3), dtype = "uint8")
        #dst_np = cv2.cvtColor(dst_np, cv2.COLOR_BGR2RGB)
        h = new_size[1]
        w = new_size[0]
        yoff = round((desired_size-h)/2)
        xoff = round((desired_size-w)/2)
        #320x320영상에 번호판을 붙여 넣는다.
        dst_np[yoff:yoff+h, xoff:xoff+w , :] = cropped_img        
    else :
        desired_size = (height,width)
        #원영상에서 ratio 만큼 곱하여 리싸이즈한 번호판 영상을 얻는다.
        dst_np = cv2.resize(obj_img,desired_size,interpolation=cv2.INTER_LINEAR)
        # plt.imshow(dst_np)
        # plt.show()

    return dst_np

#json 파일을 만든다.
# src_path 이미지의 디렉토리
# image_filename 이미지의 파일이름
# 저장 디렉토리 dst_path
# 이미지 shaep image_shape height, width channel
# CLASS_DIC label 과 사람이 인식하는 문자의 딕셔너리
# plateTable  plate table class번호, 확률, y, x, y ,x
# plateNumber 인식한 번호
# platebox 번호판 실제 좌표 정규화 아님.
# plateIndex 번호판 type  type1 ~ type9
# plate_shape 번호판 리싸이즈 한 크기 320x320
# xratio 원래 번호판에서 320x320으로 변환 하였을때 ratio
# add_platenum 인식한 번호판 내용을 붙일지 여부
def makeJson(src_path, image_filename,dst_path, image_shape,category_index, CLASS_DIC,plateTable, plateNumber,platebox,plateIndex,plate_shape,xratio,add_platenum = True) :
    #platetable의 첫번째 숫자는
    # 1 ~ 10 1, 2, 3 ... 0
    # 11 Char
    # 12 vReg
    # 13 hReg
    # 14 oReg
    commercial = False #영 포함 여부
    json_data = OrderedDict()
    json_data['version'] = '5.0.1'
    json_data['flags'] = {}
    
    shapes=[]
    
    json_data['shapes'] = shapes
    
    json_data['imageData'] = None
    json_data['imageHeight'] = image_shape[0]
    json_data['imageWidth'] = image_shape[1]

    num_detections = plateTable.shape[0]
    
    for ix in range(0,num_detections) :
        class_index = int(plateTable[ix][0])
        label = category_index[class_index]['name']
        str =  CLASS_DIC[label]
        if not str == 'x' :  # x가 나오면 인식한게 아니기 때문.
            if class_index == 11 :  #용도문자
                json_data['usage'] = str
                json_data['type'] = str # 타입숫자 ?
            if class_index >= 12 :  #지역문자
                json_data['region'] = str
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
            old_y = platebox[1][2] - platebox[1][0]  #원래 번호판 높이
            plate_real_height = old_y * xratio #320으로 변형 했을때 확대된 번호판 높이
            half_dummy_height = (plate_shape[1] - plate_real_height)/2  #상단 더미 높이
            half_dummy_ratio = half_dummy_height / plate_shape[0]
            points_x = [points_x[i]*plate_shape[1]/xratio + platebox[0][0]  for i in range(len(points_x))] #320 기준으로 좌표를 변환하고, 다시 전체영상 기준으로 바꾼다.
            points_y = [(points_y[i] - half_dummy_ratio)*plate_shape[0]/xratio + platebox[1][0]  for i in range(len(points_y))]
            insertlabel_with_xypoints(shapes,points_x,points_y,label=label)
        
    # 번호판 타입을 추가한다.
    label = 'type{}'.format(plateIndex)
    insertlabel_with_xypoints(shapes,platebox[0],platebox[1],label=label)
 
    json_data['number'] = plateNumber
    
    if commercial:
        json_data['commercial'] = 'True'
    else :
        json_data['commercial'] = None
        
   
    
    basefilename, ext = os.path.splitext(image_filename)
    
    if basefilename[-1] == 'c':
        basefilename =  basefilename[:-1]
    
    if add_platenum :
        basefilename = basefilename + '_' + plateNumber
        json_data['imagePath'] = basefilename + ext
    else:
        json_data['imagePath'] = image_filename
        
    ofilename = os.path.join(dst_path,basefilename)
    
    # json 파일을 저장한다.
    with open( ofilename +'.json','w', encoding='utf-8') as f:
            json.dump(json_data,f,ensure_ascii=False,indent="\t" , cls=NpEncoder)
            
    src_file = os.path.join(src_path,image_filename)
    ofilename = ofilename + ext
    dst_file = os.path.join(dst_path,ofilename)
    #영상 파일 복사한다.
    if os.path.isfile(src_file) :
        shutil.copyfile(src_file, dst_file)
        
#2자리 지역 dictionary key 값이 있는지 확인하다.        
# dic dictionary 이름
# kval  조회하려는 key 값
def checkKeyinRegionDictionary( dic, kval) :

    keyFind  = False

    if kval in dic.keys():
        keyFind = True
    else :
        #['서', '인', '부', '대', '광', '대', '세', '경', '강', '충', '충', '전', '전', '경', '경', '제', '울', 'x']
        # -1을 하는 것은 x를 빼기 위함이다.
        keylist = [k for k, v in dic.items()]
        
        keyslist1 = [ key[0] for key in keylist[:-1]]
        keyslist2 = [ key[1] for key in keylist[:-1]]

        #첫번째에서 찾는다.
        if kval[0] in keyslist1:
            ix = keyslist1.index(kval[0])
            keyFind = True
            kval = keylist[ix]
        else:
            
            if len(kval) > 1:
                if kval[1] in keyslist2:
                    ix = keyslist2.index(kval[1])
                    kval = keylist[ix]
                    keyFind = True

    return keyFind, kval

# box1 이 box2 안에 있는지 여부를 첵크함.
def isInside(box1, box2):
    #box1 = (y1, x1, y2, x2)
    #box2 = (y1, x1, y2, x2)
  
  # First we make sure we compare things in the right order
  # You can skip that part if you are sure that in all cases x1 < x2 and y1 < y2
  b1_xmin = min(box1[1], box1[3])
  b1_xmax = max(box1[1], box1[3])
  b1_ymin = min(box1[0], box1[2])
  b1_ymax = max(box1[0], box1[2])
  
  b2_xmin = min(box2[1], box2[3])
  b2_xmax = max(box2[1], box2[3])
  b2_ymin = min(box2[0], box2[2])
  b2_ymax = max(box2[0], box2[2])
  
  # Then you perform your checks. From what I understood,
  # you want the result to be true if any corner of the box1
  # is inside the box2's bounding box.
  
  b1_corners = [
    (b1_xmin, b1_ymin),
    (b1_xmin, b1_ymax),
    (b1_xmax, b1_ymin),
    (b1_xmax, b1_ymax)]

  status = True
  for corner in b1_corners:
    in_range_along_x = corner[0] < b2_xmax and corner[0] > b2_xmin
    in_range_along_y = corner[1] < b2_ymax and corner[1] > b2_ymin
    subcheck = in_range_along_x and in_range_along_y
    if not subcheck:
        status = False
        break
    else :
        status = status or subcheck
  
  # If we get there, then the box1 is not inside that box2
  return status
# box1 이 box2 아래에 있는지 여부를 첵크함.
def isUnderBox( box1, box2):
    #[[box_sx, box_ex, box_ex, box_sx],[box_sy,box_sy,box_ey,box_ey]]
    status = False
    
    box2_mypos = (box2[1][0] + box2[1][2])/2
    box1_mypos = (box1[1][0] + box1[1][2])/2
    box1_mxpos = (box1[0][0] + box1[0][1])/2
    
    box2_sx = box2[0][0]
    box2_ex = box2[0][1]
    
    if box1_mypos > box2_mypos and ( box1_mxpos >= box2_sx and box1_mxpos <= box2_ex ) :
        status = True
        
    return status