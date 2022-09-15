import os,sys
import numpy as np
import shutil


# 파일의 이름을 바꾸는 기능을 한다.
# 파일에 중간에 c 가 붙어 있을때 거기까지만 파일이름으로 바꾸는 기능이다.

#========================
# 여기의 내용을 용도에 맞게 수정한다.
dataset_category='plate'
test_dir_name = 'test'
#========================
ROOT_DIR = os.getcwd()
WORKSPACE_PATH = os.path.join(ROOT_DIR,'Tensorflow','workspace')
IMAGE_PATH =  os.path.join(WORKSPACE_PATH,'images',dataset_category)
#테스트할 이미지 디렉토리
images_dir = os.path.join(IMAGE_PATH,test_dir_name)

file_count = len(os.listdir(images_dir))

print('처리할 총 파일 갯수는 {} 입니다.'.format(file_count))

count = 0
for filename in os.listdir(images_dir):
    basefilename, ext = os.path.splitext(filename)
    remove_str = '_' + filename.split('_')[-1]
       
    newfilename = filename.rstrip(remove_str)
    newfilename=  newfilename + ext
    old_file = os.path.join(images_dir,filename)
    new_file = os.path.join(images_dir,newfilename)
    os.rename(old_file, new_file)
    count += 1
        
print('총 {}개를 바꾸었습니다'.format(count))
    
        