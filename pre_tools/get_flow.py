import zipfile
import glob
import cv2
import numpy as np
import os

all_path = glob.glob('/Users/shiwakaga/output_flow/*')

for video in all_path:
    flow_x_zip = video + '/flow_x.zip'
    flow_y_zip = video + '/flow_y.zip'
    with zipfile.ZipFile(flow_x_zip, 'r') as z:
        file_name = flow_x_zip.split('.')[0]
        if not os.path.exists(file_name):
            os.mkdir(file_name)
            z.extractall(file_name)
    with zipfile.ZipFile(flow_y_zip, 'r') as z_y:
        file_name = flow_y_zip.split('.')[0]
        if not os.path.exists(file_name):
            os.mkdir(file_name)
            z_y.extractall(file_name)
    x_flow = np.zeros((540,960,3)).astype(np.uint8)
    y_flow = np.zeros((540, 960,3)).astype(np.uint8)
    for i in range(244,249):
        x_img = cv2.imread(flow_x_zip[:-4]+'/'+'x_00'+str(i)+'.jpg')
        y_img = cv2.imread(flow_x_zip[:-4] + '/' + 'y_00' + str(i) + '.jpg')
        print(x_img.shape)
        x_flow += x_img
        y_flow += y_img


