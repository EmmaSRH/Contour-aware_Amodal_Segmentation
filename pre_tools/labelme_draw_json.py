#!/usr/bin/env python

import json
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import uuid

import numpy as np
import PIL.Image
import PIL.ImageDraw

from labelme import utils
#
# def Draw_json_to_png(lbl,num_ins,out_path,frame_name):
#
#      for i in range(num_ins):
#          mask = np.zeros(lbl.shape)
#          mask[lbl==(i+1)] = 255
#
#          cv2.imwrite(out_path+frame_name+'_ins_'+str(i)+'.png',mask)

def shapes_to_label(img_shape, shape, label_name_to_value):
    cls = np.zeros(img_shape[:2], dtype=np.int32)
    ins = np.zeros_like(cls)
    instances = []

    points = shape['points']
    label = shape['label']
    group_id = shape.get('group_id')
    if group_id is None:
        group_id = uuid.uuid1()
    shape_type = shape.get('shape_type', None)

    cls_name = label
    instance = (cls_name, group_id)

    if instance not in instances:
        instances.append(instance)
    ins_id = instances.index(instance) + 1
    cls_id = label_name_to_value[cls_name]

    mask = utils.shape_to_mask(img_shape[:2], points, shape_type)
    cls[mask] = cls_id
    ins[mask] = ins_id

    return cls, ins

def labelme_shapes_to_label(img_shape, shapes):

    label_name_to_value = {'_background_': 0,'ins':255,'ins1':255,'ins2':255,'ins3':255}
    lbl = []
    # print(len(shapes))
    for shape in shapes:
        label_name = shape['label']
        # print(label_name)
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value

        lbl_i, _ = shapes_to_label(img_shape, shape, label_name_to_value)
        lbl.append(lbl_i)
    return lbl, label_name_to_value

def json2label(json_file,out_path):

    data = json.load(open(json_file))  # 加载json文件

    img = utils.img_b64_to_arr(data['imageData'])  # 解析原图片数据

    lbl, lbl_names = labelme_shapes_to_label(img.shape, data['shapes'])
    # print(len(lbl))

    # print(lbl[0].shape)
    # 解析'shapes'中的字段信息，解析出每个对象的mask与对应的label   lbl存储 mask，lbl_names 存储对应的label
    # lal 像素取值 0、1、2 其中0对应背景，1对应第一个对象，2对应第二个对象
    # 使用该方法取出每个对象的mask mask=[] mask.append((lbl==1).astype(np.uint8)) # 解析出像素值为1的对象，对应第一个对象 mask 为0、1组成的（0为背景，1为对象）
    # lbl_names  ['background','cat_1','cat_2']
    frame_name = json_file.split('/')[-1][:-5]
    i = 0
    for mask in lbl:
        print(out_path + frame_name + '_ins_'+ str(i) + '.png')
        cv2.imwrite(out_path + frame_name + '_ins_'+ str(i) + '.png', mask)
        # cv2.imwrite(out_path + frame_name + '.png', mask*255)
        i += 1

if __name__ == '__main__':

    ins_path = '/Users/shiwakaga/Downloads/ins9/'
    name_list = glob.glob(ins_path+'*.json')
    i = 0
    for json_file in name_list:
        out_path = '/Users/shiwakaga/Amodel_Data/test/instrument9/amodel/'
        i += 1
        print("第%d个json文件，名字是：" % i, json_file)
        json2label(json_file,out_path)

    # json2label('/Users/shiwakaga/Amodel_Data/instrument3/frame010.json', '/Users/shiwakaga/Amodel_Data/instrument3/amodel/')
