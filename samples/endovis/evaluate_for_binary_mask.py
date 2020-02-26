import cv2
import glob
import numpy as np
from evaluate import compute_iou

train_list = glob.glob('/Users/shiwakaga/Amodel_Data/train/*/images/*.jpg')
pre_path = '/Users/shiwakaga/OUT/output/output_mrcnn/'

for img in train_list:
    img_id = str(img.split('/')[-1][:-4])
    img_ins_id = str(img.split('/')[-3])
    pre_list = glob.glob(pre_path + img_ins_id + '/' + img_id + '*.png')
    gt = '/Users/shiwakaga/Amodel_Data/train/{}/instruments_masks/{}.png'.format(img_ins_id,img_id)

    gt_mask = cv2.cvtColor(cv2.imread(gt), cv2.COLOR_RGB2GRAY)

    pre_mask = np.zeros((1024, 1280)).astype(np.uint8)
    for i in range(len(pre_list)):
        mask_i = cv2.cvtColor(cv2.imread(pre_list[i]), cv2.COLOR_RGB2GRAY)/255
        pre_mask = pre_mask + mask_i

    pre_mask[pre_mask > 0] = 1
    gt_mask[gt_mask  > 0 ] = 1

    iou = compute_iou(pre_mask,gt_mask)
    with open('binary_iou.txt','a') as f:
        f.writelines(img_ins_id+' '+img_id+' '+str(iou)+'\n')
