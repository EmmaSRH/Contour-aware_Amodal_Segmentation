import cv2
import numpy as np
import os, glob

def rgb2masks(label_name,label_dir):
    lbl_id = os.path.split(label_name)[-1].split('.')[0]
    lbl = cv2.imread(label_name, 1)
    h, w = lbl.shape[:2]
    print(lbl_id)
    # for i in range(h):
    #     for j in range(w):
    #         if tuple(lbl[i][j]) != (0, 0, 0):
    #             print(lbl[i][j])
    leaf_dict = {}
    idx = 0
    white_mask = np.ones((h, w, 3), dtype=np.uint8) * 255
    for i in range(h):
        for j in range(w):
            if tuple(lbl[i][j]) in leaf_dict or tuple(lbl[i][j]) == (0, 0, 0):
                continue
            leaf_dict[tuple(lbl[i][j])] = idx
            print(leaf_dict)
            mask = (lbl == lbl[i][j]).all(-1)
            # leaf = lbl * mask[..., None]      # colorful leaf with black background
            # np.repeat(mask[...,None],3,axis=2)    # 3D mask
            leaf = np.where(mask[..., None], white_mask, 0)
            mask_name = label_dir  + lbl_id + '_ins_' + str(idx) + '.png'
            cv2.imwrite(mask_name, leaf)
            idx += 1


# label_dir = '/Users/shiwakaga/Amodel_Data/instrument4'
# label_list = glob.glob(os.path.join(label_dir+'/instruments_masks', '*.png'))
# for label_name in label_list:
#     rgb2masks(label_name,label_dir)
rgb2masks('/Users/shiwakaga/Amodel_Data/instrument3/instruments_masks/frame015.png','/Users/shiwakaga/Amodel_Data/instrument3/')