import cv2
import glob
import numpy as np

def split(pres,img):
    """
    pres: a list of pres
    img: orignal pic
    """
    for pre in pres:
        img_ins_id = img.split('/')[-3]
        img_name = pre.split('/')[-1]

        im = cv2.imread(img)
        mask = cv2.imread(pre)
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                if any(mask[i][j] == [0,0,0]):
                    im[i][j] = [0,0,0]
        cv2.imwrite(out_path + 'pre_split/' + img_ins_id + '/' + img_name, im)


def merge(pres,img):
    """
    pres: a list of pres
    img: orignal pic
    """
    img_ins_id = img.split('/')[-3]
    img_name = img.split('/')[-1]
    im = cv2.imread(img)
    pre = np.zeros(im.shape)
    for p in pres:
        mask = cv2.imread(p)
        pre += mask
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if any(pre[i][j] == [0, 0, 0]):
                im[i][j] = [0, 0, 0]
    cv2.imwrite(out_path + 'pre_merge' + img_ins_id + '/' + img_name, im)


if __name__ == '__main__':

    train_list = glob.glob('/data/srh/Amodel_Data/train/*/images/*.jpg')
    test_list = glob.glob('/data/srh/Amodel_Data/test/*/images/*.png')
    data_path = '/data/srh/Amodel_Data/'
    out_path = '/home/srh/Mask_RCNN/vis_ins_only/'

    modes = ['test','train']

    for mode  in modes:
        pre_path = '/home/srh/Mask_RCNN/output_mrcnn/'
        if mode == 'train':
            list = train_list
        else:
            list = test_list
        for img in list:
            print(img)
            img_id = img.split('/')[-1][:-4]
            img_ins_id = img.split('/')[-3]
            pre_list = glob.glob(pre_path + img_ins_id + '/' + img_id + '*.png')
            gt_list = glob.glob(data_path + str(mode) + '/' + img_ins_id + '/amodel/' + img_id + '*.png')

            split(pre_list,img)
            # split(gt_list,img)

            merge(pre_list,img)
            # merge(gt_list, img)






