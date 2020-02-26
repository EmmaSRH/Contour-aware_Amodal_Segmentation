import os
import sys
import random
import itertools
import colorsys

import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import glob
import xlrd
from PIL import Image

from evaluate import compute_matches
import cv2



############################################################
#  Visualization
############################################################

def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interpolation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    print(hsv)
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.3):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def display_instances(image, N, masks, class_names,
                      colors=None, captions=None):

    # Generate random colors
    colors = colors

    plt.figure()
    plt.axis('off') # 不加坐标轴
    plt.text(10, 20 + 60, captions,
                color='w', size=9, backgroundcolor="none")

    masked_image = image.astype(np.uint32).copy()

    for i in range(N+N):
        color = colors[i]
        mask = masks[:, :, i]
        masked_image = apply_mask(masked_image, mask, color)

    plt.imshow(masked_image.astype(np.uint8))
    plt.savefig('/Users/shiwakaga/vis_out/{}/{}.png'.format(class_names[0],class_names[1]))



def display_differences(image,pre_list,gt_list,
                        class_names,scores, title="",):
    """Display ground truth and prediction instances on the same image."""
    # Match predictions to ground truth
    N = len(gt_list)
    pre_mask = np.zeros((1024, 1280, len(pre_list))).astype(np.uint8)
    gt_mask = np.zeros((1024, 1280, len(gt_list))).astype(np.uint8)
    for i in range(len(pre_list)):
        mask_i = cv2.cvtColor(cv2.imread(pre_list[i]), cv2.COLOR_RGB2GRAY)
        pre_mask[:, :, i] = mask_i
    for i in range(len(gt_list)):
        mask_i = cv2.cvtColor(cv2.imread(gt_list[i]), cv2.COLOR_RGB2GRAY)
        gt_mask[:, :, i] = mask_i

    gt_match, _, _= compute_matches(pre_mask, gt_mask)

    # Ground truth = green. Predictions = red
    # colors = [(0, 1, 0, .8)] * len(gt_list)\
    #        + [(1, 0, 0, 1)] * len(gt_list)
    colors_all = [[(0, 1, 0, .8)] ,[(0.5, 1, 1.0)],[(0.0, 1, 1.0)],[(0.8, 0.2, 0.4)]]
    colors = []
    for i in range(N):
        colors = colors + colors_all[i]
    colors = colors+ [(1, 0, 0, 1)] * len(gt_list)
        # Concatenate GT and predictions
    match_pre = np.zeros((1024, 1280, N)).astype(np.uint8)
    match_gt = np.zeros((1024, 1280, N)).astype(np.uint8)

    for k in range(N):
        match_pre[:,:,k] = pre_mask[:, :, int(gt_match[k])]/255
        match_gt[:,:,k] = gt_mask[:,:,k]/255

    masks = np.concatenate([match_pre, match_gt], axis=-1)

    # Captions per instance show score/IoU
    # scores = scores*100
    captions = class_names[0]+' '+class_names[1]+'\nmAP: {:.2f} mdice: {:.2f} miou {:.2f}'\
                                       .format(scores[0]*100,scores[1]*100,scores[2]*100)
    # Display
    display_instances(image, N, masks, class_names,colors=colors,captions=captions)



if __name__ == '__main__':

    # train_list = glob.glob('/Users/shiwakaga/Amodel_Data/train/*/images/*.jpg')
    # test_list = glob.glob('/Users/shiwakaga/Amodel_Data/test/*/images/*.png')
    # modes = ['test', 'train']

    # for test
    train_list = glob.glob('/Users/shiwakaga/Amodel_Data/train/instrument5/images/frame223.jpg')
    modes = ['train']

    for mode in modes:
        pre_path = '/Users/shiwakaga/OUT/output/output_mrcnn/'

        if mode == 'train':
            list = train_list
        else:
            list = test_list

        for img in list:
            print(img)
            image = cv2.imread(img)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_id = str(img.split('/')[-1][:-4])
            img_ins_id = str(img.split('/')[-3])
            pre_list = glob.glob(pre_path + img_ins_id + '/' + img_id + '*.png')
            gt_list = glob.glob(
                '/Users/shiwakaga/Amodel_Data/' + str(mode) + '/' + img_ins_id + '/amodel/' + img_id + '*.png')


            num_in_excel = 0
            if mode == 'train':
                num_in_excel = (int(img_ins_id[10:])-1)*225 + int(img_id[5:])
            else:
                if img_ins_id=='instrument10':
                    num_in_excel = (int(img_ins_id[10:])-1) * 75 + int(img_id[5:])
                else:
                    num_in_excel = 8 * 75 + 300 + int(img_id[5:])
            excel_path = '/Users/shiwakaga/OUT/result/mrcnn/vis_{}.xlsx'.format(mode)
            workbook = xlrd.open_workbook(filename=excel_path )
            sheet1 = workbook.sheets()[0]
            scores = sheet1.row_values(num_in_excel)[2:]

            display_differences(rgb, pre_list, gt_list, class_names=[img_ins_id,img_id],scores=scores)

