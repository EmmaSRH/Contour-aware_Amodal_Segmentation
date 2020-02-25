import unittest
import cv2
import numpy as np

from imageio import imread
from imageio import imsave
from evaluation.dice_calculations import compute_dice_coefficient, compute_dice_coefficient_per_instance


class TestDiceCalculation(unittest.TestCase):
    def test_bi_score(self,x_path,mask):
        gt = imread(x_path)
        gt[gt < 0.5] = 0
        gt[gt >= 0.5] = 1
        dice = compute_dice_coefficient(gt, mask)
        print(dice)
        with open('60_dice_sorce.txt', 'a+') as f:
            f.write(x_path.split('/')[-1] + ':' + str(dice) + '\n')
    def test_dice_coefficient(self):
        # paths
        image_train_path = "/Users/shiwakaga/Amodel_Data/train/*/images/*.jpg"
        image_test_path = "/Users/shiwakaga/Amodel_Data/test/*/images/*.png"
        x_path = "/Users/shiwakaga/Desktop/output/"
        y_path = "/Users/shiwakaga/Amodel_Data/*/amodel/"


        # read images
        import glob
        xs = glob.glob(x_path)
        for img in xs:
            x = imread(y_path + img.split('/')[-1].replace('raw','instrument_instances'))
            y = imread(img)

            # make images binary
            x[x < 0.5] = 0
            x[x >= 0.5] = 1
            y[y < 0.5] = 0
            y[y >= 0.5] = 1

            # calculate dice
            # dice = []
            dice = compute_dice_coefficient(x,y)
            print(dice)
            with open('dice_sorce.txt','a+') as f:
                f.write(img.split('/')[-1]+':'+str(dice)+'\n')
            # if dice<0.75 :
            #     import numpy as np
            #     img1 = imread('/home/srh/Training/imgs/'+img.split('/')[-1])
            #     img2 = imread('/home/srh/Mask_RCNN/samples/endovis/results/img/' + img.split('/')[-1])
            #     new_img = np.vstack((img1, img2))
            #     imsave('/home/srh/Mask_RCNN/samples/endovis/bad_results/'+str(dice)+img.split('/')[-1],new_img)

        # # check if correct
        # expected_dice = 0.011
        # delta = 0.0005
        # self.assertAlmostEqual(dice, expected_dice, delta=delta)

    def test_multiple_instance_dice_coefficient(self):
        # paths
        x_path = "images/img{}/instrument_instances.png".format(2)
        y_path = "images/img{}/instrument_instances.png".format(3)

        # read images
        x = imread(x_path)
        y = imread(y_path)

        # calculate instance dice
        instance_dice_scores = compute_dice_coefficient_per_instance(x, y)

        # check if correct
        expected_dice_scores = dict(background=0.8789, instrument_0=0, instrument_1=0.1676)
        delta = 0.0005

        for dice_key, expected_dice_key in zip(instance_dice_scores, expected_dice_scores):
            dice = instance_dice_scores[dice_key]
            expected_dice = expected_dice_scores[expected_dice_key]
            self.assertAlmostEqual(dice, expected_dice, delta=delta)


if __name__ == '__main__':

    test = TestDiceCalculation()
    test.test_dice_coefficient()

    pre = ['/Users/shiwakaga/Desktop/output/instrument1/frame000_ins_0.png',
           '/Users/shiwakaga/Desktop/output/instrument1/frame000_ins_1.png',
           '/Users/shiwakaga/Desktop/output/instrument1/frame000_ins_2.png']
    gt = ['/Users/shiwakaga/Amodel_Data/train/instrument1/amodel/frame000_ins_0.png',
          '/Users/shiwakaga/Amodel_Data/train/instrument1/amodel/frame000_ins_1.png',
          '/Users/shiwakaga/Amodel_Data/train/instrument1/amodel/frame000_ins_0.png']

    pre_mask = np.zeros((1024, 1280, len(pre))).astype(np.uint8)
    gt_mask = np.zeros((1024, 1280, len(gt))).astype(np.uint8)
    for i in range(len(pre)):
        mask_i = cv2.cvtColor(cv2.imread(pre[i]), cv2.COLOR_RGB2GRAY)
        pre_mask[:, :, i] = mask_i
    for i in range(len(gt)):
        mask_i = cv2.cvtColor(cv2.imread(gt[i]), cv2.COLOR_RGB2GRAY)
        gt_mask[:, :, i] = mask_i
    # y = "/home/srh/Mask_RCNN/samples/endovis/results/binary_result/Prokto_2_84000_raw.png"
    # x = "/home/srh/Training/gts/Prokto_2_84000_instrument_instances.png"
    # x = imread(x)
    # y = imread(y)
    # x[x < 0.5] = 0
    # x[x >= 0.5] = 1
    # y[y < 0.5] = 0
    # y[y >= 0.5] = 1
    # print(x.sum())
    # print(y.sum())
    # print((x & y).sum())
    # dice = compute_dice_coefficient(x,y)
    # print(dice)
