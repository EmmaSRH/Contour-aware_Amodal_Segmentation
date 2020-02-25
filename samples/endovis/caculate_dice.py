import unittest
import cv2
import numpy as np

from imageio import imread
from imageio import imsave

from scipy.optimize import linear_sum_assignment as hungarian_algorithm


def compute_dice_coefficient(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.

    compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
    and the predicted mask `mask_pred`.

    Args:
      mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
      mask_pred: 3-dim Numpy array of type bool. The predicted mask.

    Returns:
      the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if mask_gt.sum() == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum


def compute_dice_coefficient_per_instance(mask_gt, mask_pred):
    """Compute instance soerensen-dice coefficient.

        compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
        and the predicted mask `mask_pred` for multiple instances.

        Args:
          mask_gt: 3-dim Numpy array of type int. The ground truth image, where 0 means background and 1-N is an
                   instrument instance.
          mask_pred: 3-dim Numpy array of type int. The predicted mask, where 0 means background and 1-N is an
                   instrument instance.

        Returns:
          a instance dictionary with the dice coeffcient as float.
        """
    # get number of labels in image
    instances_gt = np.unique(mask_gt)
    instances_pred = np.unique(mask_pred)

    # create performance matrix
    performance_matrix = np.zeros((len(instances_gt), len(instances_pred)))
    masks = []

    # calculate dice score for each ground truth to predicted instance
    for counter_gt, instance_gt in enumerate(instances_gt):

        # create binary mask for current gt instance
        gt = mask_gt.copy()
        gt[mask_gt != instance_gt] = 0
        gt[mask_gt == instance_gt] = 1

        masks_row = []
        for counter_pred, instance_pred in enumerate(instances_pred):
            # make binary mask for current predicted instance
            prediction = mask_pred.copy()
            prediction[mask_pred != instance_pred] = 0
            prediction[mask_pred == instance_pred] = 1

            # calculate dice
            performance_matrix[counter_gt, counter_pred] = compute_dice_coefficient(gt, prediction)
            masks_row.append([gt, prediction])
        masks.append(masks_row)

    # assign instrument instances according to hungarian algorithm
    label_assignment = hungarian_algorithm(performance_matrix * -1)
    label_nr_gt, label_nr_pred = label_assignment

    # get performance per instance
    image_performance = []
    for i in range(len(label_nr_gt)):
        instance_dice = performance_matrix[label_nr_gt[i], label_nr_pred[i]]
        image_performance.append(instance_dice)

    missing_pred = np.absolute(len(instances_pred) - len(image_performance))
    missing_gt = np.absolute(len(instances_gt) - len(image_performance))
    n_missing = np.max([missing_gt, missing_pred])

    if n_missing > 0:
        for i in range(n_missing):
            image_performance.append(0)

    output = dict()
    for i, performance in enumerate(image_performance):
        if i > 0:
            output["instrument_{}".format(i - 1)] = performance
        else:
            output["background"] = performance

    return output


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
