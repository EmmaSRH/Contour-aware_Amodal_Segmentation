import unittest
import numpy as np

from evaluation.mean_average_precision_calculations import compute_mean_average_precision, compute_iou, \
    compute_statistics


class TestMAPCalculation(unittest.TestCase):

    def test_intersection_over_union(self):
        # define ground truth
        gt_1 = np.array([[0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0]], np.uint8)

        gt_2 = np.array([[0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0]], np.uint8)

        # full intersection
        expected_iou_1 = 1
        pred_1 = np.array([[0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 1, 0, 0],
                           [0, 0, 1, 1, 1, 0, 0],
                           [0, 0, 1, 1, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0]], np.uint8)

        # one missing
        expected_iou_2 = 0.8888888
        pred_2 = np.array([[0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 1, 0, 0],
                           [0, 0, 1, 0, 1, 0, 0],
                           [0, 0, 1, 1, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0]], np.uint8)

        # just one intersected
        expected_iou_3 = 0.11111
        pred_3 = np.array([[0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0]], np.uint8)

        # no intersection
        expected_iou_4 = 0
        pred_4 = np.array([[0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 1, 1],
                           [0, 0, 0, 0, 1, 1, 1],
                           [0, 0, 0, 0, 1, 1, 1]], np.uint8)

        # empty prediction
        expected_iou_5 = 0
        pred_5 = np.array([[0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0]], np.uint8)

        delta = 0.0005
        self.assertAlmostEqual(compute_iou(mask_gt=gt_1, mask_pred=pred_1), expected_iou_1, delta=delta)
        self.assertAlmostEqual(compute_iou(mask_gt=gt_1, mask_pred=pred_2), expected_iou_2, delta=delta)
        self.assertAlmostEqual(compute_iou(mask_gt=gt_1, mask_pred=pred_3), expected_iou_3, delta=delta)
        self.assertAlmostEqual(compute_iou(mask_gt=gt_1, mask_pred=pred_4), expected_iou_4, delta=delta)
        self.assertAlmostEqual(compute_iou(mask_gt=gt_1, mask_pred=pred_5), expected_iou_5, delta=delta)
        self.assertAlmostEqual(compute_iou(mask_gt=gt_2, mask_pred=pred_4), expected_iou_4, delta=delta)
        self.assertAlmostEqual(compute_iou(mask_gt=gt_2, mask_pred=pred_5), expected_iou_5, delta=delta)
        self.assertTrue(np.isnan(compute_iou(mask_gt=pred_5, mask_pred=pred_5)))

    def test_detection_statistics(self):
        # define images
        img_1 = np.array([[0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0]], np.uint8)

        img_2 = np.array([[0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0]], np.uint8)

        img_3 = np.array([[1, 1, 0, 0, 0, 0, 0],
                          [1, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 2, 2, 0],
                          [0, 0, 0, 0, 2, 2, 0],
                          [0, 0, 0, 0, 0, 0, 0]], np.uint8)

        img_4 = np.array([[1, 1, 0, 0, 0, 0, 0],
                          [1, 1, 0, 0, 3, 3, 0],
                          [0, 0, 0, 0, 3, 3, 0],
                          [0, 4, 4, 0, 0, 0, 0],
                          [0, 4, 4, 0, 2, 2, 0],
                          [5, 5, 0, 0, 2, 2, 0],
                          [5, 5, 0, 0, 0, 0, 0]], np.uint8)

        # statistics
        # precision = true_positive / (true_positive + false_positive)
        # recall = true_positive / (true_positive + false_negative)
        result_test_case_1 = compute_statistics(mask_gt=img_1, mask_pred=img_1)
        expectation_test_case_1 = dict(
            true_positive=0,
            false_positive=0,
            false_negative=0,
            precision=0,
            recall=0
        )
        result_test_case_2 = compute_statistics(mask_gt=img_1, mask_pred=img_2)
        expectation_test_case_2 = dict(
            true_positive=0,
            false_positive=1,
            false_negative=0,
            precision=0,
            recall=0
        )
        result_test_case_3 = compute_statistics(mask_gt=img_2, mask_pred=img_1)
        expectation_test_case_3 = dict(
            true_positive=0,
            false_positive=0,
            false_negative=1,
            precision=0,
            recall=0
        )
        result_test_case_4 = compute_statistics(mask_gt=img_3, mask_pred=img_2)
        expectation_test_case_4 = dict(
            true_positive=0,
            false_positive=1,
            false_negative=2,
            precision=0,
            recall=0
        )
        result_test_case_5 = compute_statistics(mask_gt=img_3, mask_pred=img_4)
        expectation_test_case_5 = dict(
            true_positive=2,
            false_positive=3,
            false_negative=0,
            precision=2 / (2 + 3),
            recall=2 / (2 + 0)
        )
        result_test_case_6 = compute_statistics(mask_gt=img_4, mask_pred=img_4)  # expect fp=0, tp=2, fn=3
        expectation_test_case_6 = dict(
            true_positive=5,
            false_positive=0,
            false_negative=0,
            precision=1,
            recall=1
        )

        self.assertDictEqual(result_test_case_1, expectation_test_case_1)
        self.assertDictEqual(result_test_case_2, expectation_test_case_2)
        self.assertDictEqual(result_test_case_3, expectation_test_case_3)
        self.assertDictEqual(result_test_case_4, expectation_test_case_4)
        self.assertDictEqual(result_test_case_5, expectation_test_case_5)
        self.assertDictEqual(result_test_case_6, expectation_test_case_6)

    def test_mean_average_precision(self):
        statistics_list_1 = [
            dict(true_positive=0,
                 false_positive=0,
                 false_negative=0,
                 precision=1.0,
                 recall=0.2),
            dict(true_positive=0,
                 false_positive=0,
                 false_negative=0,
                 precision=1.0,
                 recall=0.4),
            dict(true_positive=0,
                 false_positive=0,
                 false_negative=0,
                 precision=0.67,
                 recall=0.4),
            dict(true_positive=0,
                 false_positive=0,
                 false_negative=0,
                 precision=0.5,
                 recall=0.4),
            dict(true_positive=0,
                 false_positive=0,
                 false_negative=0,
                 precision=0.4,
                 recall=0.4),
            dict(true_positive=0,
                 false_positive=0,
                 false_negative=0,
                 precision=0.5,
                 recall=0.6),
            dict(true_positive=0,
                 false_positive=0,
                 false_negative=0,
                 precision=0.57,
                 recall=0.8),
            dict(true_positive=0,
                 false_positive=0,
                 false_negative=0,
                 precision=0.5,
                 recall=0.8),
            dict(true_positive=0,
                 false_positive=0,
                 false_negative=0,
                 precision=0.44,
                 recall=0.8),
            dict(true_positive=0,
                 false_positive=0,
                 false_negative=0,
                 precision=0.5,
                 recall=1.0)
        ]

        statistics_list_2 = [
            dict(true_positive=0,
                 false_positive=0,
                 false_negative=0,
                 precision=1.0,
                 recall=0.090909),
            dict(true_positive=0,
                 false_positive=0,
                 false_negative=0,
                 precision=0.5,
                 recall=0.090909),
            dict(true_positive=0,
                 false_positive=0,
                 false_negative=0,
                 precision=0.666667,
                 recall=0.166667),
            dict(true_positive=0,
                 false_positive=0,
                 false_negative=0,
                 precision=0.75,
                 recall=0.230769),
            dict(true_positive=0,
                 false_positive=0,
                 false_negative=0,
                 precision=0.6,
                 recall=0.230769),
            dict(true_positive=0,
                 false_positive=0,
                 false_negative=0,
                 precision=0.666667,
                 recall=0.285714),
            dict(true_positive=0,
                 false_positive=0,
                 false_negative=0,
                 precision=0.714286,
                 recall=0.33333),
            dict(true_positive=0,
                 false_positive=0,
                 false_negative=0,
                 precision=0.75,
                 recall=0.375),
            dict(true_positive=0,
                 false_positive=0,
                 false_negative=0,
                 precision=0.66667,
                 recall=0.375),
            dict(true_positive=0,
                 false_positive=0,
                 false_negative=0,
                 precision=0.7,
                 recall=0.411765),
        ]

        statistics_list_3 = [
            dict(true_positive=0,
                 false_positive=0,
                 false_negative=0,
                 precision=1.0,
                 recall=0.33),
            dict(true_positive=0,
                 false_positive=0,
                 false_negative=0,
                 precision=0.5,
                 recall=0.33),
            dict(true_positive=0,
                 false_positive=0,
                 false_negative=0,
                 precision=0.67,
                 recall=0.67),
            dict(true_positive=0,
                 false_positive=0,
                 false_negative=0,
                 precision=0.5,
                 recall=0.67),
            dict(true_positive=0,
                 false_positive=0,
                 false_negative=0,
                 precision=0.4,
                 recall=0.67),
            dict(true_positive=0,
                 false_positive=0,
                 false_negative=0,
                 precision=0.5,
                 recall=1.0),
            dict(true_positive=0,
                 false_positive=0,
                 false_negative=0,
                 precision=0.43,
                 recall=1.0),
        ]

        delta = 0.0005
        self.assertAlmostEqual(compute_mean_average_precision(statistics_list_1), (0.4*1.0+0.4*0.57+0.2*0.5), delta=delta)
        self.assertAlmostEqual(compute_mean_average_precision(statistics_list_2), (0.09*1+0.285*0.75+0.625*0.7), delta=delta)
        self.assertAlmostEqual(compute_mean_average_precision(statistics_list_3), 0.33*1+0.34*0.67+0.33*0.5, delta=delta)


if __name__ == '__main__':
    unittest.main()