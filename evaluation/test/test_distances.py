import unittest
import numpy as np
from imageio import imread

# single pixels, 2mm away
from evaluation.dice_calculations import compute_dice_coefficient
from evaluation.distance_calculations import compute_surface_distances, compute_surface_dice_at_tolerance, \
    compute_average_surface_distance, compute_robust_hausdorff, compute_surface_overlap_at_tolerance


class TestDiceCalculation(unittest.TestCase):

    def setUp(self):
        self.delta = 0.0005
        
    def test_surface_dice(self):
        path = "images\\img{}\\instrument_instances.png".format(3)

        # read image
        x = imread(path)

        # make image binary
        x[x < 0.5] = 0
        x[x >= 0.5] = 1

        mask_gt = np.reshape(x, x.shape + (1,))

        surface_distances = compute_surface_distances(mask_gt, mask_gt, (1, 1, 1))

    surface_dice = compute_surface_dice_at_tolerance(surface_distances, 1)
        
    def test_single_pixels_2mm_away(self):
        mask_gt = np.zeros((128, 128, 128), np.uint8)
        mask_pred = np.zeros((128, 128, 128), np.uint8)
        mask_gt[50, 60, 70] = 1
        mask_pred[50, 60, 72] = 1
        surface_distances = compute_surface_distances(mask_gt, mask_pred, spacing_mm=(3, 2, 1))
        surface_dice_1mm = compute_surface_dice_at_tolerance(surface_distances, 1)
        volumetric_dice = compute_dice_coefficient(mask_gt, mask_pred)
        print("surface dice at 1mm:      {}".format(surface_dice_1mm))
        print("volumetric dice:          {}".format(volumetric_dice))
        self.assertAlmostEqual(surface_dice_1mm, 0.5, delta=self.delta)
        self.assertAlmostEqual(volumetric_dice, 0.0, delta=self.delta)

    def test_two_cubes(self):
        # two cubes. cube 1 is 100x100x100 mm^3 and cube 2 is 102x100x100 mm^3
        mask_gt = np.zeros((100, 100, 100), np.uint8)
        mask_pred = np.zeros((100, 100, 100), np.uint8)
        spacing_mm = (2, 1, 1)
        mask_gt[0:50, :, :] = 1
        mask_pred[0:51, :, :] = 1
        surface_distances = compute_surface_distances(mask_gt, mask_pred, spacing_mm)
        expected_average_distance_gt_to_pred = 0.836145008498
        expected_volumetric_dice = 2. * 100 * 100 * 100 / (100 * 100 * 100 + 102 * 100 * 100)

        surface_dice_1mm = compute_surface_dice_at_tolerance(surface_distances, 1)
        volumetric_dice = compute_dice_coefficient(mask_gt, mask_pred)

        print("surface dice at 1mm:      {}".format(compute_surface_dice_at_tolerance(surface_distances, 1)))
        print("volumetric dice:          {}".format(compute_dice_coefficient(mask_gt, mask_pred)))

        self.assertAlmostEqual(surface_dice_1mm, expected_average_distance_gt_to_pred, delta=self.delta)
        self.assertAlmostEqual(volumetric_dice, expected_volumetric_dice, delta=self.delta)

    def test_empty_mask_in_pred(self):
        # test empty mask in prediction
        mask_gt = np.zeros((128, 128, 128), np.uint8)
        mask_pred = np.zeros((128, 128, 128), np.uint8)
        mask_gt[50, 60, 70] = 1
        # mask_pred[50,60,72] = 1

        surface_distances = compute_surface_distances(mask_gt, mask_pred, spacing_mm=(3, 2, 1))

        average_surface_distance = compute_average_surface_distance(surface_distances)
        hausdorf_100 = compute_robust_hausdorff(surface_distances, 100)
        hausdorf_95 = compute_robust_hausdorff(surface_distances, 95)

        surface_overlap_1_mm = compute_surface_overlap_at_tolerance(surface_distances, 1)
        surface_dice_1mm = compute_surface_dice_at_tolerance(surface_distances, 1)
        volumetric_dice = compute_dice_coefficient(mask_gt, mask_pred)

        print("average surface distance: {} mm".format(average_surface_distance))
        print("hausdorff (100%):         {} mm".format(hausdorf_100))
        print("hausdorff (95%):          {} mm".format(hausdorf_95))
        print("surface overlap at 1mm:   {}".format(surface_overlap_1_mm))
        print("surface dice at 1mm:      {}".format(surface_dice_1mm))
        print("volumetric dice:          {}".format(volumetric_dice))

        self.assertAlmostEqual(surface_dice_1mm, 0.0, delta=self.delta)
        self.assertAlmostEqual(volumetric_dice, 0.0, delta=self.delta)

    def test_empty_mask_in_gt(self):
        # test empty mask in ground truth
        mask_gt = np.zeros((128, 128, 128), np.uint8)
        mask_pred = np.zeros((128, 128, 128), np.uint8)
        # mask_gt[50,60,70] = 1
        mask_pred[50, 60, 72] = 1

        surface_distances = compute_surface_distances(mask_gt, mask_pred, spacing_mm=(3, 2, 1))

        average_surface_distance = compute_average_surface_distance(surface_distances)
        hausdorf_100 = compute_robust_hausdorff(surface_distances, 100)
        hausdorf_95 = compute_robust_hausdorff(surface_distances, 95)

        surface_overlap_1_mm = compute_surface_overlap_at_tolerance(surface_distances, 1)
        surface_dice_1mm = compute_surface_dice_at_tolerance(surface_distances, 1)
        volumetric_dice = compute_dice_coefficient(mask_gt, mask_pred)

        print("average surface distance: {} mm".format(average_surface_distance))
        print("hausdorff (100%):         {} mm".format(hausdorf_100))
        print("hausdorff (95%):          {} mm".format(hausdorf_95))
        print("surface overlap at 1mm:   {}".format(surface_overlap_1_mm))
        print("surface dice at 1mm:      {}".format(surface_dice_1mm))
        print("volumetric dice:          {}".format(volumetric_dice))

        self.assertAlmostEqual(surface_dice_1mm, 0.0, delta=self.delta)
        self.assertAlmostEqual(volumetric_dice, 0.0, delta=self.delta)

    def test_empty_mask_in_gt_and_pred(self):
        # test both masks empty
        mask_gt = np.zeros((128, 128, 128), np.uint8)
        mask_pred = np.zeros((128, 128, 128), np.uint8)
        # mask_gt[50,60,70] = 1
        # mask_pred[50,60,72] = 1
        surface_distances = compute_surface_distances(mask_gt, mask_pred, spacing_mm=(3, 2, 1))

        average_surface_distance = compute_average_surface_distance(surface_distances)
        hausdorf_100 = compute_robust_hausdorff(surface_distances, 100)
        hausdorf_95 = compute_robust_hausdorff(surface_distances, 95)

        surface_overlap_1_mm = compute_surface_overlap_at_tolerance(surface_distances, 1)
        surface_dice_1mm = compute_surface_dice_at_tolerance(surface_distances, 1)
        volumetric_dice = compute_dice_coefficient(mask_gt, mask_pred)

        print("average surface distance: {} mm".format(average_surface_distance))
        print("hausdorff (100%):         {} mm".format(hausdorf_100))
        print("hausdorff (95%):          {} mm".format(hausdorf_95))
        print("surface overlap at 1mm:   {}".format(surface_overlap_1_mm))
        print("surface dice at 1mm:      {}".format(surface_dice_1mm))
        print("volumetric dice:          {}".format(volumetric_dice))

        self.assertTrue(np.isnan(surface_dice_1mm))
        self.assertTrue(np.isnan(volumetric_dice))

if __name__ == '__main__':
    unittest.main()
