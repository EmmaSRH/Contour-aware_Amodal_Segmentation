import numpy as np
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

